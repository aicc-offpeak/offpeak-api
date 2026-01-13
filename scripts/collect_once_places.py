from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

# -----------------------------
# Path + .env (MUST be before app imports)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env", override=True)

# -----------------------------
# App imports
# -----------------------------
from app.db import SessionLocal  # noqa: E402
from app.models import CrowdingSnapshot as CrowdingSnapshotRow, Zone  # noqa: E402
from app.services.crowding import CrowdingService, crowding_color  # noqa: E402
from app.services.place_cache import PlaceCacheService  # noqa: E402
from app.services.place_crowding_snapshot import PlaceCrowdingSnapshotService  # noqa: E402

KAKAO_BASE_URL = "https://dapi.kakao.com"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _kakao_rest_key() -> str:
    rest_key = (os.getenv("KAKAO_REST_API_KEY") or "").strip()
    if not rest_key:
        raise RuntimeError("KAKAO_REST_API_KEY is missing in environment variables.")
    return rest_key


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _jitter_point(lat: float, lng: float, jitter_m: int) -> Tuple[float, float]:
    """lat/lng 주변으로 랜덤 지터(미터)"""
    if jitter_m <= 0:
        return lat, lng
    dx = random.uniform(-jitter_m, jitter_m)
    dy = random.uniform(-jitter_m, jitter_m)

    lat_delta = dy / 111320.0
    cosv = math.cos(math.radians(lat)) or 1e-9
    lng_delta = dx / (111320.0 * cosv)
    return lat + lat_delta, lng + lng_delta


def kakao_category_search(
    *,
    category_group_code: str,
    lat: float,
    lng: float,
    radius_m: int,
    max_results: int,
    sleep_ms: int = 0,
) -> List[Dict[str, Any]]:
    """Kakao category 검색 (최대 45개 정도까지 페이지네이션)"""
    url = f"{KAKAO_BASE_URL}/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    radius_m = int(max(0, min(int(radius_m), 20000)))
    max_results = int(max(1, min(int(max_results), 45)))

    docs_all: List[Dict[str, Any]] = []
    seen: set[str] = set()
    page = 1

    while len(docs_all) < max_results and page <= 45:
        remaining = max_results - len(docs_all)
        size = min(15, remaining)

        params = {
            "category_group_code": (category_group_code or "").strip(),
            "x": f"{lng:.7f}",
            "y": f"{lat:.7f}",
            "radius": radius_m,
            "size": size,
            "sort": "distance",
            "page": page,
        }

        with httpx.Client(timeout=10.0, trust_env=False) as client:
            r = client.get(url, headers=headers, params=params)

        if r.status_code != 200:
            raise RuntimeError(f"Kakao category search failed: {r.status_code} {r.text}")

        data = r.json()
        docs = data.get("documents") or []
        meta = data.get("meta") or {}
        is_end = bool(meta.get("is_end", True))

        for d in docs:
            pid = str(d.get("id") or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            docs_all.append(d)
            if len(docs_all) >= max_results:
                break

        if is_end:
            break
        page += 1

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    return docs_all


def _get_latest_zone_snapshot(db: Session, zone_code: str) -> Optional[CrowdingSnapshotRow]:
    stmt = (
        select(CrowdingSnapshotRow)
        .where(CrowdingSnapshotRow.zone_code == zone_code)
        .order_by(desc(CrowdingSnapshotRow.ts))
        .limit(1)
    )
    return db.scalar(stmt)


def _is_fresh(ts: Optional[datetime], min_interval_s: int) -> bool:
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (_now_utc() - ts.astimezone(timezone.utc)).total_seconds()
    return age_s < float(min_interval_s)


def _nearest_zone(zones: List[Zone], lat: float, lng: float) -> Tuple[Zone, float]:
    best: Optional[Zone] = None
    best_d = 1e18
    for z in zones:
        zlat = float(z.lat or 0.0)
        zlng = float(z.lng or 0.0)
        if zlat == 0.0 and zlng == 0.0:
            continue
        d = _haversine_m(lat, lng, zlat, zlng)
        if d < best_d:
            best, best_d = z, d
    if best is None:
        raise RuntimeError("No valid zones in DB (coords missing?)")
    return best, float(best_d)


# -----------------------------
# Zone rotation (cursor)
# -----------------------------
def _load_cursor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"offset": 0, "shuffle": False, "seed": None, "order_hash": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"offset": 0, "shuffle": False, "seed": None, "order_hash": None}


def _save_cursor(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _order_hash(zone_codes: List[str]) -> str:
    # 간단한 해시(순서/개수 바뀌면 cursor 리셋 유도)
    s = "|".join(zone_codes)
    return str(abs(hash(s)))


def _pick_zones_with_rotation(
    zones_all: List[Zone],
    *,
    zones_per_run: int,
    cursor_file: Path,
    shuffle_zones: bool,
) -> Tuple[List[Zone], Dict[str, Any]]:
    zones_per_run = max(1, int(zones_per_run))

    zones_sorted = sorted(zones_all, key=lambda z: (z.code or ""))
    codes = [z.code for z in zones_sorted]
    oh = _order_hash(codes)

    cur = _load_cursor(cursor_file)

    # order가 달라졌거나 shuffle 설정이 달라졌으면 offset 리셋
    if cur.get("order_hash") != oh or bool(cur.get("shuffle")) != bool(shuffle_zones):
        cur = {"offset": 0, "shuffle": bool(shuffle_zones), "seed": None, "order_hash": oh}

    # shuffle 모드면 "고정 seed"로 항상 같은 랜덤 순서
    if shuffle_zones:
        seed = cur.get("seed")
        if seed is None:
            seed = random.randint(1, 2_147_483_647)
            cur["seed"] = seed
        rnd = random.Random(int(seed))
        idxs = list(range(len(zones_sorted)))
        rnd.shuffle(idxs)
        zones_ordered = [zones_sorted[i] for i in idxs]
    else:
        zones_ordered = zones_sorted

    n = len(zones_ordered)
    if n == 0:
        return [], cur

    offset = int(cur.get("offset") or 0) % n

    # wrap-around slice
    end = offset + zones_per_run
    if end <= n:
        picked = zones_ordered[offset:end]
    else:
        picked = zones_ordered[offset:n] + zones_ordered[0 : (end % n)]

    # 다음 실행을 위한 offset 업데이트
    cur["offset"] = (offset + zones_per_run) % n

    _save_cursor(cursor_file, cur)
    return picked, cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones-per-run", type=int, default=30)
    parser.add_argument("--seed-radius-m", type=int, default=3000)
    parser.add_argument("--per-zone", type=int, default=45)
    parser.add_argument("--jitter-m", type=int, default=600)
    parser.add_argument("--sleep-ms", type=int, default=150)
    parser.add_argument("--commit-every-zone", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--categories",
        type=str,
        default="CE7,FD6",
        help="comma separated category_group_code. ex) CE7,FD6",
    )
    parser.add_argument(
        "--zone-snapshot-min-interval-s",
        type=int,
        default=int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "900")),
    )

    # rotation options
    parser.add_argument(
        "--cursor-file",
        type=str,
        default=str(ROOT / ".zone_cursor"),
        help="cursor file path (stores offset/seed). default: .zone_cursor",
    )
    parser.add_argument(
        "--shuffle-zones",
        action="store_true",
        help="shuffle zone order once with a fixed seed (persisted in cursor file)",
    )

    args = parser.parse_args()

    # run-level random (jitter용). shuffle-zones는 별도 seed로 고정 처리됨.
    random.seed()

    categories = [c.strip().upper() for c in (args.categories or "").split(",") if c.strip()]
    if not categories:
        categories = ["CE7", "FD6"]

    cursor_path = Path(args.cursor_file)

    with SessionLocal() as db:
        zones_all = list(db.scalars(select(Zone)).all())
        zones, cursor = _pick_zones_with_rotation(
            zones_all,
            zones_per_run=int(args.zones_per_run),
            cursor_file=cursor_path,
            shuffle_zones=bool(args.shuffle_zones),
        )

        place_cache = PlaceCacheService(db)
        place_snap = PlaceCrowdingSnapshotService(db)
        crowding = CrowdingService()

        total_cache_writes = 0
        total_snap_added = 0
        total_zone_snap_writes = 0
        total_places_seen = 0
        total_errors = 0

        print(
            f"[collect-once] zones_picked={len(zones)} zones_total={len(zones_all)} "
            f"offset_next={cursor.get('offset')} shuffle={cursor.get('shuffle')} seed={cursor.get('seed')} "
            f"seed_radius_m={args.seed_radius_m} per_zone={args.per_zone} jitter_m={args.jitter_m} "
            f"sleep_ms={args.sleep_ms} categories={categories} dry_run={args.dry_run}"
        )

        for zi, z in enumerate(zones, start=1):
            try:
                zlat = float(z.lat or 0.0)
                zlng = float(z.lng or 0.0)
                if zlat == 0.0 and zlng == 0.0:
                    print(f"[skip] zone={z.code} name={z.name} (no coords)")
                    continue

                # (1) Kakao로 zone 주변 places 수집 (지터 포함)
                points = [(zlat, zlng)]
                points.append(_jitter_point(zlat, zlng, int(args.jitter_m)))
                points.append(_jitter_point(zlat, zlng, int(args.jitter_m)))

                docs_all: List[Dict[str, Any]] = []
                for (plat, plng) in points:
                    for cgc in categories:
                        docs = kakao_category_search(
                            category_group_code=cgc,
                            lat=plat,
                            lng=plng,
                            radius_m=int(args.seed_radius_m),
                            max_results=int(args.per_zone),
                            sleep_ms=int(args.sleep_ms),
                        )
                        docs_all.extend(docs)

                # 중복 제거 (place_id)
                uniq: Dict[str, Dict[str, Any]] = {}
                for d in docs_all:
                    pid = str(d.get("id") or "").strip()
                    if not pid:
                        continue
                    uniq[pid] = d

                docs = list(uniq.values())
                total_places_seen += len(docs)

                cache_writes = 0
                for d in docs:
                    try:
                        if place_cache.upsert_from_kakao_doc(d):
                            cache_writes += 1
                    except Exception:
                        pass

                total_cache_writes += cache_writes

                # (2) place_crowding_snapshots 기록: "place -> nearest zone" 매핑 후 zone crowding 값 저장
                snap_added = 0
                zone_snap_writes = 0

                zone_crowding_cache: Dict[str, Dict[str, Any]] = {}

                for d in docs:
                    pid = str(d.get("id") or "").strip()
                    if not pid:
                        continue
                    plat = _safe_float(d.get("y") or d.get("lat"))
                    plng = _safe_float(d.get("x") or d.get("lng"))
                    if plat == 0.0 and plng == 0.0:
                        continue

                    nz, dist_m = _nearest_zone(zones_all, plat, plng)

                    if nz.code in zone_crowding_cache:
                        zc = zone_crowding_cache[nz.code]
                    else:
                        latest = _get_latest_zone_snapshot(db, nz.code)
                        if latest and _is_fresh(latest.ts, int(args.zone_snapshot_min_interval_s)):
                            level = latest.level or ""
                            rank = int(latest.rank or 0)
                            msg = latest.message or ""
                            updated = int(latest.updated_at_epoch or 0)
                            raw = latest.raw or {}
                            color = (raw.get("color") or "").strip() or crowding_color(level)
                        else:
                            dto = crowding.get(area_code=nz.code, area_name=nz.name)
                            level = dto.level
                            rank = int(dto.rank or 0)
                            msg = dto.message
                            updated = int(dto.updated_at_epoch or 0)
                            color = dto.color or crowding_color(level)

                            db.add(
                                CrowdingSnapshotRow(
                                    zone_code=nz.code,
                                    level=level,
                                    rank=rank,
                                    message=msg,
                                    updated_at_epoch=updated,
                                    raw={"color": color, "seoul": dto.raw},
                                )
                            )
                            zone_snap_writes += 1

                        zc = {
                            "level": level,
                            "rank": rank,
                            "message": msg,
                            "updated": updated,
                            "color": color,
                        }
                        zone_crowding_cache[nz.code] = zc

                    added = place_snap.record(
                        place_id=pid,
                        place_name=_safe_str(d.get("place_name") or d.get("name")) or None,
                        category_group_code=_safe_str(d.get("category_group_code")) or None,
                        lat=plat,
                        lng=plng,
                        zone_code=nz.code,
                        zone_distance_m=float(dist_m),
                        level=zc["level"],
                        rank=int(zc["rank"]),
                        message=zc["message"],
                        updated_at_epoch=int(zc["updated"]),
                        raw={"color": zc["color"], "source": "zone"},
                    )
                    if added:
                        snap_added += 1

                total_snap_added += snap_added
                total_zone_snap_writes += zone_snap_writes

                if args.dry_run:
                    db.rollback()
                else:
                    if args.commit_every_zone:
                        db.commit()

                print(
                    f"[{zi:03d}/{len(zones)}] zone={z.code} {z.name} "
                    f"places={len(docs)} cache_writes={cache_writes} "
                    f"zone_snap_writes={zone_snap_writes} place_snap_added={snap_added}"
                )

            except Exception as e:
                total_errors += 1
                db.rollback()
                print(f"[err] zone={getattr(z, 'code', '?')} name={getattr(z, 'name', '?')} -> {e}")

        if (not args.dry_run) and (not args.commit_every_zone):
            db.commit()

        print(
            f"[done] places_seen={total_places_seen} cache_writes={total_cache_writes} "
            f"zone_snap_writes={total_zone_snap_writes} place_snap_added={total_snap_added} errors={total_errors}"
        )
        print(f"[cursor] file={cursor_path} next_offset={cursor.get('offset')} shuffle={cursor.get('shuffle')} seed={cursor.get('seed')}")


if __name__ == "__main__":
    main()
