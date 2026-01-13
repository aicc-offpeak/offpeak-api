from __future__ import annotations

import argparse
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import CrowdingSnapshot, Zone
from app.services.crowding import CrowdingService, crowding_color
from app.services.place_cache import PlaceCacheService
from app.services.place_crowding_snapshot import PlaceCrowdingSnapshotService


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _kakao_rest_key() -> str:
    rest_key = (os.getenv("KAKAO_REST_API_KEY") or "").strip()
    if not rest_key:
        raise RuntimeError("KAKAO_REST_API_KEY is missing in environment variables.")
    return rest_key


def kakao_category_search(
    *,
    category_group_code: str,
    lat: float,
    lng: float,
    radius_m: int,
    max_results: int,
    sleep_ms: int = 120,
) -> List[Dict[str, Any]]:
    """
    Kakao category search (CE7/FD6)
    - max_results <= 45
    - 자동 페이지네이션
    """
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    radius_m = int(max(0, min(radius_m, 20000)))
    max_results = int(max(1, min(max_results, 45)))

    docs_all: List[Dict[str, Any]] = []
    seen: set[str] = set()
    page = 1

    with httpx.Client(timeout=10.0, trust_env=False) as client:
        while len(docs_all) < max_results and page <= 45:
            remaining = max_results - len(docs_all)
            size = min(15, remaining)

            params = {
                "category_group_code": category_group_code,
                "x": f"{lng:.7f}",
                "y": f"{lat:.7f}",
                "radius": radius_m,
                "size": size,
                "sort": "distance",
                "page": page,
            }

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


def _get_latest_snapshot(db: Session, zone_code: str) -> Optional[CrowdingSnapshot]:
    stmt = (
        select(CrowdingSnapshot)
        .where(CrowdingSnapshot.zone_code == zone_code)
        .order_by(desc(CrowdingSnapshot.ts))
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
            best = z
            best_d = d
    if best is None:
        raise RuntimeError("No valid zones (coords missing?)")
    return best, float(best_d)


def _get_zone_state(
    *,
    db: Session,
    crowding: CrowdingService,
    zone: Zone,
    min_interval_s: int,
    zone_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    zone_code별 crowding 상태 캐싱
    - DB 최신 스냅샷이 fresh면 재사용
    - 아니면 API 호출 + crowding_snapshots insert
    """
    if zone.code in zone_cache:
        return zone_cache[zone.code]

    latest = _get_latest_snapshot(db, zone.code)

    if latest and _is_fresh(latest.ts, min_interval_s):
        level = latest.level or ""
        rank = int(latest.rank or 0)
        message = latest.message or ""
        updated_at_epoch = int(latest.updated_at_epoch or 0)
        raw = latest.raw or {}
        color = (raw.get("color") or "").strip() or crowding_color(level)
        wrote = False
    else:
        dto = crowding.get(area_name=zone.name, area_code=zone.code)
        level = dto.level
        rank = int(dto.rank or 0)
        message = dto.message
        updated_at_epoch = int(dto.updated_at_epoch or 0)
        color = dto.color or crowding_color(level)
        db.add(
            CrowdingSnapshot(
                zone_code=zone.code,
                level=level,
                rank=rank,
                message=message or "",
                updated_at_epoch=updated_at_epoch,
                raw={"color": color, "seoul": dto.raw},
            )
        )
        wrote = True

    state = {
        "level": level,
        "rank": rank,
        "message": message,
        "updated_at_epoch": updated_at_epoch,
        "color": color,
        "wrote": wrote,
    }
    zone_cache[zone.code] = state
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones-per-run", type=int, default=20, help="한 번 실행에 처리할 zone 개수(회전)")
    parser.add_argument("--seed-radius-m", type=int, default=2500, help="place_cache 확장용 반경")
    parser.add_argument("--per-zone", type=int, default=30, help="zone당 카테고리별 최대 결과")
    parser.add_argument("--sleep-ms", type=int, default=120, help="Kakao 호출 사이 sleep(ms)")
    parser.add_argument("--jitter-m", type=int, default=400, help="zone 중심에서 랜덤 좌표 흔들기(m)")
    parser.add_argument("--categories", type=str, default="CE7,FD6", help="예: CE7,FD6")
    parser.add_argument("--min-zone-interval-s", type=int, default=int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "600")))
    parser.add_argument("--commit-every-zone", action="store_true", help="zone마다 커밋(기본: 마지막에 한번)")
    args = parser.parse_args()

    cats = [c.strip().upper() for c in (args.categories or "").split(",") if c.strip()]
    if not cats:
        cats = ["CE7", "FD6"]

    wrote_places = 0
    added_place_snaps = 0
    wrote_zone_snaps = 0

    crowding = CrowdingService()

    with SessionLocal() as db:
        zones_all = list(db.scalars(select(Zone).order_by(Zone.code)).all())
        if not zones_all:
            raise RuntimeError("zones table is empty")

        # zone 회전: 매번 랜덤 subset
        zones = random.sample(zones_all, k=min(int(args.zones_per_run), len(zones_all)))

        place_cache = PlaceCacheService(db)
        place_snap = PlaceCrowdingSnapshotService(db)
        zone_cache: dict[str, dict[str, Any]] = {}

        print(f"[run] zones={len(zones)} cats={cats} seed_radius_m={args.seed_radius_m} per_zone={args.per_zone}")

        for z in zones:
            # zone 좌표 jitter (더 다양한 place 잡기)
            zlat = float(z.lat or 0.0)
            zlng = float(z.lng or 0.0)
            if zlat == 0.0 and zlng == 0.0:
                continue

            # 아주 간단한 m->deg 근사
            jm = float(max(0, int(args.jitter_m)))
            dlat = (random.uniform(-jm, jm) / 111320.0) if jm else 0.0
            dlng = (random.uniform(-jm, jm) / (111320.0 * (math.cos(math.radians(zlat)) or 1e-9))) if jm else 0.0

            lat = zlat + dlat
            lng = zlng + dlng

            docs_all: List[Dict[str, Any]] = []
            for cgc in cats:
                docs = kakao_category_search(
                    category_group_code=cgc,
                    lat=lat,
                    lng=lng,
                    radius_m=int(args.seed_radius_m),
                    max_results=int(args.per_zone),
                    sleep_ms=int(args.sleep_ms),
                )
                docs_all.extend(docs)

            # place_cache upsert + place snapshot record
            for d in docs_all:
                pid = str(d.get("id") or "").strip()
                if not pid:
                    continue

                # 1) upsert place_cache
                try:
                    if place_cache.upsert_from_kakao_doc(d):
                        wrote_places += 1
                except Exception:
                    pass

                # 2) place snapshot: nearest zone로 매핑
                plat = float(d.get("y") or 0.0)
                plng = float(d.get("x") or 0.0)
                if plat == 0.0 and plng == 0.0:
                    continue

                nz, nz_dist = _nearest_zone(zones_all, plat, plng)
                st = _get_zone_state(
                    db=db,
                    crowding=crowding,
                    zone=nz,
                    min_interval_s=int(args.min_zone_interval_s),
                    zone_cache=zone_cache,
                )
                if st.get("wrote"):
                    wrote_zone_snaps += 1

                try:
                    ok = place_snap.record(
                        place_id=pid,
                        place_name=str(d.get("place_name") or d.get("name") or "") or None,
                        category_group_code=str(d.get("category_group_code") or "") or None,
                        lat=plat,
                        lng=plng,
                        zone_code=nz.code,
                        zone_distance_m=float(nz_dist),
                        level=str(st.get("level") or ""),
                        rank=int(st.get("rank") or 0),
                        message=str(st.get("message") or ""),
                        updated_at_epoch=int(st.get("updated_at_epoch") or 0),
                        raw={"color": st.get("color"), "source": "zone"},
                    )
                    if ok:
                        added_place_snaps += 1
                except Exception:
                    pass

            if args.commit_every_zone:
                db.commit()

        db.commit()

    print(f"[done] wrote_places={wrote_places} wrote_zone_snaps={wrote_zone_snaps} added_place_snaps={added_place_snaps}")


if __name__ == "__main__":
    # .env는 프로젝트 루트에서 실행할 거라 가정
    load_dotenv(".env", override=True)
    main()
