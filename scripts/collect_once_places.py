from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

# ---------------------------------
# Path + .env (MUST be before app imports)
# ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env", override=True)

# ---------------------------------
# App imports
# ---------------------------------
from app.db import SessionLocal  # noqa: E402
from app.models import CrowdingSnapshot as CrowdingSnapshotModel  # noqa: E402
from app.models import PlaceCache as PlaceCacheModel  # noqa: E402
from app.models import Zone  # noqa: E402
from app.services.crowding import CrowdingService, crowding_color  # noqa: E402
from app.services.place_crowding_snapshot import PlaceCrowdingSnapshotService  # noqa: E402


# -------------------------
# Utils
# -------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math

    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _cursor_read(path: Path) -> int:
    try:
        s = path.read_text(encoding="utf-8").strip()
        return int(s)
    except Exception:
        return 0


def _cursor_write(path: Path, v: int) -> None:
    try:
        path.write_text(str(int(v)), encoding="utf-8")
    except Exception:
        pass


def _pick_zone_batch(db: Session, *, zones_per_run: int, cursor_file: Path) -> Tuple[List[Zone], Dict[str, Any]]:
    zones = list(db.scalars(select(Zone).order_by(Zone.code)).all())
    n_total = len(zones)
    if n_total == 0:
        return [], {"total": 0, "start": 0, "next": 0}

    zones_per_run = max(1, min(int(zones_per_run), n_total))

    start = _cursor_read(cursor_file) % n_total
    batch = zones[start : start + zones_per_run]
    if len(batch) < zones_per_run:
        batch += zones[0 : (zones_per_run - len(batch))]

    nxt = (start + zones_per_run) % n_total
    _cursor_write(cursor_file, nxt)

    return batch, {"total": n_total, "start": start, "next": nxt}


def _is_fresh_ts(ts: Optional[datetime], min_interval_s: int) -> bool:
    if not ts:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (_now_utc() - ts.astimezone(timezone.utc)).total_seconds()
    return age_s < float(min_interval_s)


@dataclass
class ZoneSnap:
    level: str
    rank: int
    message: str
    updated_at_epoch: int
    color: str


def _get_or_fetch_zone_snapshot(db: Session, *, z: Zone, crowding: CrowdingService, min_interval_s: int) -> Tuple[ZoneSnap, bool]:
    latest: CrowdingSnapshotModel | None = db.scalar(
        select(CrowdingSnapshotModel)
        .where(CrowdingSnapshotModel.zone_code == z.code)
        .order_by(desc(CrowdingSnapshotModel.ts))
        .limit(1)
    )

    if latest and _is_fresh_ts(latest.ts, min_interval_s):
        raw = latest.raw or {}
        color = (raw.get("color") or "").strip() or crowding_color(latest.level or "")
        return (
            ZoneSnap(
                level=latest.level or "",
                rank=int(latest.rank or 0),
                message=latest.message or "",
                updated_at_epoch=int(latest.updated_at_epoch or 0),
                color=color or "unknown",
            ),
            False,
        )

    dto = crowding.get(area_code=z.code, area_name=z.name)
    db.add(
        CrowdingSnapshotModel(
            zone_code=z.code,
            level=dto.level,
            rank=int(dto.rank or 0),
            message=dto.message or "",
            updated_at_epoch=int(dto.updated_at_epoch or 0),
            raw={"color": dto.color, "seoul": dto.raw},
        )
    )
    return (
        ZoneSnap(
            level=dto.level or "",
            rank=int(dto.rank or 0),
            message=dto.message or "",
            updated_at_epoch=int(dto.updated_at_epoch or 0),
            color=(dto.color or "").strip() or "unknown",
        ),
        True,
    )


def _tracked_places_for_zone(
    db: Session,
    *,
    z: Zone,
    radius_m: int,
    limit: int,
    prelimit: int = 400,
) -> List[Tuple[PlaceCacheModel, float]]:
    """
    place_cache에서 zone 중심 기준 '가까운 순' 상위 limit개 뽑기
    - bbox + (lat/lng 단순 거리)로 prelimit 추린 뒤
    - haversine로 radius_m 필터 + 정렬
    """
    radius_m = int(max(100, min(int(radius_m), 20000)))
    limit = int(max(1, min(int(limit), 50)))
    prelimit = int(max(limit, min(int(prelimit), 2000)))

    lat0 = float(z.lat or 0.0)
    lng0 = float(z.lng or 0.0)
    if lat0 == 0.0 and lng0 == 0.0:
        return []

    # bbox(대략)
    lat_delta = radius_m / 111320.0
    import math

    cosv = math.cos(math.radians(lat0)) or 1e-9
    lng_delta = radius_m / (111320.0 * cosv)

    # SQL 상에서 대충 가까운 후보(prelimit)만 먼저 뽑기
    dist2 = func.pow(PlaceCacheModel.lat - lat0, 2) + func.pow(PlaceCacheModel.lng - lng0, 2)

    stmt = (
        select(PlaceCacheModel)
        .where(PlaceCacheModel.lat.between(lat0 - lat_delta, lat0 + lat_delta))
        .where(PlaceCacheModel.lng.between(lng0 - lng_delta, lng0 + lng_delta))
        .order_by(dist2.asc())
        .limit(prelimit)
    )

    cand = list(db.scalars(stmt).all())

    out: List[Tuple[PlaceCacheModel, float]] = []
    for r in cand:
        plat = float(r.lat or 0.0)
        plng = float(r.lng or 0.0)
        d = _haversine_m(lat0, lng0, plat, plng)
        if d <= float(radius_m):
            out.append((r, d))

    out.sort(key=lambda x: x[1])
    return out[:limit]


def run_profile_once(
    *,
    zones_per_run: int,
    seed_radius_m: int,
    tracked_per_zone: int,
    sleep_ms: int,
    commit_every_zone: bool,
    cursor_path: Path,
) -> None:
    crowding = CrowdingService()

    # zone snapshot은 너무 자주 안 찍어도 되게(기본 10분)
    zone_min_interval_s = int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "600"))

    wrote_zone = 0
    wrote_place = 0
    skipped_zone = 0
    errors = 0

    with SessionLocal() as db:
        batch, meta = _pick_zone_batch(db, zones_per_run=zones_per_run, cursor_file=cursor_path)
        print(f"[profile] zones_per_run={zones_per_run} tracked_per_zone={tracked_per_zone} radius_m={seed_radius_m}")
        print(f"[cursor] total_zones={meta['total']} start={meta['start']} next={meta['next']} cursor_file={cursor_path.name}")

        place_snap = PlaceCrowdingSnapshotService(db)

        for z in batch:
            try:
                zsnap, wrote = _get_or_fetch_zone_snapshot(db, z=z, crowding=crowding, min_interval_s=zone_min_interval_s)
                if wrote:
                    wrote_zone += 1
                else:
                    skipped_zone += 1

                tracked = _tracked_places_for_zone(db, z=z, radius_m=seed_radius_m, limit=tracked_per_zone)
                if not tracked:
                    if commit_every_zone:
                        db.commit()
                    continue

                added_here = 0
                for row, dist_m in tracked:
                    ok = place_snap.record(
                        place_id=str(row.place_id),
                        place_name=row.name,
                        category_group_code=row.category_group_code,
                        lat=float(row.lat or 0.0),
                        lng=float(row.lng or 0.0),
                        zone_code=z.code,
                        zone_distance_m=float(dist_m),
                        level=zsnap.level,
                        rank=int(zsnap.rank or 0),
                        message=zsnap.message,
                        updated_at_epoch=int(zsnap.updated_at_epoch or 0),
                        raw={"color": zsnap.color, "source": "zone", "mode": "profile"},
                    )
                    if ok:
                        wrote_place += 1
                        added_here += 1

                if commit_every_zone:
                    db.commit()

                print(f"[zone] {z.code} {z.name} tracked={len(tracked)} added={added_here} level={zsnap.level} rank={zsnap.rank}")

                if sleep_ms > 0:
                    time.sleep(float(sleep_ms) / 1000.0)

            except Exception as e:
                db.rollback()
                errors += 1
                print(f"[err] zone={z.code} name={z.name} -> {e}")

        if not commit_every_zone:
            db.commit()

    print(f"[done] wrote_zone={wrote_zone} skipped_zone={skipped_zone} wrote_place={wrote_place} errors={errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile"], default="profile")

    parser.add_argument("--zones-per-run", type=int, default=30)
    parser.add_argument("--seed-radius-m", type=int, default=3000)

    # 기본 15개
    parser.add_argument("--tracked-per-zone", type=int, default=15)

    parser.add_argument("--sleep-ms", type=int, default=150)
    parser.add_argument("--commit-every-zone", action="store_true")
    parser.add_argument("--cursor-file", type=str, default=".zone_cursor")

    args = parser.parse_args()

    cursor_path = (ROOT / args.cursor_file).resolve()

    run_profile_once(
        zones_per_run=int(args.zones_per_run),
        seed_radius_m=int(args.seed_radius_m),
        tracked_per_zone=int(args.tracked_per_zone),
        sleep_ms=int(args.sleep_ms),
        commit_every_zone=bool(args.commit_every_zone),
        cursor_path=cursor_path,
    )


if __name__ == "__main__":
    main()
