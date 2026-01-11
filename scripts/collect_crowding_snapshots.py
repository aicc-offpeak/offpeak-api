from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import desc, select

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
from app.models import CrowdingSnapshot as CrowdingSnapshotModel, Zone  # noqa: E402
from app.services.crowding import CrowdingService  # noqa: E402


def _is_fresh(ts: datetime, min_interval_s: int) -> bool:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
    return age_s < float(min_interval_s)


def _get_latest_snapshot_ts(db, zone_code: str) -> datetime | None:
    stmt = (
        select(CrowdingSnapshotModel.ts)
        .where(CrowdingSnapshotModel.zone_code == zone_code)
        .order_by(desc(CrowdingSnapshotModel.ts))
        .limit(1)
    )
    return db.scalar(stmt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--min-interval-s",
        type=int,
        default=int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "900")),
        help="Skip if latest snapshot is newer than this interval (seconds)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    crowding = CrowdingService()

    wrote = 0
    skipped = 0
    errors = 0

    with SessionLocal() as db:
        zones = list(db.scalars(select(Zone).order_by(Zone.code).limit(int(args.limit))).all())
        print(f"[collect] zones_to_check={len(zones)} min_interval_s={args.min_interval_s} dry_run={args.dry_run}")

        for z in zones:
            try:
                latest_ts = _get_latest_snapshot_ts(db, z.code)
                if latest_ts and _is_fresh(latest_ts, int(args.min_interval_s)):
                    skipped += 1
                    continue

                # FIX: keyword call (area_code + area_name)
                dto = crowding.get(area_code=z.code, area_name=z.name)

                if args.dry_run:
                    print(f"[dry] zone={z.code} name={z.name} level={dto.level} rank={dto.rank}")
                    continue

                db.add(
                    CrowdingSnapshotModel(
                        zone_code=z.code,
                        level=dto.level,
                        rank=int(dto.rank or 0),
                        message=dto.message,
                        updated_at_epoch=int(dto.updated_at_epoch or 0),
                        raw={"color": dto.color, "seoul": dto.raw},
                    )
                )
                db.commit()
                wrote += 1
                print(f"[ok] inserted snapshot: zone={z.code} name={z.name} level={dto.level} rank={dto.rank}")

            except Exception as e:
                db.rollback()
                errors += 1
                print(f"[err] zone={z.code} name={z.name} -> {e}")

    print(f"[done] wrote={wrote} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
