from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import desc, select, func

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
        zone_codes = [z.code for z in zones]
        print(f"[collect] zones_to_check={len(zones)} min_interval_s={args.min_interval_s} dry_run={args.dry_run}")

        # 1) 최신 스냅샷 ts를 zone별로 한 번에 가져오기 (N번 SELECT -> 1번)
        latest_rows = db.execute(
            select(
                CrowdingSnapshotModel.zone_code,
                func.max(CrowdingSnapshotModel.ts).label("max_ts"),
            )
            .where(CrowdingSnapshotModel.zone_code.in_(zone_codes))
            .group_by(CrowdingSnapshotModel.zone_code)
        ).all()
        latest_map = {zc: ts for (zc, ts) in latest_rows}

        for z in zones:
            try:
                latest_ts = latest_map.get(z.code)
                if latest_ts and _is_fresh(latest_ts, int(args.min_interval_s)):
                    skipped += 1
                    continue

                dto = crowding.get(area_code=z.code, area_name=z.name)

                if args.dry_run:
                    print(f"[dry] zone={z.code} name={z.name} level={dto.level} rank={dto.rank}")
                    continue

                # 2) zone 단위 savepoint(부분 롤백)로 안정성 확보
                with db.begin_nested():
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

                wrote += 1
                print(f"[ok] staged snapshot: zone={z.code} name={z.name} level={dto.level} rank={dto.rank}")

            except Exception as e:
                # begin_nested() 안에서 실패하면 그 부분만 롤백되고 계속 진행됨
                errors += 1
                print(f"[err] zone={z.code} name={z.name} -> {e}")

        # 3) 마지막에 한 번만 commit (속도↑)
        if not args.dry_run:
            if wrote:
                db.commit()
            else:
                db.rollback()

    print(f"[done] wrote={wrote} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
