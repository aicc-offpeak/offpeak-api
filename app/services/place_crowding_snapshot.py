from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import PlaceCrowdingSnapshot


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class PlaceCrowdingSnapshotService:
    """
    place_crowding_snapshots 기록 서비스
    - 너무 자주 쌓이는 걸 막기 위해 place_id 별 min_interval_s 내에는 skip
    """
    def __init__(self, db: Session) -> None:
        self.db = db
        self.min_interval_s = int(os.getenv("PLACE_CROWDING_SNAPSHOT_MIN_INTERVAL_S", "1800"))  # 기본 30분

    def _is_fresh(self, ts: Optional[datetime]) -> bool:
        if ts is None:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (_now_utc() - ts.astimezone(timezone.utc)).total_seconds()
        return age_s < float(self.min_interval_s)

    def _get_latest_ts(self, place_id: str) -> Optional[datetime]:
        stmt = (
            select(PlaceCrowdingSnapshot.ts)
            .where(PlaceCrowdingSnapshot.place_id == place_id)
            .order_by(desc(PlaceCrowdingSnapshot.ts))
            .limit(1)
        )
        return self.db.scalar(stmt)

    def record(
        self,
        *,
        place_id: str,
        place_name: str | None,
        category_group_code: str | None,
        lat: float | None,
        lng: float | None,
        zone_code: str,
        zone_distance_m: float | None,
        level: str,
        rank: int,
        message: str,
        updated_at_epoch: int,
        raw: dict[str, Any] | None = None,
    ) -> bool:
        place_id = (place_id or "").strip()
        if not place_id:
            return False

        latest = self._get_latest_ts(place_id)
        if self._is_fresh(latest):
            return False

        row = PlaceCrowdingSnapshot(
            place_id=place_id,
            place_name=(place_name or None),
            category_group_code=(category_group_code or None),
            lat=lat,
            lng=lng,
            zone_code=zone_code,
            zone_distance_m=zone_distance_m,
            level=level or "",
            rank=int(rank or 0),
            message=message or "",
            updated_at_epoch=int(updated_at_epoch or 0),
            raw=raw,
        )
        self.db.add(row)
        return True
