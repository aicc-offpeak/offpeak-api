# app/services/crowding.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache

from app.clients.seoul_citydata import SeoulCityDataClient


@dataclass(frozen=True)
class CrowdingSnapshot:
    level: str
    rank: int
    color: str
    message: str
    updated_at_epoch: int
    raw: dict[str, Any]  # SeoulCityDataClient record


def crowding_rank(level: str) -> int:
    m = {"여유": 4, "보통": 3, "약간 붐빔": 2, "붐빔": 1}
    return m.get((level or "").strip(), 0)


def crowding_color(level: str) -> str:
    m = {"여유": "green", "보통": "yellow", "약간 붐빔": "orange", "붐빔": "red"}
    return m.get((level or "").strip(), "unknown")


class CrowdingService:
    def __init__(self) -> None:
        ttl = int(os.getenv("CROWDING_CACHE_TTL_S", "900"))
        self._cache: TTLCache[str, CrowdingSnapshot] = TTLCache(maxsize=2048, ttl=ttl)
        self._client = SeoulCityDataClient()

    def get(self, *, area_name: str | None = None, area_code: str | None = None) -> CrowdingSnapshot:
        # 캐시 키는 code 우선
        key = (area_code or "").strip() or (area_name or "").strip()
        if not key:
            return CrowdingSnapshot("", 0, "unknown", "", 0, raw={})

        if key in self._cache:
            return self._cache[key]

        crowd = self._client.fetch_area_crowding(area_code=area_code, area_name=area_name)

        snap = CrowdingSnapshot(
            level=crowd.level or "",
            rank=crowding_rank(crowd.level),
            color=crowding_color(crowd.level),
            message=crowd.message or "",
            updated_at_epoch=int(crowd.updated_at or 0),
            raw=(crowd.raw or {}),
        )
        self._cache[key] = snap
        return snap
