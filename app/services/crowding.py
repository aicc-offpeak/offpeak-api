cat > app/services/crowding.py <<'PY'
# app/services/crowding.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  

import os
from dataclasses import dataclass
from typing import Optional

from cachetools import TTLCache

from app.clients.seoul_city import SeoulCityClient, CityCrowding


@dataclass(frozen=True)
class CrowdingSnapshot:
    level: str
    rank: int           # 높을수록 덜 붐빔
    color: str          # green/yellow/orange/red/unknown
    message: str
    updated_at: int     # epoch seconds


def crowding_rank(level: str) -> int:
    m = {
        "여유": 4,
        "보통": 3,
        "약간 붐빔": 2,
        "붐빔": 1,
    }
    return m.get((level or "").strip(), 0)


def crowding_color(level: str) -> str:
    m = {
        "여유": "green",
        "보통": "yellow",
        "약간 붐빔": "orange",
        "붐빔": "red",
    }
    return m.get((level or "").strip(), "unknown")


class CrowdingService:
    def __init__(self) -> None:
        ttl = int(os.getenv("CROWDING_CACHE_TTL_S", "300"))
        self._cache: TTLCache[str, CrowdingSnapshot] = TTLCache(maxsize=2048, ttl=ttl)
        self._client = SeoulCityClient() 

    def get(self, area_nm: str) -> Optional[CrowdingSnapshot]:
        key = (area_nm or "").strip()
        if not key:
            return None

        if key in self._cache:
            return self._cache[key]

        city: Optional[CityCrowding] = self._client.get_citydata_ppltn(key)
        if not city:
            snap = CrowdingSnapshot(
                level="",
                rank=0,
                color="unknown",
                message="",
                updated_at=0,
            )
            self._cache[key] = snap
            return snap

        snap = CrowdingSnapshot(
            level=city.level,
            rank=crowding_rank(city.level),
            color=crowding_color(city.level),
            message=city.message,
            updated_at=city.updated_at_epoch,
        )
        self._cache[key] = snap
        return snap
PY
