# app/services/recommender.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from cachetools import TTLCache

from app.clients.kakao_local import KakaoLocalClient
from app.clients.seoul_citydata import SeoulCityDataClient


Category = Literal["cafe", "restaurant"]

CATEGORY_TO_KAKAO_CODE = {
    "cafe": "CE7",
    "restaurant": "FD6",
}

# crowding rank (bigger = better)
CROWDING_RANK = {
    "여유": 4,
    "보통": 3,
    "약간 붐빔": 2,
    "붐빔": 1,
    "정보없음": 0,
}

@dataclass(frozen=True)
class Zone:
    code: str
    name: str
    lat: float
    lng: float


@dataclass(frozen=True)
class ZoneCrowding:
    level: str
    updated_at_epoch: int
    raw: Dict[str, Any]


class OffpeakRecommender:
    def __init__(
        self,
        *,
        kakao: KakaoLocalClient,
        seoul: SeoulCityDataClient,
        zones: List[Zone],
        crowding_cache_ttl_s: int = 300,   # 5 minutes
        crowding_cache_maxsize: int = 512,
    ) -> None:
        self.kakao = kakao
        self.seoul = seoul
        self.zones = zones
        self._crowding_cache: TTLCache[str, ZoneCrowding] = TTLCache(
            maxsize=crowding_cache_maxsize,
            ttl=crowding_cache_ttl_s,
        )

    async def get_zone_crowding(self, zone: Zone) -> ZoneCrowding:
        cached = self._crowding_cache.get(zone.code)
        if cached:
            return cached

        # SeoulCityDataClient는 동기 클라이언트이므로 그대로 호출 (I/O 블로킹 허용)
        sc = self.seoul.fetch_area_crowding(area_name=zone.name)

        # Use a lightweight timestamp (epoch seconds) without extra deps
        import time
        zc = ZoneCrowding(level=str(sc.level), updated_at_epoch=int(time.time()), raw=sc.raw)
        self._crowding_cache[zone.code] = zc
        return zc

    def nearby_zones(
        self,
        *,
        user_lat: float,
        user_lng: float,
        radius_m: int = 3000,
        top_k: int = 8,
    ) -> List[Tuple[Zone, float]]:
        scored: List[Tuple[Zone, float]] = []
        for z in self.zones:
            d = haversine_m(user_lat, user_lng, z.lat, z.lng)
            if d <= radius_m:
                scored.append((z, d))

        # if none in radius, fallback to closest
        if not scored:
            scored = [(z, haversine_m(user_lat, user_lng, z.lat, z.lng)) for z in self.zones]

        scored.sort(key=lambda x: x[1])
        return scored[: max(1, top_k)]

    async def recommend_places(
        self,
        *,
        user_lat: float,
        user_lng: float,
        category: Category,
        user_radius_m: int = 3000,
        top_zones: int = 5,
        per_zone: int = 7,
        max_results: int = 20,
        zone_search_radius_m: int = 700,
    ) -> List[Dict[str, Any]]:
        kakao_code = CATEGORY_TO_KAKAO_CODE[category]

        # 1) candidate zones
        candidates = self.nearby_zones(
            user_lat=user_lat,
            user_lng=user_lng,
            radius_m=user_radius_m,
            top_k=max(top_zones * 2, 8),
        )

        # 2) fetch crowding + score
        zone_rows: List[Dict[str, Any]] = []
        for z, z_dist in candidates:
            zc = await self.get_zone_crowding(z)
            rank = CROWDING_RANK.get(zc.level, 0)
            # score: prefer low crowding(=high rank) and nearer distance
            score = (rank * 100000) - z_dist
            zone_rows.append(
                {
                    "zone": z,
                    "zone_distance_m": z_dist,
                    "crowding_level": zc.level,
                    "crowding_rank": rank,
                    "score": score,
                    "crowding_updated_at": zc.updated_at_epoch,
                }
            )

        zone_rows.sort(key=lambda r: r["score"], reverse=True)
        selected = zone_rows[: max(1, top_zones)]

        # 3) query Kakao around each selected zone center
        merged: Dict[str, Dict[str, Any]] = {}  # key: kakao place id
        for zr in selected:
            z: Zone = zr["zone"]
            docs = await self.kakao.search_category(
                category_group_code=kakao_code,
                x=z.lng,  # Kakao uses x=lng
                y=z.lat,  # y=lat
                radius_m=zone_search_radius_m,
                size=min(15, max(1, per_zone)),
                page=1,
                sort="distance",
            )

            for d in docs:
                pid = str(d.get("id") or "")
                if not pid:
                    continue

                place_lat = _safe_float(d.get("y"))
                place_lng = _safe_float(d.get("x"))
                user_dist = haversine_m(user_lat, user_lng, place_lat, place_lng)

                item = {
                    "id": pid,
                    "name": d.get("place_name"),
                    "category_name": d.get("category_name"),
                    "category_code": kakao_code,
                    "phone": d.get("phone"),
                    "address_name": d.get("address_name"),
                    "road_address_name": d.get("road_address_name"),
                    "place_url": d.get("place_url"),
                    "lat": place_lat,
                    "lng": place_lng,
                    "distance_m": round(user_dist, 1),
                    "zone": {
                        "code": z.code,
                        "name": z.name,
                        "lat": z.lat,
                        "lng": z.lng,
                        "distance_m": round(float(zr["zone_distance_m"]), 1),
                        "crowding_level": zr["crowding_level"],
                        "crowding_rank": zr["crowding_rank"],
                        "crowding_updated_at": zr["crowding_updated_at"],
                    },
                }

                # dedupe: keep best (less crowded zone, then nearer)
                prev = merged.get(pid)
                if not prev:
                    merged[pid] = item
                else:
                    prev_rank = int(prev["zone"]["crowding_rank"])
                    new_rank = int(item["zone"]["crowding_rank"])
                    if new_rank > prev_rank:
                        merged[pid] = item
                    elif new_rank == prev_rank and item["distance_m"] < prev["distance_m"]:
                        merged[pid] = item

        # 4) finalize sort
        results = list(merged.values())
        results.sort(
            key=lambda x: (
                -int(x["zone"]["crowding_rank"]),
                float(x["distance_m"]),
            )
        )
        return results[: max(1, max_results)]


def load_zones_from_seed(path: str) -> List[Zone]:
    """
    Seed JSON example:
    [
      {"code":"Z001","name":"광화문·덕수궁","lat":37.571,"lng":126.976},
      ...
    ]
    """
    if not os.path.exists(path):
        # fallback minimal seed to avoid crash
        return [
            Zone(code="Z001", name="광화문·덕수궁", lat=37.571, lng=126.976),
            Zone(code="Z002", name="홍대입구역", lat=37.557, lng=126.925),
        ]

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    zones: List[Zone] = []
    for r in raw:
        try:
            zones.append(
                Zone(
                    code=str(r["code"]),
                    name=str(r["name"]),
                    lat=float(r["lat"]),
                    lng=float(r["lng"]),
                )
            )
        except Exception:
            continue
    return zones


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    # Earth radius in meters
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0
