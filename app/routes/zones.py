# app/routes/zones.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List

from fastapi import APIRouter, Query

from app.clients.kakao_local import KakaoLocalClient
from app.clients.seoul_citydata import SeoulCityDataClient
from app.services.recommender import OffpeakRecommender, load_zones_from_seed

router = APIRouter(prefix="/v1/zones", tags=["zones"])


@lru_cache(maxsize=1)
def _get_recommender() -> OffpeakRecommender:
    zones_path = os.getenv("ZONES_SEED_PATH", "app/resources/zones_seed.json")
    zones = load_zones_from_seed(zones_path)
    return OffpeakRecommender(
        kakao=KakaoLocalClient(),
        seoul=SeoulCityDataClient(),
        zones=zones,
        crowding_cache_ttl_s=int(os.getenv("CROWDING_CACHE_TTL_S", "300")),
    )


@router.get("/nearby")
async def nearby_zones(
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    radius_m: int = Query(3000, ge=200, le=20000),
    top_k: int = Query(8, ge=1, le=30),
) -> Dict[str, Any]:
    rec = _get_recommender()
    candidates = rec.nearby_zones(user_lat=lat, user_lng=lng, radius_m=radius_m, top_k=top_k)

    out: List[Dict[str, Any]] = []
    for z, dist in candidates:
        zc = await rec.get_zone_crowding(z)
        out.append(
            {
                "code": z.code,
                "name": z.name,
                "lat": z.lat,
                "lng": z.lng,
                "distance_m": round(dist, 1),
                "crowding_level": zc.level,
                "crowding_updated_at": zc.updated_at_epoch,
            }
        )

    return {"items": out}
