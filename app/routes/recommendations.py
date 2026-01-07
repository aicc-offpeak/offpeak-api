# app/routes/recommendations.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Literal

from fastapi import APIRouter, Query

from app.clients.kakao_local import KakaoLocalClient
from app.clients.seoul_citydata import SeoulCityDataClient
from app.services.recommender import OffpeakRecommender, load_zones_from_seed

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])

Category = Literal["cafe", "restaurant"]


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


@router.get("")
async def recommend(
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    category: Category = Query(..., description="cafe | restaurant"),
    radius_m: int = Query(3000, ge=200, le=20000),
    max_results: int = Query(20, ge=1, le=50),
) -> Dict[str, Any]:
    rec = _get_recommender()

    items = await rec.recommend_places(
        user_lat=lat,
        user_lng=lng,
        category=category,
        user_radius_m=radius_m,
        top_zones=int(os.getenv("TOP_ZONES", "5")),
        per_zone=int(os.getenv("PER_ZONE", "7")),
        max_results=max_results,
        zone_search_radius_m=int(os.getenv("ZONE_SEARCH_RADIUS_M", "700")),
    )
    return {"items": items}
