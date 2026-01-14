from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import CrowdingSnapshot, Zone
from app.services.crowding import CrowdingService, crowding_color
from app.services.place_cache import PlaceCacheService

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

Category = Literal["cafe", "restaurant"]

KAKAO_BASE_URL = "https://dapi.kakao.com"
KAKAO_GROUP_CODE = {"cafe": "CE7", "restaurant": "FD6"}


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _get_latest_snapshot(db: Session, zone_code: str) -> CrowdingSnapshot | None:
    stmt = (
        select(CrowdingSnapshot)
        .where(CrowdingSnapshot.zone_code == zone_code)
        .order_by(desc(CrowdingSnapshot.ts))
        .limit(1)
    )
    return db.scalar(stmt)


def _is_fresh(row: CrowdingSnapshot, min_interval_s: int) -> bool:
    ts = row.ts
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    # ts는 tz-aware로 가정하고 utc로 비교
    age_s = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
    return age_s < float(min_interval_s)


def _kakao_rest_key() -> str:
    key = (os.getenv("KAKAO_REST_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("KAKAO_REST_API_KEY is missing")
    return key


def _kakao_category_search(
    *,
    category_group_code: str,
    lat: float,
    lng: float,
    radius_m: int,
    max_results: int,
) -> List[Dict[str, Any]]:
    """
    카카오 카테고리 검색을 페이지 돌려서 max_results까지 수집
    """
    url = f"{KAKAO_BASE_URL}/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    radius_m = int(max(0, min(radius_m, 20000)))
    max_results = int(max(1, min(max_results, 45)))

    docs_all: List[Dict[str, Any]] = []
    seen: set[str] = set()
    page = 1

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

        with httpx.Client(timeout=8.0, trust_env=False) as client:
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

    return docs_all


def _doc_to_place_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    def sf(v: Any, default: str = "") -> str:
        return default if v is None else str(v)

    def ff(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    return {
        "id": sf(d.get("id")),
        "name": sf(d.get("place_name")),
        "category_name": sf(d.get("category_name")),
        "category_group_code": sf(d.get("category_group_code")),
        "category_group_name": sf(d.get("category_group_name")),
        "phone": sf(d.get("phone")),
        "address_name": sf(d.get("address_name")),
        "road_address_name": sf(d.get("road_address_name")),
        "place_url": sf(d.get("place_url")),
        "lat": ff(d.get("y")),
        "lng": ff(d.get("x")),
        "distance_m": ff(d.get("distance")),
    }


@router.get("")
def recommend(
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    category: Category = Query(..., description="cafe | restaurant"),
    radius_m: int = Query(3000, ge=200, le=20000),
    max_results: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    “지금 덜 붐비는 곳 추천” 같은 원샷 추천 API
    """
    try:
        group_code = KAKAO_GROUP_CODE[category]
    except KeyError:
        raise HTTPException(status_code=400, detail="category must be cafe|restaurant")

    top_zones = int(os.getenv("TOP_ZONES", "5"))
    per_zone = int(os.getenv("PER_ZONE", "7"))
    zone_search_radius_m = int(os.getenv("ZONE_SEARCH_RADIUS_M", "700"))

    # 10~15분 주기 방향: 기본 600초(10분)
    min_interval_s = int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "600"))
    crowding = CrowdingService()

    # place_cache 재사용 정책
    place_cache_ttl_s = int(os.getenv("PLACE_CACHE_TTL_S", "3600"))  # 1h default
    place_cache_min_per_zone = int(os.getenv("PLACE_CACHE_MIN_PER_ZONE", str(per_zone)))

    zones: List[Zone] = list(db.scalars(select(Zone)).all())

    # 1) 사용자 반경 내 zone 후보
    cand: List[Tuple[Zone, float]] = []
    for z in zones:
        zlat = float(z.lat or 0.0)
        zlng = float(z.lng or 0.0)
        if zlat == 0.0 and zlng == 0.0:
            continue
        d = _haversine_m(lat, lng, zlat, zlng)
        if d <= float(radius_m):
            cand.append((z, d))

    if not cand:
        return {"items": []}

    cand.sort(key=lambda x: x[1])
    cand = cand[: max(10, top_zones * 3)]  

    # 2) zone별 crowding 스냅샷 확보 + 덜 붐비는 순 정렬
    zone_rows: List[Dict[str, Any]] = []
    wrote_any_snapshots = False

    for z, dist in cand:
        latest = _get_latest_snapshot(db, z.code)
        if latest and _is_fresh(latest, min_interval_s):
            level = latest.level or ""
            rank = int(latest.rank or 0)
            msg = latest.message or ""
            updated = int(latest.updated_at_epoch or 0)

            raw = latest.raw or {}
            color = (raw.get("color") or "").strip() or crowding_color(level)
        else:
            dto = crowding.get(area_name=z.name, area_code=z.code)

            level = dto.level
            rank = int(dto.rank or 0)
            color = dto.color
            msg = dto.message
            updated = int(dto.updated_at_epoch or 0)

            db.add(
                CrowdingSnapshot(
                    zone_code=z.code,
                    level=level,
                    rank=rank,
                    message=msg,
                    updated_at_epoch=updated,
                    raw={"color": color, "seoul": dto.raw},
                )
            )
            wrote_any_snapshots = True

        zone_rows.append(
            {
                "zone": z,
                "zone_dist_m": float(dist),
                "crowding_level": level,
                "crowding_rank": rank,
                "crowding_color": color,
                "crowding_message": msg,
                "crowding_updated_at": updated,
            }
        )

    # 덜 붐비는 순(=rank 높은 순) + 가까운 순
    zone_rows.sort(key=lambda r: (-int(r["crowding_rank"]), float(r["zone_dist_m"])))
    zone_rows = zone_rows[: int(top_zones)]

    if wrote_any_snapshots:
        db.commit()

    # 3) zone 순서대로 POI 모아서 추천 (place_cache 우선)
    out: List[Dict[str, Any]] = []
    seen_place_ids: set[str] = set()

    place_cache = PlaceCacheService(db)
    place_cache_dirty = False

    for row in zone_rows:
        z: Zone = row["zone"]

        # (A) DB 캐시 먼저 조회 (TTL 내 데이터만)
        cached = place_cache.get_cached_places_near(
            lat=float(z.lat),
            lng=float(z.lng),
            radius_m=zone_search_radius_m,
            category_group_code=group_code,
            limit=per_zone,
            ttl_s=place_cache_ttl_s,
        )

        use_cache = len(cached) >= min(place_cache_min_per_zone, per_zone)

        if use_cache:
            places = cached  # list[dict] (id/name/lat/lng/.../distance_m 포함)
        else:
            # 캐시가 부족하면 카카오 호출 (키가 없으면 캐시만으로 최대한 반환)
            try:
                docs = _kakao_category_search(
                    category_group_code=group_code,
                    lat=float(z.lat),
                    lng=float(z.lng),
                    radius_m=zone_search_radius_m,
                    max_results=per_zone,
                )
            except Exception as e:
                logger.exception("kakao category search failed: %s", e)
                docs = []

            for d in docs:
                try:
                    wrote = place_cache.upsert_from_kakao_doc(d)
                    if wrote:
                        place_cache_dirty = True
                except Exception as e:
                    logger.exception("place_cache upsert failed: %s", e)

            places = [_doc_to_place_dict(d) for d in docs]

        for place in places:
            pid = (place.get("id") or "").strip()
            if not pid or pid in seen_place_ids:
                continue
            seen_place_ids.add(pid)

            out.append(
                {
                    "place": {
                        "id": pid,
                        "name": place.get("name", ""),
                        "category_name": place.get("category_name", ""),
                        "category_group_code": place.get("category_group_code", ""),
                        "category_group_name": place.get("category_group_name", ""),
                        "phone": place.get("phone", ""),
                        "address_name": place.get("address_name", ""),
                        "road_address_name": place.get("road_address_name", ""),
                        "place_url": place.get("place_url", ""),
                        "lat": float(place.get("lat", 0.0)),
                        "lng": float(place.get("lng", 0.0)),
                        "distance_m": float(place.get("distance_m", 0.0)),
                    },
                    "zone": {
                        "code": z.code,
                        "name": z.name,
                        "lat": float(z.lat),
                        "lng": float(z.lng),
                        "distance_m": round(float(row["zone_dist_m"]), 1),
                        "crowding_level": row["crowding_level"],
                        "crowding_rank": int(row["crowding_rank"]),
                        "crowding_color": row["crowding_color"],
                        "crowding_message": row["crowding_message"],
                        "crowding_updated_at": int(row["crowding_updated_at"]),
                    },
                }
            )

            if len(out) >= int(max_results):
                break

        if len(out) >= int(max_results):
            break

    if place_cache_dirty:
        db.commit()

    return {"items": out}
