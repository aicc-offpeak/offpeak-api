from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import CrowdingSnapshot as CrowdingSnapshotRow
from app.models import Zone
from app.services.crowding import CrowdingService, crowding_color, crowding_rank
from app.services.place_cache import PlaceCacheService

KAKAO_BASE_URL = "https://dapi.kakao.com"
router = APIRouter(prefix="/places", tags=["places"])


# -------------------------
# Models
# -------------------------
class PlaceItem(BaseModel):
    id: str
    name: str
    category_name: str = ""
    category_group_code: str = ""
    category_group_name: str = ""
    phone: str = ""
    address_name: str = ""
    road_address_name: str = ""
    place_url: str = ""
    image_url: str = ""
    lat: float
    lng: float
    distance_m: float = 0.0


class ZoneInfo(BaseModel):
    code: str
    name: str
    lat: float
    lng: float
    distance_m: float
    crowding_level: str = ""
    crowding_rank: int = 0
    crowding_color: str = "unknown"
    crowding_updated_at: int = 0
    crowding_message: str = ""


class PlaceWithZone(BaseModel):
    place: PlaceItem
    zone: ZoneInfo


class PlacesSearchResponse(BaseModel):
    items: List[PlaceItem] = Field(default_factory=list)


class PlacesInsightRequest(BaseModel):
    selected: PlaceItem
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None

    recommend_from: str = Field(default="selected", description="selected(기본) / user")
    radius_m: int = 1200
    max_candidates: int = 25
    max_alternatives: int = 3

    category: Optional[str] = None
    include_image: bool = True


class PlacesInsightResponse(BaseModel):
    selected: PlaceWithZone
    alternatives: List[PlaceWithZone] = Field(default_factory=list)


# -------------------------
# Utils
# -------------------------
def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _kakao_rest_key() -> str:
    rest_key = (os.getenv("KAKAO_REST_API_KEY") or "").strip()
    if not rest_key:
        raise RuntimeError("KAKAO_REST_API_KEY is missing in environment variables.")
    return rest_key


# -------------------------
# Kakao local calls
# -------------------------
def kakao_keyword_search(
    *,
    query: str,
    lat: float,
    lng: float,
    radius_m: int,
    size: int,
) -> List[Dict[str, Any]]:
    url = f"{KAKAO_BASE_URL}/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    radius_m = int(max(0, min(radius_m, 20000)))
    size = int(max(1, min(size, 15)))

    params = {
        "query": query,
        "x": f"{lng:.7f}",
        "y": f"{lat:.7f}",
        "radius": radius_m,
        "size": size,
        "sort": "distance",
        "page": 1,
    }

    with httpx.Client(timeout=8.0, trust_env=False) as client:
        r = client.get(url, headers=headers, params=params)

    if r.status_code != 200:
        raise RuntimeError(f"Kakao keyword search failed: {r.status_code} {r.text}")

    data = r.json()
    return data.get("documents") or []


def kakao_category_search_page(
    *,
    category_group_code: str,
    lat: float,
    lng: float,
    radius_m: int,
    size: int,
    page: int,
) -> tuple[List[Dict[str, Any]], bool]:
    url = f"{KAKAO_BASE_URL}/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    radius_m = int(max(0, min(radius_m, 20000)))
    size = int(max(1, min(size, 15)))
    page = int(max(1, min(page, 45)))

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
    return docs, is_end


def kakao_category_search(
    *,
    category_group_code: str,
    lat: float,
    lng: float,
    radius_m: int,
    max_results: int,
) -> List[Dict[str, Any]]:
    max_results = int(max(1, min(max_results, 45)))
    docs_all: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    page = 1
    while len(docs_all) < max_results and page <= 45:
        remaining = max_results - len(docs_all)
        page_size = min(15, remaining)

        docs, is_end = kakao_category_search_page(
            category_group_code=category_group_code,
            lat=lat,
            lng=lng,
            radius_m=radius_m,
            size=page_size,
            page=page,
        )

        for d in docs:
            pid = _safe_str(d.get("id"))
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            docs_all.append(d)
            if len(docs_all) >= max_results:
                break

        if is_end:
            break
        page += 1

    return docs_all


def _doc_to_place(d: Dict[str, Any]) -> PlaceItem:
    return PlaceItem(
        id=_safe_str(d.get("id")),
        name=_safe_str(d.get("place_name")),
        category_name=_safe_str(d.get("category_name")),
        category_group_code=_safe_str(d.get("category_group_code")),
        category_group_name=_safe_str(d.get("category_group_name")),
        phone=_safe_str(d.get("phone")),
        address_name=_safe_str(d.get("address_name")),
        road_address_name=_safe_str(d.get("road_address_name")),
        place_url=_safe_str(d.get("place_url")),
        image_url="",
        lat=_safe_float(d.get("y")),
        lng=_safe_float(d.get("x")),
        distance_m=_safe_float(d.get("distance")),
    )


# -------------------------
# place_url -> og:image
# -------------------------
_image_cache = TTLCache(maxsize=1024, ttl=int(os.getenv("PLACE_IMAGE_CACHE_TTL_S", "3600") or "3600"))
_OG_IMAGE_RE = re.compile(
    r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _extract_og_image(html: str) -> str:
    m = _OG_IMAGE_RE.search(html or "")
    return (m.group(1).strip() if m else "")


def _place_image_from_url(place_url: str) -> str:
    place_url = (place_url or "").strip()
    if not place_url:
        return ""
    if place_url in _image_cache:
        return _image_cache[place_url]

    try:
        with httpx.Client(timeout=6.0, trust_env=False, follow_redirects=True) as client:
            r = client.get(place_url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            _image_cache[place_url] = ""
            return ""
        img = _extract_og_image(r.text)
        _image_cache[place_url] = img
        return img
    except Exception:
        _image_cache[place_url] = ""
        return ""


def _maybe_enrich_image(p: PlaceItem, include: bool) -> PlaceItem:
    if not include:
        return p
    if p.image_url:
        return p
    img = _place_image_from_url(p.place_url)
    return p.model_copy(update={"image_url": img})


# -------------------------
# DB helpers: zone + crowding snapshot
# -------------------------
def _get_latest_snapshot(db: Session, zone_code: str) -> CrowdingSnapshotRow | None:
    stmt = (
        select(CrowdingSnapshotRow)
        .where(CrowdingSnapshotRow.zone_code == zone_code)
        .order_by(desc(CrowdingSnapshotRow.ts))
        .limit(1)
    )
    return db.scalar(stmt)


def _is_fresh(row: CrowdingSnapshotRow, min_interval_s: int) -> bool:
    ts = row.ts
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - ts).total_seconds()
    return age_s < float(min_interval_s)


def _nearest_zone(zones: List[Zone], lat: float, lng: float) -> Tuple[Zone, float]:
    best: Zone | None = None
    best_d = 1e18

    for z in zones:
        zlat = float(z.lat or 0.0)
        zlng = float(z.lng or 0.0)
        if zlat == 0.0 and zlng == 0.0:
            continue
        d = _haversine_m(lat, lng, zlat, zlng)
        if d < best_d:
            best = z
            best_d = d

    if best is None:
        raise RuntimeError("No valid zones in DB (coords missing?)")

    return best, float(best_d)


def _zone_info_for_point(
    *,
    db: Session,
    zones: List[Zone],
    lat: float,
    lng: float,
    crowding: CrowdingService,
    min_interval_s: int,
    snapshot_cache: Dict[str, ZoneInfo],
) -> tuple[ZoneInfo, bool]:
    """
    returns: (ZoneInfo, wrote_snapshot?)
    """
    z, dist = _nearest_zone(zones, lat, lng)

    if z.code in snapshot_cache:
        cached = snapshot_cache[z.code]
        # distance만 현재 포인트 기준으로 갱신
        return cached.model_copy(update={"distance_m": float(dist)}), False

    wrote = False
    latest = _get_latest_snapshot(db, z.code)

    if latest and _is_fresh(latest, min_interval_s):
        level = latest.level or ""
        rank = int(latest.rank or 0)
        msg = latest.message or ""
        updated = int(latest.updated_at_epoch or 0)
        raw = latest.raw or {}
        color = (raw.get("color") or "").strip() or crowding_color(level)
    else:
        dto = crowding.get(z.name)
        level = dto.level
        rank = int(dto.rank or 0)
        color = dto.color
        msg = dto.message
        updated = int(dto.updated_at_epoch or 0)

        db.add(
            CrowdingSnapshotRow(
                zone_code=z.code,
                level=level,
                rank=rank,
                message=msg,
                updated_at_epoch=updated,
                raw={"color": color},
            )
        )
        wrote = True

    info = ZoneInfo(
        code=z.code,
        name=z.name,
        lat=float(z.lat),
        lng=float(z.lng),
        distance_m=float(dist),
        crowding_level=level,
        crowding_rank=rank,
        crowding_color=color or "unknown",
        crowding_updated_at=updated,
        crowding_message=msg or "",
    )

    snapshot_cache[z.code] = info
    return info, wrote


# -------------------------
# Endpoints
# -------------------------
@router.get("/search", response_model=PlacesSearchResponse)
def search_places(
    query: str = Query(..., min_length=1, description="검색어(예: 스타벅스)"),
    lat: float = Query(..., ge=-90, le=90, description="현재 위도"),
    lng: float = Query(..., ge=-180, le=180, description="현재 경도"),
    size: int = Query(5, ge=1, le=15, description="가까운 순 결과 개수(기본 5)"),
    radius_m: int = Query(3000, ge=0, le=20000, description="검색 반경(m) (기본 3000)"),
    scope: str = Query(
        "food_cafe",
        description="검색 스코프: cafe(카페), food(음식점), food_cafe(기본), all(필터 없음)",
    ),
    category_group_code: Optional[str] = Query(
        None,
        description="카테고리 그룹 코드 필터 (카페=CE7, 음식점=FD6)",
    ),
    prefer_db: int = Query(0, ge=0, le=1, description="1이면 DB(place_cache) 우선 검색(캐시 테스트용)"),
    response: Response = None,
    db: Session = Depends(get_db),
) -> PlacesSearchResponse:
    """
    1) prefer_db=1이면 DB(place_cache)에서 먼저 검색 (TTL 내 캐시만)
       - 충분히 나오면 Kakao 호출 없이 반환
    2) 부족하면 Kakao 호출 -> 결과를 place_cache에 upsert -> 반환
    """
    cache = PlaceCacheService(db)

    valid_scopes = {"cafe", "food", "food_cafe", "all"}
    if scope not in valid_scopes:
        raise HTTPException(status_code=400, detail=f"invalid scope: {scope}")

    if category_group_code:
        allowed = {category_group_code}
    elif scope == "cafe":
        allowed = {"CE7"}
    elif scope == "food":
        allowed = {"FD6"}
    elif scope == "food_cafe":
        allowed = {"CE7", "FD6"}
    else:
        allowed = None

    # 1) DB 우선 모드
    if int(prefer_db) == 1:
        rows = cache.search_nearby_in_db(
            query=query,
            lat=lat,
            lng=lng,
            radius_m=int(radius_m),
            limit=int(size),
            allowed_category_group_codes=allowed,
        )
        if rows:
            items = []
            for r, d in rows:
                items.append(
                    PlaceItem(
                        id=r.place_id,
                        name=r.name or "",
                        category_name=r.category_name or "",
                        category_group_code=r.category_group_code or "",
                        category_group_name=r.category_group_name or "",
                        phone=r.phone or "",
                        address_name=r.address_name or "",
                        road_address_name=r.road_address_name or "",
                        place_url=r.place_url or "",
                        image_url="",
                        lat=float(r.lat or 0.0),
                        lng=float(r.lng or 0.0),
                        distance_m=float(round(d, 1)),
                    )
                )
            if response is not None:
                response.headers["X-Place-Cache"] = f"source=db hits={len(items)} writes=0"
            return PlacesSearchResponse(items=items)
        # DB에서 못 찾으면 Kakao로 폴백

    # 2) Kakao 호출
    try:
        docs = kakao_keyword_search(query=query, lat=lat, lng=lng, radius_m=radius_m, size=size)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"places search failed: {e}")

    # 카테고리 필터
    if allowed is not None:
        docs = [d for d in docs if (d.get("category_group_code") in allowed)]

    # DB upsert + 응답 변환
    wrote = 0
    items: List[PlaceItem] = []
    for d in docs:
        try:
            if cache.upsert_from_kakao_doc(d):
                wrote += 1
        except Exception:
            # 캐시 실패해도 검색은 성공해야 함
            pass
        items.append(_doc_to_place(d))

    if wrote:
        db.commit()

    if response is not None:
        response.headers["X-Place-Cache"] = f"source=kakao hits=0 writes={wrote}"

    return PlacesSearchResponse(items=items)


@router.post("/insight", response_model=PlacesInsightResponse)
def insight(req: PlacesInsightRequest, db: Session = Depends(get_db)) -> PlacesInsightResponse:
    if req.recommend_from not in {"selected", "user"}:
        raise HTTPException(status_code=400, detail="recommend_from must be 'selected' or 'user'")

    if req.recommend_from == "user" and req.user_lat is not None and req.user_lng is not None:
        center_lat = float(req.user_lat)
        center_lng = float(req.user_lng)
    else:
        center_lat = float(req.selected.lat)
        center_lng = float(req.selected.lng)

    # category_group_code 결정
    cgc = (req.selected.category_group_code or "").strip()
    if req.category:
        if req.category.lower() == "cafe":
            cgc = "CE7"
        elif req.category.lower() in {"food", "restaurant"}:
            cgc = "FD6"
    if not cgc:
        cgc = "CE7"

    zones: List[Zone] = list(db.scalars(select(Zone)).all())

    # 10~15분 주기: 기본 600초(10분)
    min_interval_s = int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "600"))
    crowding = CrowdingService()

    snapshot_cache: Dict[str, ZoneInfo] = {}
    wrote_any = False

    selected_zone, wrote = _zone_info_for_point(
        db=db,
        zones=zones,
        lat=req.selected.lat,
        lng=req.selected.lng,
        crowding=crowding,
        min_interval_s=min_interval_s,
        snapshot_cache=snapshot_cache,
    )
    wrote_any = wrote_any or wrote

    selected_place = _maybe_enrich_image(req.selected, req.include_image)
    selected = PlaceWithZone(place=selected_place, zone=selected_zone)

    # candidates
    try:
        docs = kakao_category_search(
            category_group_code=cgc,
            lat=center_lat,
            lng=center_lng,
            radius_m=int(req.radius_m),
            max_results=int(req.max_candidates),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"insight candidate search failed: {e}")

    candidates: List[PlaceWithZone] = []
    for d in docs:
        p = _doc_to_place(d)
        if not p.id or p.id == req.selected.id:
            continue

        p = _maybe_enrich_image(p, req.include_image)

        zinfo, wrote = _zone_info_for_point(
            db=db,
            zones=zones,
            lat=p.lat,
            lng=p.lng,
            crowding=crowding,
            min_interval_s=min_interval_s,
            snapshot_cache=snapshot_cache,
        )
        wrote_any = wrote_any or wrote
        candidates.append(PlaceWithZone(place=p, zone=zinfo))

    if wrote_any:
        db.commit()

    # alternatives selection:
    sel_rank = selected_zone.crowding_rank
    better = [x for x in candidates if x.zone.crowding_rank > sel_rank]
    pool = better if better else candidates

    pool.sort(key=lambda x: (-x.zone.crowding_rank, x.place.distance_m))
    alternatives = pool[: int(max(0, req.max_alternatives))]

    return PlacesInsightResponse(selected=selected, alternatives=alternatives)
