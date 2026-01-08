from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.clients.seoul_citydata import SeoulCityDataClient

KAKAO_BASE_URL = "https://dapi.kakao.com"

router = APIRouter(prefix="/places", tags=["places"])


# -------------------------
# Models
# -------------------------
class PlaceItem(BaseModel):
    id: str
    name: str
    category_name: str = ""  # 상세 카테고리 (예: "카페,디저트", "한식,육류,고기요리")
    category_group_code: str = ""  # 대분류 코드 (CE7: 카페, FD6: 음식점)
    category_group_name: str = ""  # 대분류명 (카페, 음식점)
    phone: str = ""
    address_name: str = ""
    road_address_name: str = ""
    place_url: str = ""

    # 카드용(대안): place_url에서 og:image 추출
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

    crowding_level: str = ""          # 여유/보통/약간 붐빔/붐빔
    crowding_rank: int = 0            # 여유=4 ... 붐빔=1, 정보없음=0
    crowding_color: str = ""          # green/yellow/orange/red/""
    crowding_updated_at: int = 0
    crowding_message: str = ""


class PlaceWithZone(BaseModel):
    place: PlaceItem
    zone: ZoneInfo


class PlacesSearchResponse(BaseModel):
    items: List[PlaceItem] = Field(default_factory=list)


class PlacesInsightRequest(BaseModel):
    selected: PlaceItem

    # (선택) 사용자 위치
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None

    # 추천 중심점 선택: selected(기본) / user
    recommend_from: str = Field(
        default="selected",
        description="대체 후보 중심점: selected(기본)=선택가게 기준, user=사용자 기준",
    )

    radius_m: int = 1200
    max_candidates: int = 25
    max_alternatives: int = 3

    # (선택) cafe -> CE7, food -> FD6 강제 지정
    category: Optional[str] = None

    # 카드 사진 포함 여부(기본 true)
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


# 4단계로 정규화
def _normalize_crowding(level: str) -> str:
    level = (level or "").strip()
    if level in {"여유", "보통", "약간 붐빔", "붐빔"}:
        return level
    if level in {"매우 붐빔"}:
        return "붐빔"
    return ""


CROWDING_RANK = {"여유": 4, "보통": 3, "약간 붐빔": 2, "붐빔": 1, "": 0}
CROWDING_COLOR = {"여유": "green", "보통": "yellow", "약간 붐빔": "orange", "붐빔": "red", "": ""}


# -------------------------
# Zones index (from zones_seed.json)
# -------------------------
_ZONES: Optional[List[Dict[str, Any]]] = None


def _load_zones() -> List[Dict[str, Any]]:
    global _ZONES
    if _ZONES is not None:
        return _ZONES

    seed_path = os.getenv("ZONES_SEED_PATH", "app/resources/zones_seed.json")
    p = Path(seed_path)
    if not p.exists():
        raise RuntimeError(f"Zones seed not found: {seed_path}")

    _ZONES = json.loads(p.read_text(encoding="utf-8"))
    return _ZONES


def _nearest_zone(lat: float, lng: float) -> Tuple[Dict[str, Any], float]:
    zones = _load_zones()
    best = None
    best_d = 1e18
    for z in zones:
        zlat = float(z.get("lat") or 0.0)
        zlng = float(z.get("lng") or 0.0)
        if zlat == 0.0 and zlng == 0.0:
            continue
        d = _haversine_m(lat, lng, zlat, zlng)
        if d < best_d:
            best = z
            best_d = d

    if not best:
        raise RuntimeError("No valid zones in seed (all coords missing?)")

    return best, best_d


# -------------------------
# Seoul crowding client + cache
# -------------------------
_seoul_client = SeoulCityDataClient()
_crowd_cache = TTLCache(maxsize=512, ttl=int(os.getenv("CROWDING_CACHE_TTL_S", "300") or "300"))


def _fetch_crowding_for_zone(code: str, name: str) -> Tuple[str, str, int]:
    """
    Returns (level, message, updated_at_epoch)
    """
    cache_key = code or name
    if cache_key in _crowd_cache:
        return _crowd_cache[cache_key]

    # SeoulCityDataClient에는 is_configured()가 없으므로 api_key 존재 여부로 판단
    if not getattr(_seoul_client, "api_key", ""):
        result = ("", "", 0)
        _crowd_cache[cache_key] = result
        return result

    c = _seoul_client.fetch_area_crowding(area_code=code, area_name=name)
    lvl = _normalize_crowding(c.level)
    msg = c.message or ""
    updated = int(c.updated_at or 0)

    result = (lvl, msg, updated)
    _crowd_cache[cache_key] = result
    return result


def _zone_info_for_point(lat: float, lng: float) -> ZoneInfo:
    z, d = _nearest_zone(lat, lng)
    code = _safe_str(z.get("code"))
    name = _safe_str(z.get("name"))
    lvl, msg, updated = _fetch_crowding_for_zone(code, name)

    return ZoneInfo(
        code=code,
        name=name,
        lat=float(z.get("lat") or 0.0),
        lng=float(z.get("lng") or 0.0),
        distance_m=float(d),
        crowding_level=lvl,
        crowding_rank=int(CROWDING_RANK.get(lvl, 0)),
        crowding_color=str(CROWDING_COLOR.get(lvl, "")),
        crowding_updated_at=int(updated),
        crowding_message=msg,
    )


# -------------------------
# Kakao local calls
# -------------------------
def _kakao_rest_key() -> str:
    rest_key = os.getenv("KAKAO_REST_API_KEY")
    if not rest_key:
        raise RuntimeError("KAKAO_REST_API_KEY is missing in environment variables.")
    return rest_key


def kakao_keyword_search(
    *,
    query: str,
    lat: float,
    lng: float,
    radius_m: int,
    size: int,
) -> List[Dict[str, Any]]:
    """
    키워드 기반 검색(검색창 리스트용)
    """
    url = f"{KAKAO_BASE_URL}/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    # Kakao: radius<=20000, size<=15
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
    """
    카테고리 검색 1페이지 호출 (Kakao 제약: size <= 15)
    Returns (docs, is_end)
    """
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
    """
    카테고리 검색을 여러 페이지로 호출해 max_results까지 모음.
    (카카오 size<=15 제약 해결)
    """
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
# place_url -> og:image (카드 썸네일 대안)
# -------------------------
_image_cache = TTLCache(maxsize=1024, ttl=int(os.getenv("PLACE_IMAGE_CACHE_TTL_S", "3600") or "3600"))
_OG_IMAGE_RE = re.compile(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE)


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
    return p.copy(update={"image_url": img})


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
) -> PlacesSearchResponse:
    """
    내 위치와 검색어 기준으로 가까운 카페/음식점 리스트를 거리순으로 조회

    """
    try:
        docs = kakao_keyword_search(query=query, lat=lat, lng=lng, radius_m=radius_m, size=size)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"places search failed: {e}")

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

    if allowed is not None:
        docs = [d for d in docs if (d.get("category_group_code") in allowed)]

    items = [_doc_to_place(d) for d in docs]
    return PlacesSearchResponse(items=items)


@router.post("/insight", response_model=PlacesInsightResponse)
def insight(req: PlacesInsightRequest) -> PlacesInsightResponse:
    """
    선택한 가게의 Zone 혼잡도와 덜 붐비는 대체 가게 몇 곳을 함께 조회

    """
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

    selected_zone = _zone_info_for_point(req.selected.lat, req.selected.lng)
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
        z = _zone_info_for_point(p.lat, p.lng)
        candidates.append(PlaceWithZone(place=p, zone=z))

    # alternatives selection:
    # 1) 덜 붐비는 것 우선 (rank 높은 것)
    # 2) 같으면 가까운 것 우선 (place.distance_m: center 기준 distance)
    sel_rank = selected_zone.crowding_rank
    better = [x for x in candidates if x.zone.crowding_rank > sel_rank]
    pool = better if better else candidates

    pool.sort(key=lambda x: (-x.zone.crowding_rank, x.place.distance_m))
    alternatives = pool[: int(max(0, req.max_alternatives))]

    return PlacesInsightResponse(selected=selected, alternatives=alternatives)
