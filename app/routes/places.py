from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import quote

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import desc, select, text
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import CrowdingSnapshot as CrowdingSnapshotRow
from app.models import Zone
from app.services.crowding import CrowdingService, crowding_color
from app.services.place_cache import PlaceCacheService
from app.services.place_crowding_snapshot import PlaceCrowdingSnapshotService
from app.services.place_profile import PlaceProfileService

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


def _doc_cgc(d: Dict[str, Any]) -> str:
    return (d.get("category_group_code") or "").strip().upper()


# -------------------------
# Kakao local calls
# -------------------------
def kakao_keyword_search(
    *,
    query: str,
    lat: float,
    lng: float,
    radius_m: int,
    max_results: int,
) -> List[Dict[str, Any]]:
    url = f"{KAKAO_BASE_URL}/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {_kakao_rest_key()}"}

    q = (query or "").strip()
    if not q:
        return []

    radius_m = int(max(0, min(radius_m, 20000)))
    max_results = int(max(1, min(max_results, 45)))

    docs_all: List[Dict[str, Any]] = []
    seen: set[str] = set()
    page = 1

    while len(docs_all) < max_results and page <= 45:
        remaining = max_results - len(docs_all)
        size = min(15, remaining)

        params = {
            "query": q,
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
            raise RuntimeError(f"Kakao keyword search failed: {r.status_code} {r.text}")

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


def kakao_category_search(
    *,
    category_group_code: str,
    lat: float,
    lng: float,
    radius_m: int,
    max_results: int,
) -> List[Dict[str, Any]]:
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


def _doc_to_place(d: Dict[str, Any]) -> PlaceItem:
    return PlaceItem(
        id=_safe_str(d.get("id")),
        name=_safe_str(d.get("place_name") or d.get("name")),
        category_name=_safe_str(d.get("category_name")),
        category_group_code=_safe_str(d.get("category_group_code")),
        category_group_name=_safe_str(d.get("category_group_name")),
        phone=_safe_str(d.get("phone")),
        address_name=_safe_str(d.get("address_name")),
        road_address_name=_safe_str(d.get("road_address_name")),
        place_url=_safe_str(d.get("place_url")),
        image_url="",
        lat=_safe_float(d.get("y") or d.get("lat")),
        lng=_safe_float(d.get("x") or d.get("lng")),
        distance_m=_safe_float(d.get("distance") or d.get("distance_m")),
    )


# -------------------------
# place_url -> og:image (optional)
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


def _is_snapshot_fresh(row: CrowdingSnapshotRow, min_interval_s: int) -> bool:
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
    z, dist = _nearest_zone(zones, lat, lng)

    if z.code in snapshot_cache:
        cached = snapshot_cache[z.code]
        return cached.model_copy(update={"distance_m": float(dist)}), False

    wrote = False
    latest = _get_latest_snapshot(db, z.code)

    if latest and _is_snapshot_fresh(latest, min_interval_s):
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
            CrowdingSnapshotRow(
                zone_code=z.code,
                level=level,
                rank=rank,
                message=msg,
                updated_at_epoch=updated,
                raw={"color": color, "seoul": dto.raw},
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
SearchScope = Literal["cache", "api", "all"]
CategoryScope = Literal["cafe", "food", "food_cafe", "all"]


@router.get("/search", response_model=PlacesSearchResponse)
def search_places(
    query: str = Query(..., min_length=1, description="검색어(예: 스타벅스)"),
    lat: float = Query(..., ge=-90, le=90, description="현재 위도"),
    lng: float = Query(..., ge=-180, le=180, description="현재 경도"),
    size: int = Query(15, ge=1, le=45, description="가까운 순 결과 개수(기본 15, 최대 45)"),
    radius_m: int = Query(3000, ge=0, le=20000, description="검색 반경(m) (기본 3000)"),
    scope: SearchScope = Query("all", description="cache | api | all"),
    category_scope: CategoryScope = Query("food_cafe", description="cafe | food | food_cafe | all"),
    category_group_code: Optional[str] = Query(None, description="카테고리 그룹 코드(CE7, FD6)"),
    response: Response = None,
    db: Session = Depends(get_db),
) -> PlacesSearchResponse:
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    size = int(max(1, min(size, 45)))
    cache = PlaceCacheService(db)

    # DEBUG headers (ASCII only)
    if response is not None:
        try:
            db_name = db.execute(text("select current_database()")).scalar()
            cache_cnt = db.execute(text("select count(*) from place_cache")).scalar()
            response.headers["X-DB"] = str(db_name)
            response.headers["X-PlaceCache-Count"] = str(cache_cnt)
            response.headers["X-PlaceCache-TTL"] = str(cache.ttl_s)
            response.headers["X-App-File"] = __file__
            response.headers["X-Query"] = quote(q, safe="")
        except Exception:
            pass

    # category filter
    if category_group_code:
        allowed = {(category_group_code or "").strip().upper()}
    elif category_scope == "cafe":
        allowed = {"CE7"}
    elif category_scope == "food":
        allowed = {"FD6"}
    elif category_scope == "food_cafe":
        allowed = {"CE7", "FD6"}
    else:
        allowed = None

    cached_items: List[PlaceItem] = []
    if scope in ("cache", "all"):
        rows = cache.search_nearby_in_db(
            query=q,
            lat=lat,
            lng=lng,
            radius_m=int(radius_m),
            limit=size,
            allowed_category_group_codes=allowed,
        )

        if response is not None:
            try:
                response.headers["X-Cache-DBRows"] = str(len(rows))
                if cache._last_debug_info:
                    debug = cache._last_debug_info
                    response.headers["X-Cache-Debug"] = (
                        f"total={debug.get('total_count', 0)} "
                        f"fresh={debug.get('fresh_count', 0)} "
                        f"name_match={debug.get('name_match_count', 0)} "
                        f"bbox={debug.get('bbox_count', 0)} "
                        f"combined={debug.get('combined_count', 0)}"
                    )
            except Exception as e:
                try:
                    response.headers["X-Cache-Debug-Error"] = e.__class__.__name__
                except Exception:
                    pass

        for r, d in rows:
            cached_items.append(PlaceItem(**cache.row_to_place_dict(r, distance_m=d)))

    cache_hits = len(cached_items)

    if scope == "cache":
        if response is not None:
            response.headers["X-Place-Cache"] = f"source=cache hits={cache_hits} writes=0"
        return PlacesSearchResponse(items=cached_items[:size])

    if scope == "all" and cache_hits >= size:
        if response is not None:
            response.headers["X-Place-Cache"] = f"source=cache hits={cache_hits} writes=0"
        return PlacesSearchResponse(items=cached_items[:size])

    need = size if scope == "api" else (size - cache_hits)
    try:
        docs = kakao_keyword_search(query=q, lat=lat, lng=lng, radius_m=radius_m, max_results=need)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"places search failed: {e}")

    if response is not None:
        try:
            response.headers["X-Kakao-Docs-Raw"] = str(len(docs))
        except Exception:
            pass

    if allowed is not None:
        docs = [d for d in docs if (_doc_cgc(d) in allowed) or (_doc_cgc(d) == "")]

    if response is not None:
        try:
            response.headers["X-Kakao-Docs-AfterFilter"] = str(len(docs))
            response.headers["X-Kakao-Allowed"] = ",".join(sorted(allowed)) if allowed else ""
        except Exception:
            pass

    wrote = 0
    seen_ids = {it.id for it in cached_items}
    api_items: List[PlaceItem] = []

    for d in docs:
        pid = _safe_str(d.get("id"))
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)

        try:
            if cache.upsert_from_kakao_doc(d):
                wrote += 1
        except Exception:
            pass

        api_items.append(_doc_to_place(d))
        if len(api_items) >= need:
            break

    if wrote:
        db.commit()

    if scope == "api":
        if response is not None:
            response.headers["X-Place-Cache"] = f"source=api hits=0 writes={wrote}"
        return PlacesSearchResponse(items=api_items[:size])

    merged = (cached_items + api_items)[:size]
    if response is not None:
        response.headers["X-Place-Cache"] = f"source=merged hits={cache_hits} writes={wrote}"
    return PlacesSearchResponse(items=merged)


@router.post("/insight", response_model=PlacesInsightResponse)
def insight(
    req: PlacesInsightRequest,
    response: Response = None,
    db: Session = Depends(get_db),
) -> PlacesInsightResponse:
    if req.recommend_from not in {"selected", "user"}:
        raise HTTPException(status_code=400, detail="recommend_from must be 'selected' or 'user'")

    if req.recommend_from == "user" and req.user_lat is not None and req.user_lng is not None:
        center_lat = float(req.user_lat)
        center_lng = float(req.user_lng)
    else:
        center_lat = float(req.selected.lat)
        center_lng = float(req.selected.lng)

    cgc = (req.selected.category_group_code or "").strip()
    if req.category:
        if req.category.lower() == "cafe":
            cgc = "CE7"
        elif req.category.lower() in {"food", "restaurant"}:
            cgc = "FD6"
    if not cgc:
        cgc = "CE7"

    zones: List[Zone] = list(db.scalars(select(Zone)).all())
    min_interval_s = int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "600"))
    crowding = CrowdingService()

    # -------------------------
    # Place snapshot helpers (ORM + fallback raw SQL)
    # -------------------------
    place_snap = PlaceCrowdingSnapshotService(db)
    place_snap_dirty = False
    place_snap_selected_added = 0
    place_snap_candidate_added = 0
    place_snap_err: str = ""

    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def _is_fresh_ts(ts: Optional[datetime], min_interval_s: int) -> bool:
        if ts is None:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (_now_utc() - ts.astimezone(timezone.utc)).total_seconds()
        return age_s < float(min_interval_s)

    def _raw_latest_ts(place_id: str) -> Optional[datetime]:
        return db.execute(
            text(
                """
                select ts
                from place_crowding_snapshots
                where place_id = :pid
                order by ts desc
                limit 1
                """
            ),
            {"pid": place_id},
        ).scalar()

    def _raw_insert(**params: Any) -> bool:
        # NOTE: 컬럼명이 migration과 다르면 여기서 오류가 나고 header로 드러남
        db.execute(
            text(
                """
                insert into place_crowding_snapshots
                (place_id, place_name, category_group_code, lat, lng,
                 zone_code, zone_distance_m, level, rank, message, updated_at_epoch, raw)
                values
                (:place_id, :place_name, :category_group_code, :lat, :lng,
                 :zone_code, :zone_distance_m, :level, :rank, :message, :updated_at_epoch, :raw)
                """
            ),
            params,
        )
        return True

    def _record_place_snapshot(
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
        raw: dict[str, Any] | None,
    ) -> bool:
        nonlocal place_snap_err

        pid = (place_id or "").strip()
        if not pid:
            return False

        # 1) ORM 방식 (서비스 재사용)
        try:
            with db.begin_nested():
                did_add = place_snap.record(
                    place_id=pid,
                    place_name=place_name,
                    category_group_code=category_group_code,
                    lat=lat,
                    lng=lng,
                    zone_code=zone_code,
                    zone_distance_m=zone_distance_m,
                    level=level,
                    rank=int(rank or 0),
                    message=message,
                    updated_at_epoch=int(updated_at_epoch or 0),
                    raw=raw,
                )
                if did_add:
                    db.flush()  # 여기서 스키마/테이블 문제면 바로 터짐
            return bool(did_add)
        except Exception as e:
            place_snap_err = e.__class__.__name__

        # 2) fallback raw SQL (테이블/컬럼이 실제로 존재하면 insert라도 되게)
        try:
            latest = _raw_latest_ts(pid)
            min_s = int(os.getenv("PLACE_CROWDING_SNAPSHOT_MIN_INTERVAL_S", "1800"))
            if _is_fresh_ts(latest, min_s):
                return False

            with db.begin_nested():
                _raw_insert(
                    place_id=pid,
                    place_name=place_name,
                    category_group_code=category_group_code,
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
                db.flush()
            return True
        except Exception as e2:
            # 두 단계 모두 실패하면 "A->B" 형태로 남김 (헤더 ASCII OK)
            place_snap_err = f"{place_snap_err}->{e2.__class__.__name__}" if place_snap_err else e2.__class__.__name__
            return False

    # -------------------------
    # zone snapshot
    # -------------------------
    snapshot_cache: Dict[str, ZoneInfo] = {}
    wrote_any = False
    cache_dirty = False

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

    # -------------------------
    # Record selected snapshot
    # -------------------------
    try:
        did = _record_place_snapshot(
            place_id=req.selected.id,
            place_name=req.selected.name,
            category_group_code=req.selected.category_group_code,
            lat=float(req.selected.lat),
            lng=float(req.selected.lng),
            zone_code=selected_zone.code,
            zone_distance_m=float(selected_zone.distance_m),
            level=selected_zone.crowding_level,
            rank=int(selected_zone.crowding_rank or 0),
            message=selected_zone.crowding_message,
            updated_at_epoch=int(selected_zone.crowding_updated_at or 0),
            raw={"color": selected_zone.crowding_color, "source": "zone"},
        )
        if did:
            place_snap_dirty = True
            place_snap_selected_added += 1
    except Exception as e:
        place_snap_err = e.__class__.__name__

    # -------------------------
    # candidate fetching (cache -> api)
    # -------------------------
    place_cache = PlaceCacheService(db)
    ttl_s = int(os.getenv("PLACE_CACHE_TTL_S", "86400"))

    cached = place_cache.get_cached_places_near(
        lat=center_lat,
        lng=center_lng,
        radius_m=int(req.radius_m),
        category_group_code=cgc,
        limit=int(req.max_candidates),
        ttl_s=ttl_s,
    )

    docs: List[Dict[str, Any]] = []
    if len(cached) >= int(req.max_candidates):
        for item in cached:
            docs.append(
                {
                    "id": item.get("id"),
                    "place_name": item.get("name"),
                    "category_name": item.get("category_name"),
                    "category_group_code": item.get("category_group_code"),
                    "category_group_name": item.get("category_group_name"),
                    "phone": item.get("phone"),
                    "address_name": item.get("address_name"),
                    "road_address_name": item.get("road_address_name"),
                    "place_url": item.get("place_url"),
                    "y": item.get("lat"),
                    "x": item.get("lng"),
                    "distance": item.get("distance_m"),
                }
            )
    else:
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

        for d in docs:
            try:
                if place_cache.upsert_from_kakao_doc(d):
                    cache_dirty = True
            except Exception:
                pass

    # -------------------------
    # build candidates + record snapshots
    # -------------------------
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

        try:
            did = _record_place_snapshot(
                place_id=p.id,
                place_name=p.name,
                category_group_code=p.category_group_code,
                lat=float(p.lat),
                lng=float(p.lng),
                zone_code=zinfo.code,
                zone_distance_m=float(zinfo.distance_m),
                level=zinfo.crowding_level,
                rank=int(zinfo.crowding_rank or 0),
                message=zinfo.crowding_message,
                updated_at_epoch=int(zinfo.crowding_updated_at or 0),
                raw={"color": zinfo.crowding_color, "source": "zone"},
            )
            if did:
                place_snap_dirty = True
                place_snap_candidate_added += 1
        except Exception as e:
            place_snap_err = e.__class__.__name__

        candidates.append(PlaceWithZone(place=p, zone=zinfo))

    # -------------------------
    # commit
    # -------------------------
    committed = False
    if wrote_any or cache_dirty or place_snap_dirty:
        try:
            db.commit()
            committed = True
        except Exception as e:
            db.rollback()
            place_snap_err = place_snap_err or e.__class__.__name__

    # -------------------------
    # debug headers
    # -------------------------
    if response is not None:
        try:
            response.headers["X-PlaceSnap-Selected-Added"] = str(place_snap_selected_added)
            response.headers["X-PlaceSnap-Candidate-Added"] = str(place_snap_candidate_added)
            response.headers["X-PlaceSnap-Dirty"] = "1" if place_snap_dirty else "0"
            response.headers["X-PlaceSnap-Commit"] = "1" if committed else "0"
            response.headers["X-PlaceSnap-Err"] = (place_snap_err or "")
        except Exception:
            pass

    sel_rank = selected_zone.crowding_rank
    better = [x for x in candidates if x.zone.crowding_rank > sel_rank]
    pool = better if better else candidates

    pool.sort(key=lambda x: (-x.zone.crowding_rank, x.place.distance_m))
    alternatives = pool[: int(max(0, req.max_alternatives))]

    return PlacesInsightResponse(selected=selected, alternatives=alternatives)

@router.get("/profile")
def place_profile(
    place_id: str = Query(..., description="Kakao place id"),
    days: Optional[int] = Query(None, ge=1, le=365, description="최근 N일만 집계(옵션)"),
    min_samples: int = Query(3, ge=1, le=100, description="셀당 최소 샘플 수"),
    db: Session = Depends(get_db),
):
    svc = PlaceProfileService(db)
    return svc.weekly_hourly_profile(place_id=place_id, days=days, min_samples=min_samples)

@router.get("/recommend_times")
def recommend_times(
    place_id: str = Query(...),
    days: int = Query(30, ge=1, le=365),
    min_samples: int = Query(3, ge=1, le=100),
    per_day: int = Query(3, ge=1, le=10),
    window_h: int = Query(2, ge=1, le=6),
    include_low_samples: bool = Query(False),
    fallback_to_hourly: bool = Query(True),  
    db: Session = Depends(get_db),
):
    svc = PlaceProfileService(db)
    return svc.recommend_quiet_times(
        place_id=place_id,
        days=days,
        min_samples=min_samples,
        per_day=per_day,
        window_h=window_h,
        include_low_samples=include_low_samples,
        fallback_to_hourly=fallback_to_hourly, 
    )
