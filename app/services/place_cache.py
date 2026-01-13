from __future__ import annotations

import os
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, text, func, or_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import PlaceCache as PlaceCacheModel


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _norm_cgc(v: Any) -> str:
    return ("" if v is None else str(v)).strip().upper()


class PlaceCacheService:
    """
    place_cache 테이블 기반 read-through 캐시
    - upsert_from_kakao_doc: place_id 단위 upsert (TTL fresh면 write 생략)
    - search_*: DB에서 먼저 검색(nearby) -> API fallback 시나리오 지원
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ttl_s = int(os.getenv("PLACE_CACHE_TTL_S", "86400"))  # 기본 24h
        self._last_debug_info: Optional[Dict[str, Any]] = None

    def _fresh_since(self) -> datetime:
        return _now_utc() - timedelta(seconds=int(self.ttl_s))

    def is_fresh_ts(self, ts: Optional[datetime]) -> bool:
        if not ts:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= self._fresh_since()

    def is_fresh(self, row: PlaceCacheModel) -> bool:
        return self.is_fresh_ts(row.last_fetched_at)

    def get(self, place_id: str, *, allow_stale: bool = False) -> Optional[PlaceCacheModel]:
        if not place_id:
            return None
        row = self.db.get(PlaceCacheModel, place_id)
        if not row:
            return None
        if allow_stale or self.is_fresh(row):
            return row
        return None

    def _get_last_fetched_at(self, place_id: str) -> Optional[datetime]:
        if not place_id:
            return None
        stmt = select(PlaceCacheModel.last_fetched_at).where(PlaceCacheModel.place_id == place_id)
        return self.db.scalar(stmt)

    def upsert_from_kakao_doc(self, d: Dict[str, Any]) -> bool:
        """
        Kakao doc(dict) -> place_cache upsert
        TTL 내 fresh면 write 생략 (API 호출 줄이는 핵심)
        Returns: wrote(True/False)
        """
        place_id = str(d.get("id") or "").strip()
        if not place_id:
            return False

        last = self._get_last_fetched_at(place_id)
        if self.is_fresh_ts(last):
            return False

        def s(x: Any) -> str:
            return "" if x is None else str(x)

        def f(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        values = {
            "place_id": place_id,
            "name": s(d.get("place_name") or d.get("name")),
            "phone": s(d.get("phone")),
            "address_name": s(d.get("address_name")),
            "road_address_name": s(d.get("road_address_name")),
            "place_url": s(d.get("place_url")),
            "lat": f(d.get("y") or d.get("lat")),
            "lng": f(d.get("x") or d.get("lng")),
            "category_name": s(d.get("category_name")),
            # FIX: 공백/소문자/None 정규화
            "category_group_code": _norm_cgc(d.get("category_group_code")),
            "category_group_name": s(d.get("category_group_name")),
            "last_fetched_at": _now_utc(),
        }

        stmt = pg_insert(PlaceCacheModel).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values.keys() if k != "place_id"}
        stmt = stmt.on_conflict_do_update(index_elements=["place_id"], set_=update_cols)
        self.db.execute(stmt)
        return True

    def row_to_place_dict(self, row: PlaceCacheModel, *, distance_m: float) -> Dict[str, Any]:
        return {
            "id": row.place_id,
            "name": row.name,
            "category_name": row.category_name,
            "category_group_code": row.category_group_code,
            "category_group_name": row.category_group_name,
            "phone": row.phone,
            "address_name": row.address_name,
            "road_address_name": row.road_address_name,
            "place_url": row.place_url,
            "image_url": "",  # 호환 필드
            "lat": float(row.lat or 0.0),
            "lng": float(row.lng or 0.0),
            "distance_m": float(round(distance_m, 1)),
        }

    def search_nearby_in_db(
        self,
        *,
        query: str,
        lat: float,
        lng: float,
        radius_m: int,
        limit: int,
        allowed_category_group_codes: Optional[set[str]] = None,
    ) -> List[Tuple[PlaceCacheModel, float]]:
        """
        DB place_cache에서:
        - TTL fresh row만
        - name ilike '%query%'
        - bbox로 1차 필터 후 haversine로 정확 필터
        Returns: [(row, distance_m)]
        """
        q = (query or "").strip()
        if not q:
            return []

        limit = int(max(1, min(limit, 50)))
        radius_m = int(max(0, min(radius_m, 20000)))

        # bbox(대략)
        lat_delta = radius_m / 111320.0
        cosv = math.cos(math.radians(lat)) or 1e-9
        lng_delta = radius_m / (111320.0 * cosv)

        fresh_since = self._fresh_since()

        allowed_norm: Optional[set[str]] = None
        if allowed_category_group_codes:
            allowed_norm = {_norm_cgc(x) for x in allowed_category_group_codes if _norm_cgc(x)}

        # 디버그 카운트는 "fresh 기준"으로 맞춤 (진단 정확도 ↑)
        total_count = self.db.scalar(select(func.count()).select_from(PlaceCacheModel))
        fresh_count = self.db.scalar(
            select(func.count()).select_from(PlaceCacheModel).where(PlaceCacheModel.last_fetched_at >= fresh_since)
        )
        name_match_count = self.db.scalar(
            select(func.count())
            .select_from(PlaceCacheModel)
            .where(PlaceCacheModel.last_fetched_at >= fresh_since)
            .where(PlaceCacheModel.name.ilike(f"%{q}%"))
        )
        bbox_count = self.db.scalar(
            select(func.count())
            .select_from(PlaceCacheModel)
            .where(PlaceCacheModel.last_fetched_at >= fresh_since)
            .where(PlaceCacheModel.lat.between(lat - lat_delta, lat + lat_delta))
            .where(PlaceCacheModel.lng.between(lng - lng_delta, lng + lng_delta))
        )
        combined_count = self.db.scalar(
            select(func.count())
            .select_from(PlaceCacheModel)
            .where(PlaceCacheModel.last_fetched_at >= fresh_since)
            .where(PlaceCacheModel.name.ilike(f"%{q}%"))
            .where(PlaceCacheModel.lat.between(lat - lat_delta, lat + lat_delta))
            .where(PlaceCacheModel.lng.between(lng - lng_delta, lng + lng_delta))
        )

        stmt = (
            select(PlaceCacheModel)
            .where(PlaceCacheModel.last_fetched_at >= fresh_since)
            .where(PlaceCacheModel.name.ilike(f"%{q}%"))
            .where(PlaceCacheModel.lat.between(lat - lat_delta, lat + lat_delta))
            .where(PlaceCacheModel.lng.between(lng - lng_delta, lng + lng_delta))
        )

        # FIX: allowed가 있을 때도 DB에서 1차 필터
        # (단, category_group_code가 비어있으면 통과시켜서 "Kakao가 code 안 준 row"도 살림)
        if allowed_norm:
            stmt = stmt.where(
                or_(
                    PlaceCacheModel.category_group_code.in_(sorted(list(allowed_norm))),
                    PlaceCacheModel.category_group_code.is_(None),
                    PlaceCacheModel.category_group_code == "",
                )
            )

        rows = list(self.db.scalars(stmt).all())

        self._last_debug_info = {
            "total_count": total_count or 0,
            "fresh_count": fresh_count or 0,
            "name_match_count": name_match_count or 0,
            "bbox_count": bbox_count or 0,
            "combined_count": combined_count or 0,
            "fresh_since": fresh_since.isoformat() if fresh_since else None,
            "query": q,
        }

        out: List[Tuple[PlaceCacheModel, float]] = []
        for r in rows:
            # FIX: Python 레벨에서도 정규화 비교 + 비어있으면 통과
            if allowed_norm:
                rcgc = _norm_cgc(r.category_group_code)
                if rcgc and (rcgc not in allowed_norm):
                    continue

            d = _haversine_m(lat, lng, float(r.lat or 0.0), float(r.lng or 0.0))
            if d <= float(radius_m):
                out.append((r, d))

        out.sort(key=lambda x: x[1])
        return out[:limit]

    def search_category_nearby_in_db(
        self,
        *,
        lat: float,
        lng: float,
        radius_m: int,
        limit: int,
        allowed_category_group_codes: set[str],
    ) -> List[Tuple[PlaceCacheModel, float]]:
        """
        추천처럼 'query 없이 카테고리 주변검색'을 DB에서 먼저 하기 위한 함수.
        """
        if not allowed_category_group_codes:
            return []

        limit = int(max(1, min(limit, 50)))
        radius_m = int(max(0, min(radius_m, 20000)))

        lat_delta = radius_m / 111320.0
        cosv = math.cos(math.radians(lat)) or 1e-9
        lng_delta = radius_m / (111320.0 * cosv)

        fresh_since = self._fresh_since()
        allowed_norm = sorted({_norm_cgc(x) for x in allowed_category_group_codes if _norm_cgc(x)})
        if not allowed_norm:
            return []

        stmt = (
            select(PlaceCacheModel)
            .where(PlaceCacheModel.last_fetched_at >= fresh_since)
            .where(PlaceCacheModel.category_group_code.in_(allowed_norm))
            .where(PlaceCacheModel.lat.between(lat - lat_delta, lat + lat_delta))
            .where(PlaceCacheModel.lng.between(lng - lng_delta, lng + lng_delta))
        )

        rows = list(self.db.scalars(stmt).all())

        out: List[Tuple[PlaceCacheModel, float]] = []
        for r in rows:
            d = _haversine_m(lat, lng, float(r.lat or 0.0), float(r.lng or 0.0))
            if d <= float(radius_m):
                out.append((r, d))

        out.sort(key=lambda x: x[1])
        return out[:limit]

    def get_cached_places_near(
        self,
        *,
        lat: float,
        lng: float,
        radius_m: int,
        category_group_code: str,
        limit: int,
        ttl_s: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        zone 중심(lat,lng) 기준 반경 내 place_cache 조회
        - TTL 내(last_fetched_at 최신)만 사용
        - bbox로 1차 필터 후 haversine로 정확 정렬
        Returns: List[dict]
        """
        radius_m = int(max(0, min(radius_m, 20000)))
        limit = int(max(1, min(limit, 50)))
        ttl_s = int(self.ttl_s if ttl_s is None else max(1, int(ttl_s)))

        cgc = _norm_cgc(category_group_code)
        if not cgc:
            return []

        lat_delta = radius_m / 111320.0
        cosv = math.cos(math.radians(lat)) or 1e-9
        lng_delta = radius_m / (111320.0 * cosv)

        fresh_since = _now_utc() - timedelta(seconds=ttl_s)

        stmt = (
            select(PlaceCacheModel)
            .where(PlaceCacheModel.last_fetched_at >= fresh_since)
            .where(PlaceCacheModel.category_group_code == cgc)
            .where(PlaceCacheModel.lat.between(lat - lat_delta, lat + lat_delta))
            .where(PlaceCacheModel.lng.between(lng - lng_delta, lng + lng_delta))
        )

        rows = list(self.db.scalars(stmt).all())

        items: List[Dict[str, Any]] = []
        for r in rows:
            plat = float(r.lat or 0.0)
            plng = float(r.lng or 0.0)
            dist = _haversine_m(lat, lng, plat, plng)
            if dist > float(radius_m):
                continue

            items.append(
                {
                    "id": str(r.place_id),
                    "name": r.name or "",
                    "phone": r.phone or "",
                    "address_name": r.address_name or "",
                    "road_address_name": r.road_address_name or "",
                    "place_url": r.place_url or "",
                    "lat": plat,
                    "lng": plng,
                    "category_name": r.category_name or "",
                    "category_group_code": r.category_group_code or "",
                    "category_group_name": r.category_group_name or "",
                    "distance_m": float(dist),
                }
            )

        items.sort(key=lambda x: x["distance_m"])
        return items[:limit]
