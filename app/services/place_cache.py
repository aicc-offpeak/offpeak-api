from __future__ import annotations

import os
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import PlaceCache as PlaceCacheModel

ALLOWED_CGC_DEFAULT: set[str] = {"CE7", "FD6"} 


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

    정책
    - category_group_code가 빈 문자열/None 이거나, CE7/FD6가 아니면:
      1) 캐시에 저장하지 않음 (upsert 차단)
      2) DB 검색 결과에서도 포함되지 않도록 필터링
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ttl_s = int(os.getenv("PLACE_CACHE_TTL_S", "86400"))  # 기본 24h
        self._last_debug_info: Optional[Dict[str, Any]] = None

    # -------------------------
    # TTL helpers
    # -------------------------
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

    # -------------------------
    # DB helpers
    # -------------------------
    def get(self, place_id: str, *, allow_stale: bool = False) -> Optional[PlaceCacheModel]:
        if not place_id:
            return None
        row = self.db.get(PlaceCacheModel, place_id)
        if not row:
            return None
        if allow_stale or self.is_fresh(row):
            # get()도 안전하게: 허용 코드가 아니면 반환하지 않음
            if _norm_cgc(row.category_group_code) not in ALLOWED_CGC_DEFAULT:
                return None
            return row
        return None

    def _get_last_fetched_at(self, place_id: str) -> Optional[datetime]:
        if not place_id:
            return None
        stmt = select(PlaceCacheModel.last_fetched_at).where(PlaceCacheModel.place_id == place_id)
        return self.db.scalar(stmt)

    # -------------------------
    # write-through (Kakao -> DB cache)
    # -------------------------
    def upsert_from_kakao_doc(self, d: Dict[str, Any]) -> bool:
        """
        Kakao doc(dict) -> place_cache upsert
        TTL 내 fresh면 write 생략 (API 호출 줄이는 핵심)
        Returns: wrote(True/False)

        변경: category_group_code가 CE7/FD6가 아니면 캐시에 저장하지 않음
        """
        place_id = str(d.get("id") or "").strip()
        if not place_id:
            return False

        cgc = _norm_cgc(d.get("category_group_code"))
        # 빈 코드/허용되지 않은 코드 차단
        if cgc not in ALLOWED_CGC_DEFAULT:
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
            "category_group_code": cgc, 
            "category_group_name": s(d.get("category_group_name")),
            "last_fetched_at": _now_utc(),
        }

        stmt = pg_insert(PlaceCacheModel).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values.keys() if k != "place_id"}
        stmt = stmt.on_conflict_do_update(index_elements=["place_id"], set_=update_cols)
        self.db.execute(stmt)
        return True

    # -------------------------
    # row -> response dict
    # -------------------------
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
            "image_url": "",
            "lat": float(row.lat or 0.0),
            "lng": float(row.lng or 0.0),
            "distance_m": float(round(distance_m, 1)),
        }

    # -------------------------
    # search (query 포함)
    # -------------------------
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

        변경:
        - allowed_category_group_codes가 주어지면 그 코드만 허용(빈/NULL 절대 통과 X)
        - allowed가 없더라도 기본은 CE7/FD6만 허용(서비스 정책)
        """
        q = (query or "").strip()
        if not q:
            return []

        limit = int(max(1, min(limit, 50)))
        radius_m = int(max(0, min(radius_m, 20000)))

        lat_delta = radius_m / 111320.0
        cosv = math.cos(math.radians(lat)) or 1e-9
        lng_delta = radius_m / (111320.0 * cosv)

        fresh_since = self._fresh_since()

        # allowed가 없으면 기본 CE7/FD6로 제한
        if allowed_category_group_codes is None:
            allowed_norm: set[str] = set(ALLOWED_CGC_DEFAULT)
        else:
            allowed_norm = {_norm_cgc(x) for x in allowed_category_group_codes if _norm_cgc(x)}
            # caller가 실수로 빈 set을 주면 결과도 비게 처리
            if not allowed_norm:
                return []

        # debug counts (기존 유지)
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
            .where(PlaceCacheModel.category_group_code.in_(sorted(allowed_norm)))
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
            "allowed": sorted(list(allowed_norm)),
        }

        out: List[Tuple[PlaceCacheModel, float]] = []
        for r in rows:
            rcgc = _norm_cgc(r.category_group_code)
            if rcgc not in allowed_norm:
                continue

            d = _haversine_m(lat, lng, float(r.lat or 0.0), float(r.lng or 0.0))
            if d <= float(radius_m):
                out.append((r, d))

        out.sort(key=lambda x: x[1])
        return out[:limit]

    # -------------------------
    # search (query 없이 카테고리 주변검색)
    # -------------------------
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

        변경:
        - 입력 allowed_category_group_codes가 없거나(또는 CE7/FD6 외)면 빈 결과
        """
        if not allowed_category_group_codes:
            return []

        limit = int(max(1, min(limit, 50)))
        radius_m = int(max(0, min(radius_m, 20000)))

        lat_delta = radius_m / 111320.0
        cosv = math.cos(math.radians(lat)) or 1e-9
        lng_delta = radius_m / (111320.0 * cosv)

        fresh_since = self._fresh_since()

        # 허용 집합으로 교집합
        allowed_norm = sorted(
            {_norm_cgc(x) for x in allowed_category_group_codes if _norm_cgc(x)} & ALLOWED_CGC_DEFAULT
        )
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

    # -------------------------
    # cached nearby for recommendation (category required)
    # -------------------------
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

        변경:
        - 요청 cgc가 CE7/FD6가 아니면 캐시 조회 자체를 하지 않음
        """
        radius_m = int(max(0, min(radius_m, 20000)))
        limit = int(max(1, min(limit, 50)))
        ttl_s = int(self.ttl_s if ttl_s is None else max(1, int(ttl_s)))

        cgc = _norm_cgc(category_group_code)
        if cgc not in ALLOWED_CGC_DEFAULT:
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
            if _norm_cgc(r.category_group_code) != cgc:
                continue

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
