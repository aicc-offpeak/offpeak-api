from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session


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


def _bbox(lat: float, lng: float, radius_m: float) -> tuple[float, float, float, float]:
    # 대략 bbox (1도 위도 ≈ 111,320m)
    lat_delta = radius_m / 111_320.0
    cosv = math.cos(math.radians(lat))
    if abs(cosv) < 1e-9:
        lng_delta = radius_m / 111_320.0
    else:
        lng_delta = radius_m / (111_320.0 * cosv)

    return (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta)


@dataclass(frozen=True)
class PlaceCacheRow:
    place_id: str
    name: str
    phone: str
    address_name: str
    road_address_name: str
    place_url: str
    lat: float
    lng: float
    category_name: str
    category_group_code: str
    category_group_name: str
    last_fetched_at: datetime


class PlaceCacheService:
    """
    place_cache 테이블 기반 장소 캐시 서비스 (SQL text 기반)
    - place_id(PK) 단위로 Kakao 장소 정보를 upsert
    - TTL 내에는 fresh로 간주 (fresh면 write 생략)
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ttl_s = int(os.getenv("PLACE_CACHE_TTL_S", "86400"))  # 기본 24h

    @staticmethod
    def _s(x: Any) -> str:
        return "" if x is None else str(x)

    @staticmethod
    def _f(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def is_fresh(self, row: PlaceCacheRow) -> bool:
        ts = row.last_fetched_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (_now_utc() - ts.astimezone(timezone.utc)).total_seconds()
        return age_s <= float(self.ttl_s)

    def get(self, place_id: str, *, allow_stale: bool = False) -> Optional[PlaceCacheRow]:
        pid = (place_id or "").strip()
        if not pid:
            return None

        q = text(
            """
            select
              place_id, name, phone, address_name, road_address_name, place_url,
              lat, lng, category_name, category_group_code, category_group_name,
              last_fetched_at
            from place_cache
            where place_id = :pid
            """
        )
        r = self.db.execute(q, {"pid": pid}).first()
        if not r:
            return None

        row = PlaceCacheRow(
            place_id=str(r.place_id),
            name=r.name or "",
            phone=r.phone or "",
            address_name=r.address_name or "",
            road_address_name=r.road_address_name or "",
            place_url=r.place_url or "",
            lat=float(r.lat),
            lng=float(r.lng),
            category_name=r.category_name or "",
            category_group_code=r.category_group_code or "",
            category_group_name=r.category_group_name or "",
            last_fetched_at=r.last_fetched_at,
        )

        if allow_stale or self.is_fresh(row):
            return row
        return None

    def upsert_from_kakao_doc(self, d: Dict[str, Any]) -> bool:
        """
        Kakao Local doc(dict) -> place_cache upsert
        fresh면 write 생략해서 DB 부담 줄임
        Returns: wrote(True/False)
        """
        place_id = self._s(d.get("id")).strip()
        if not place_id:
            return False

        # ✅ TTL 내 fresh면 업데이트 생략 (DB에서 판단)
        fresh_check = text(
            """
            select 1
            from place_cache
            where place_id = :pid
              and last_fetched_at >= (now() - (:ttl_s || ' seconds')::interval)
            limit 1
            """
        )
        is_fresh = self.db.execute(fresh_check, {"pid": place_id, "ttl_s": int(self.ttl_s)}).first() is not None
        if is_fresh:
            return False

        values = {
            "place_id": place_id,
            "name": self._s(d.get("place_name")).strip(),
            "phone": self._s(d.get("phone")).strip(),
            "address_name": self._s(d.get("address_name")).strip(),
            "road_address_name": self._s(d.get("road_address_name")).strip(),
            "place_url": self._s(d.get("place_url")).strip(),
            "lat": self._f(d.get("y")),
            "lng": self._f(d.get("x")),
            "category_name": self._s(d.get("category_name")).strip(),
            "category_group_code": self._s(d.get("category_group_code")).strip(),
            "category_group_name": self._s(d.get("category_group_name")).strip(),
            "last_fetched_at": _now_utc(),
        }

        upsert = text(
            """
            insert into place_cache (
              place_id, name, phone, address_name, road_address_name, place_url,
              lat, lng, category_name, category_group_code, category_group_name,
              last_fetched_at
            ) values (
              :place_id, :name, :phone, :address_name, :road_address_name, :place_url,
              :lat, :lng, :category_name, :category_group_code, :category_group_name,
              :last_fetched_at
            )
            on conflict (place_id) do update set
              name = excluded.name,
              phone = excluded.phone,
              address_name = excluded.address_name,
              road_address_name = excluded.road_address_name,
              place_url = excluded.place_url,
              lat = excluded.lat,
              lng = excluded.lng,
              category_name = excluded.category_name,
              category_group_code = excluded.category_group_code,
              category_group_name = excluded.category_group_name,
              last_fetched_at = excluded.last_fetched_at
            """
        )

        self.db.execute(upsert, values)
        return True

    def search_nearby_in_db(
        self,
        *,
        query: str,
        lat: float,
        lng: float,
        radius_m: int,
        limit: int,
        allowed_category_group_codes: Optional[Set[str]] = None,
    ) -> List[Tuple[PlaceCacheRow, float]]:
        """
        DB place_cache에서 이름 ilike + 근처(대략 bbox 후 haversine) 검색.
        TTL 지난 row는 제외.
        Returns: List[(PlaceCacheRow, distance_m)]
        """
        q = (query or "").strip()
        if not q:
            return []

        limit = int(max(1, min(limit, 50)))
        radius_m = int(max(0, min(radius_m, 20000)))

        min_lat, max_lat, min_lng, max_lng = _bbox(lat, lng, float(radius_m))

        stmt = text(
            """
            select
              place_id, name, phone, address_name, road_address_name, place_url,
              lat, lng, category_name, category_group_code, category_group_name,
              last_fetched_at
            from place_cache
            where last_fetched_at >= (now() - (:ttl_s || ' seconds')::interval)
              and name ilike :name_like
              and lat between :min_lat and :max_lat
              and lng between :min_lng and :max_lng
            limit 800
            """
        )

        rows = self.db.execute(
            stmt,
            {
                "ttl_s": int(self.ttl_s),
                "name_like": f"%{q}%",
                "min_lat": float(min_lat),
                "max_lat": float(max_lat),
                "min_lng": float(min_lng),
                "max_lng": float(max_lng),
            },
        ).all()

        out: List[Tuple[PlaceCacheRow, float]] = []
        for r in rows:
            cgc = (r.category_group_code or "").strip()
            if allowed_category_group_codes and cgc not in allowed_category_group_codes:
                continue

            plat = float(r.lat)
            plng = float(r.lng)
            dist = _haversine_m(lat, lng, plat, plng)
            if dist > float(radius_m):
                continue

            row = PlaceCacheRow(
                place_id=str(r.place_id),
                name=r.name or "",
                phone=r.phone or "",
                address_name=r.address_name or "",
                road_address_name=r.road_address_name or "",
                place_url=r.place_url or "",
                lat=plat,
                lng=plng,
                category_name=r.category_name or "",
                category_group_code=cgc,
                category_group_name=r.category_group_name or "",
                last_fetched_at=r.last_fetched_at,
            )
            out.append((row, float(dist)))

        out.sort(key=lambda x: x[1])
        return out[:limit]

    # ✅ recommendations.py 용: category_group_code로 캐시만 빠르게 꺼내기
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
        Returns: List[dict] (recommendations.py가 바로 쓰는 형태)
        """
        radius_m = int(max(0, min(radius_m, 20000)))
        limit = int(max(1, min(limit, 50)))
        ttl_s = int(self.ttl_s if ttl_s is None else max(1, int(ttl_s)))

        cgc = (category_group_code or "").strip()
        if not cgc:
            return []

        min_lat, max_lat, min_lng, max_lng = _bbox(lat, lng, float(radius_m))

        stmt = text(
            """
            select
              place_id, name, phone, address_name, road_address_name, place_url,
              lat, lng, category_name, category_group_code, category_group_name,
              last_fetched_at
            from place_cache
            where category_group_code = :cgc
              and last_fetched_at >= (now() - (:ttl_s || ' seconds')::interval)
              and lat between :min_lat and :max_lat
              and lng between :min_lng and :max_lng
            limit 1200
            """
        )

        rows = self.db.execute(
            stmt,
            {
                "cgc": cgc,
                "ttl_s": ttl_s,
                "min_lat": float(min_lat),
                "max_lat": float(max_lat),
                "min_lng": float(min_lng),
                "max_lng": float(max_lng),
            },
        ).all()

        items: List[Dict[str, Any]] = []
        for r in rows:
            plat = float(r.lat)
            plng = float(r.lng)
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
