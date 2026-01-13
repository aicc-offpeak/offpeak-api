from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

DOW_NAMES_KO = ["월", "화", "수", "목", "금", "토", "일"]


class PlaceProfileService:
    """
    place_crowding_snapshots 기반
    - 요일(월~일) x 시간(0~23) 프로필
    - 초기 데이터가 sparse하므로 "표본 부족 셀"도 반환할 수 있게 함
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def weekly_hourly_profile(
        self,
        *,
        place_id: str,
        days: Optional[int] = None,          # 최근 N일만 (None이면 전체)
        min_samples: int = 3,                # reliable 기준
        tz: str = "Asia/Seoul",
        include_low_samples: bool = True,    # 표본 부족 셀도 반환
    ) -> Dict[str, Any]:
        place_id = (place_id or "").strip()
        if not place_id:
            return {
                "place_id": place_id,
                "tz": tz,
                "min_samples": int(min_samples),
                "days": days,
                "include_low_samples": bool(include_low_samples),
                "total_samples": 0,
                "cells": [],
                "matrix": [[None for _ in range(24)] for _ in range(7)],
            }

        where_days = ""
        params: Dict[str, Any] = {
            "place_id": place_id,
            "min_samples": int(min_samples),
            "tz": tz,
        }
        if days is not None:
            where_days = "AND pcs.ts >= (now() - (:days || ' days')::interval)"
            params["days"] = int(days)

        # Postgres extract(dow): 0=Sun..6=Sat
        # 우리가 원하는 dow: 0=Mon..6=Sun => (dow + 6) % 7
        sql = text(f"""
            WITH base AS (
                SELECT
                    ((extract(dow  from (pcs.ts AT TIME ZONE :tz))::int + 6) % 7) AS dow,
                    (extract(hour from (pcs.ts AT TIME ZONE :tz))::int)           AS hour,
                    COALESCE(pcs.rank, 0)::int                                    AS rank,
                    COALESCE(pcs.level, '')                                       AS level
                FROM place_crowding_snapshots pcs
                WHERE pcs.place_id = :place_id
                {where_days}
            ),
            total AS (
                SELECT count(*)::int AS n_total FROM base
            ),
            stats AS (
                SELECT
                    dow,
                    hour,
                    count(*)::int AS n,
                    avg(rank)::float AS avg_rank,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY rank)::float AS p50_rank
                FROM base
                GROUP BY dow, hour
            ),
            mode_level AS (
                SELECT
                    dow,
                    hour,
                    (array_agg(level ORDER BY cnt DESC, level ASC))[1] AS mode_level
                FROM (
                    SELECT dow, hour, level, count(*)::int AS cnt
                    FROM base
                    GROUP BY dow, hour, level
                ) t
                GROUP BY dow, hour
            )
            SELECT
                s.dow, s.hour, s.n, s.avg_rank, s.p50_rank,
                COALESCE(m.mode_level, '') AS mode_level,
                (SELECT n_total FROM total) AS total_samples
            FROM stats s
            LEFT JOIN mode_level m
              ON m.dow = s.dow AND m.hour = s.hour
            ORDER BY s.dow, s.hour
        """)

        rows = self.db.execute(sql, params).fetchall()

        total_samples = 0
        cells: List[Dict[str, Any]] = []
        matrix: List[List[Optional[Dict[str, Any]]]] = [[None for _ in range(24)] for _ in range(7)]

        for r in rows:
            dow = int(r[0])
            hour = int(r[1])
            n = int(r[2])
            total_samples = int(r[6] or 0)

            cell = {
                "dow": dow,
                "hour": hour,
                "n": n,
                "avg_rank": float(r[3]),
                "p50_rank": float(r[4]),
                "mode_level": str(r[5] or ""),
                "reliable": (n >= int(min_samples)),
            }

            if include_low_samples or cell["reliable"]:
                cells.append(cell)
                matrix[dow][hour] = cell

        return {
            "place_id": place_id,
            "tz": tz,
            "min_samples": int(min_samples),
            "days": days,
            "include_low_samples": bool(include_low_samples),
            "total_samples": int(total_samples),
            "cells": cells,
            "matrix": matrix,
        }

    def recommend_quiet_times(
        self,
        *,
        place_id: str,
        days: int = 30,
        min_samples: int = 3,
        tz: str = "Asia/Seoul",
        per_day: int = 3,
        window_h: int = 2,
        include_low_samples: bool = False,
        fallback_to_hourly: bool = True,
    ) -> Dict[str, Any]:
        place_id = (place_id or "").strip()
        if not place_id:
            return {
                "place_id": place_id,
                "tz": tz,
                "days": int(days),
                "min_samples": int(min_samples),
                "per_day": int(per_day),
                "window_h": int(window_h),
                "include_low_samples": bool(include_low_samples),
                "fallback_to_hourly": bool(fallback_to_hourly),
                "total_samples": 0,
                "recommendations": [],
                "note": "place_id is required",
            }

        prof = self.weekly_hourly_profile(
            place_id=place_id,
            days=int(days),
            min_samples=int(min_samples),
            tz=tz,
            include_low_samples=True,  # profile은 넓게 뽑고 추천에서 필터링
        )
        tz = str(prof.get("tz") or tz)
        cells: List[Dict[str, Any]] = list(prof.get("cells") or [])
        total_samples = int(prof.get("total_samples") or 0)

        if not cells:
            return {
                "place_id": place_id,
                "tz": tz,
                "days": int(days),
                "min_samples": int(min_samples),
                "per_day": int(per_day),
                "window_h": int(window_h),
                "include_low_samples": bool(include_low_samples),
                "fallback_to_hourly": bool(fallback_to_hourly),
                "total_samples": total_samples,
                "recommendations": [],
                "note": "no cells. collect more snapshots or increase days / lower min_samples",
            }

        # dow -> hour -> cell
        by_dow: Dict[int, Dict[int, Dict[str, Any]]] = {i: {} for i in range(7)}
        for c in cells:
            try:
                dow = int(c.get("dow"))
                hour = int(c.get("hour"))
            except Exception:
                continue
            if 0 <= dow <= 6 and 0 <= hour <= 23:
                by_dow[dow][hour] = c

        window_h = int(max(1, min(int(window_h), 6)))
        per_day = int(max(1, min(int(per_day), 10)))
        min_samples = int(max(1, int(min_samples)))

        def eligible_cell(c: Dict[str, Any]) -> bool:
            n = int(c.get("n") or 0)
            if include_low_samples:
                return n > 0
            return n >= min_samples

        def build_hourly_candidates(dow: int) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            for h, c in by_dow[dow].items():
                if not c or not eligible_cell(c):
                    continue
                items.append(
                    {
                        "dow": dow,
                        "dow_name": DOW_NAMES_KO[dow],
                        "start_hour": int(h),
                        "end_hour": int(h + 1),
                        "label": f"{h:02d}:00-{(h+1):02d}:00",
                        "avg_rank": float(c.get("avg_rank") or 0.0),
                        "n": int(c.get("n") or 0),
                        "hours": [int(h)],
                        "fallback": True,
                    }
                )
            items.sort(key=lambda w: (w["avg_rank"], math.log1p(float(w["n"]))), reverse=True)
            return items

        def build_window_candidate(dow: int, start: int) -> Optional[Dict[str, Any]]:
            hrs = list(range(start, start + window_h))
            if hrs[-1] >= 24:
                return None

            xs: List[float] = []
            total_n = 0
            for h in hrs:
                c = by_dow[dow].get(h)
                if not c or not eligible_cell(c):
                    return None
                xs.append(float(c.get("avg_rank") or 0.0))
                total_n += int(c.get("n") or 0)

            if not xs:
                return None

            avg_rank_win = sum(xs) / len(xs)
            return {
                "dow": dow,
                "dow_name": DOW_NAMES_KO[dow],
                "start_hour": int(start),
                "end_hour": int(start + window_h),
                "label": f"{start:02d}:00-{(start+window_h):02d}:00",
                "avg_rank": round(avg_rank_win, 3),
                "n": int(total_n),
                "hours": hrs,
                "fallback": False,
            }

        recommendations: List[Dict[str, Any]] = []

        for dow in range(7):
            windows: List[Dict[str, Any]] = []
            for start in range(0, 24 - window_h + 1):
                w = build_window_candidate(dow, start)
                if w:
                    windows.append(w)

            # window가 없으면 hourly fallback
            if not windows:
                if fallback_to_hourly:
                    hourly = build_hourly_candidates(dow)
                    picked = hourly[:per_day]
                    recommendations.append(
                        {
                            "dow": dow,
                            "dow_name": DOW_NAMES_KO[dow],
                            "windows": picked,
                            "note": "fallback_to_hourly" if picked else "no eligible hours",
                        }
                    )
                else:
                    recommendations.append(
                        {"dow": dow, "dow_name": DOW_NAMES_KO[dow], "windows": [], "note": "no eligible windows"}
                    )
                continue

            windows.sort(key=lambda w: (w["avg_rank"], math.log1p(float(w["n"]))), reverse=True)

            # 겹치는 시간대는 빼고 per_day개 선택
            picked: List[Dict[str, Any]] = []
            used_hours = set()
            for w in windows:
                if len(picked) >= per_day:
                    break
                if any(h in used_hours for h in w["hours"]):
                    continue
                picked.append(w)
                for h in w["hours"]:
                    used_hours.add(h)

            recommendations.append({"dow": dow, "dow_name": DOW_NAMES_KO[dow], "windows": picked})

        return {
            "place_id": place_id,
            "tz": tz,
            "days": int(days),
            "min_samples": int(min_samples),
            "per_day": int(per_day),
            "window_h": int(window_h),
            "include_low_samples": bool(include_low_samples),
            "fallback_to_hourly": bool(fallback_to_hourly),
            "total_samples": total_samples,
            "recommendations": recommendations,
        }
