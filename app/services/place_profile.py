from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session


class PlaceProfileService:
    """
    place_crowding_snapshots 기반
    - 요일(월~일) x 시간(0~23) 프로필
    - 추천(요일별 덜 붐비는 시간대): window(연속 시간) 기반 + 부족하면 hourly로 fallback
    - 원인 분석 문장(reason): 숫자(rank/Δ/n/신뢰도) 노출 없이 사용자 친화 문장으로 생성
    """

    DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
    DOW_FULL = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

    def __init__(self, db: Session) -> None:
        self.db = db

    # -------------------------
    # Profile
    # -------------------------
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

    # -------------------------
    # Recommend (quiet times)
    # -------------------------
    def recommend_quiet_times(
        self,
        *,
        place_id: str,
        days: int = 7,
        min_samples: int = 3,
        per_day: int = 3,
        window_h: int = 2,
        tz: str = "Asia/Seoul",
        include_low_samples: bool = False,
    ) -> Dict[str, Any]:
        place_id = (place_id or "").strip()
        days = int(max(1, min(days, 365)))
        min_samples = int(max(1, min(min_samples, 100)))
        per_day = int(max(1, min(per_day, 10)))
        window_h = int(max(1, min(window_h, 6)))

        prof_vis = self.weekly_hourly_profile(
            place_id=place_id,
            days=days,
            min_samples=min_samples,
            tz=tz,
            include_low_samples=bool(include_low_samples),
        )
        prof_all = self.weekly_hourly_profile(
            place_id=place_id,
            days=days,
            min_samples=1,
            tz=tz,
            include_low_samples=True,
        )

        recent_days = 7 if days >= 7 else days
        prof_recent = self.weekly_hourly_profile(
            place_id=place_id,
            days=recent_days,
            min_samples=1,
            tz=tz,
            include_low_samples=True,
        )

        total_samples = int(prof_all.get("total_samples") or 0)
        if total_samples <= 0:
            return {
                "place_id": place_id,
                "tz": tz,
                "days": days,
                "min_samples": min_samples,
                "per_day": per_day,
                "window_h": window_h,
                "include_low_samples": bool(include_low_samples),
                "fallback_to_hourly": True,
                "total_samples": 0,
                "recommendations": [
                    {"dow": d, "dow_name": self.DOW_NAMES[d], "windows": [], "note": "no data"}
                    for d in range(7)
                ],
            }

        dow_baseline = self._weighted_dow_baseline(prof_all)

        recs, used_fallback = self._build_recommendations(
            prof_vis=prof_vis,
            prof_all=prof_all,
            prof_recent=prof_recent,
            dow_baseline=dow_baseline,
            per_day=per_day,
            window_h=window_h,
            min_samples=min_samples,
            include_low_samples=bool(include_low_samples),
            days=days,
            recent_days=recent_days,
        )

        return {
            "place_id": place_id,
            "tz": tz,
            "days": days,
            "min_samples": min_samples,
            "per_day": per_day,
            "window_h": window_h,
            "include_low_samples": bool(include_low_samples),
            "fallback_to_hourly": bool(used_fallback),
            "total_samples": int(total_samples),
            "recommendations": recs,
        }

    # -------------------------
    # Internals
    # -------------------------
    def _weighted_dow_baseline(self, prof_all: Dict[str, Any]) -> Dict[int, float]:
        baseline: Dict[int, float] = {}
        sum_w: Dict[int, float] = {d: 0.0 for d in range(7)}
        sum_x: Dict[int, float] = {d: 0.0 for d in range(7)}

        for c in (prof_all.get("cells") or []):
            try:
                d = int(c["dow"])
                n = float(c.get("n") or 0)
                x = float(c.get("avg_rank") or 0.0)
            except Exception:
                continue
            if 0 <= d <= 6 and n > 0:
                sum_w[d] += n
                sum_x[d] += n * x

        for d in range(7):
            baseline[d] = (sum_x[d] / sum_w[d]) if sum_w[d] > 0 else 0.0
        return baseline

    def _build_recommendations(
        self,
        *,
        prof_vis: Dict[str, Any],
        prof_all: Dict[str, Any],
        prof_recent: Dict[str, Any],
        dow_baseline: Dict[int, float],
        per_day: int,
        window_h: int,
        min_samples: int,
        include_low_samples: bool,
        days: int,
        recent_days: int,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        matrix_vis = prof_vis.get("matrix") or [[None for _ in range(24)] for _ in range(7)]
        matrix_recent = prof_recent.get("matrix") or [[None for _ in range(24)] for _ in range(7)]

        def build_windows_for_dow(dow: int, wh: int) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for start in range(0, 24 - wh + 1):
                hours = list(range(start, start + wh))

                cells = []
                ok = True
                for h in hours:
                    cell = matrix_vis[dow][h]
                    if cell is None:
                        ok = False
                        break
                    if (not include_low_samples) and (not bool(cell.get("reliable"))):
                        ok = False
                        break
                    cells.append(cell)

                if not ok or not cells:
                    continue

                avg_rank = sum(float(c.get("avg_rank") or 0.0) for c in cells) / float(len(cells))
                n_eff = min(int(c.get("n") or 0) for c in cells)

                mode_level = ""
                for c in cells:
                    lv = (c.get("mode_level") or "").strip()
                    if lv:
                        mode_level = lv
                        break

                out.append(
                    {
                        "dow": dow,
                        "dow_name": self.DOW_NAMES[dow],
                        "start_hour": start,
                        "end_hour": start + wh,
                        "label": f"{start:02d}:00-{(start + wh):02d}:00",
                        "avg_rank": float(avg_rank),
                        "n": int(n_eff),
                        "hours": hours,
                        "mode_level": mode_level,
                        "fallback": False,
                    }
                )

            out.sort(key=lambda w: (-float(w["avg_rank"]), -int(w["n"]), int(w["start_hour"])))
            return out[:per_day]

        # 1) window_h
        recs: List[Dict[str, Any]] = []
        any_windows = False
        for d in range(7):
            wins = build_windows_for_dow(d, window_h)
            if wins:
                any_windows = True
            recs.append({"dow": d, "dow_name": self.DOW_NAMES[d], "windows": wins})

        # 2) fallback to hourly
        used_fallback = False
        if (window_h > 1) and (not any_windows):
            used_fallback = True
            recs = []
            for d in range(7):
                wins = build_windows_for_dow(d, 1)
                for w in wins:
                    w["fallback"] = True
                recs.append({"dow": d, "dow_name": self.DOW_NAMES[d], "windows": wins})

        # 3) reason
        for day in recs:
            d = int(day["dow"])
            wins = day.get("windows") or []
            if not wins:
                day["note"] = "no eligible hours" if used_fallback else "no eligible windows"
                continue

            base_avg = float(dow_baseline.get(d, 0.0) or 0.0)

            for w in wins:
                delta = float(w["avg_rank"]) - base_avg

                recent_avg = self._recent_window_avg(
                    dow=d,
                    hours=w["hours"],
                    matrix_recent=matrix_recent,
                )
                trend = None
                if recent_avg is not None:
                    trend = float(recent_avg) - float(w["avg_rank"])

                conf = self._confidence_bucket(int(w.get("n") or 0), min_samples=min_samples)
                w["confidence"] = conf

                # 숫자(baseline/delta)는 응답에 넣지 않음
                w["reason"] = self._make_reason(
                    dow=d,
                    time_label=str(w["label"]),
                    mode_level=str(w.get("mode_level") or ""),
                    delta=delta,
                    trend_delta=trend,
                    days=days,
                    recent_days=recent_days,
                    confidence=conf,
                )

        return recs, used_fallback

    def _recent_window_avg(
        self,
        *,
        dow: int,
        hours: List[int],
        matrix_recent: List[List[Optional[Dict[str, Any]]]],
    ) -> Optional[float]:
        vals: List[float] = []
        for h in hours:
            cell = matrix_recent[dow][h]
            if cell is None:
                continue
            try:
                vals.append(float(cell.get("avg_rank") or 0.0))
            except Exception:
                continue
        if not vals:
            return None
        return sum(vals) / float(len(vals))

    def _confidence_bucket(self, n_eff: int, *, min_samples: int) -> str:
        if n_eff >= max(5, min_samples * 2):
            return "high"
        if n_eff >= min_samples:
            return "medium"
        return "low"

    @staticmethod
    def _days_label(days: int) -> str:
        if days == 7:
            return "일주일"
        if days == 14:
            return "2주"
        if days == 30:
            return "한 달"
        return f"{days}일"

    def _make_reason(
        self,
        *,
        dow: int,
        time_label: str,
        mode_level: str,
        delta: float,
        trend_delta: Optional[float],
        days: int,
        recent_days: int,
        confidence: str,
    ) -> str:
        dow_name = self.DOW_FULL[dow] if 0 <= dow <= 6 else ""

        # 1) 평소 대비
        if delta >= 0.8:
            base_msg = "평소보다 확실히 한산한 편이에요"
        elif delta >= 0.3:
            base_msg = "평소보다 덜 붐비는 편이에요"
        elif delta <= -0.3:
            base_msg = "평소보다 붐빌 수 있어요"
        else:
            base_msg = "평소와 비슷한 편이에요"

        # 2) 최근 흐름
        recent_label = self._days_label(int(recent_days))
        if trend_delta is None:
            trend_msg = f"최근 {recent_label} 흐름은 아직 판단하기 어려워요"
        else:
            if trend_delta >= 0.5:
                trend_msg = f"최근 {recent_label}은 이 시간대가 더 한산한 흐름이었어요"
            elif trend_delta <= -0.5:
                trend_msg = f"최근 {recent_label}은 이 시간대가 조금 더 붐빈 흐름이었어요"
            else:
                trend_msg = f"최근 {recent_label} 흐름도 크게 다르지 않았어요"

        # 3) 데이터 충분도(숫자 없이)
        if confidence == "low":
            conf_msg = " 다만 데이터가 아직 적어 참고용이에요"
        elif confidence == "medium":
            conf_msg = " 데이터는 보통 수준이에요"
        else:
            conf_msg = ""

        days_label = self._days_label(int(days))
        return f"최근 {days_label} 동안 {dow_name} 기준 {time_label}은 {base_msg}, {trend_msg}.{conf_msg}"
