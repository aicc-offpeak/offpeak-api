from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import CrowdingSnapshot

KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class HourStat:
    hour: int
    samples: int
    avg_rank: float


def _to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)


def load_recent_snapshots(
    db: Session,
    *,
    zone_code: str,
    days: int = 14,
    limit: int = 5000,
) -> List[CrowdingSnapshot]:
    since = datetime.now(tz=KST) - timedelta(days=int(days))
    stmt = (
        select(CrowdingSnapshot)
        .where(CrowdingSnapshot.zone_code == zone_code)
        .where(CrowdingSnapshot.ts >= since)
        .order_by(desc(CrowdingSnapshot.ts))
        .limit(int(limit))
    )
    return list(db.scalars(stmt).all())


def hourly_profile(rows: List[CrowdingSnapshot]) -> List[HourStat]:
    # hour -> (sum_rank, cnt)
    agg: Dict[int, Tuple[float, int]] = {}
    for r in rows:
        ts = _to_kst(r.ts)
        h = int(ts.hour)
        s, c = agg.get(h, (0.0, 0))
        agg[h] = (s + float(r.rank or 0), c + 1)

    stats: List[HourStat] = []
    for h in range(24):
        s, c = agg.get(h, (0.0, 0))
        avg = (s / c) if c else 0.0
        stats.append(HourStat(hour=h, samples=c, avg_rank=avg))
    return stats


def best_worst_hours(stats: List[HourStat], k: int = 3) -> Tuple[List[int], List[int]]:
    # 표본이 너무 적은 시간대는 제외(예: 0개)
    valid = [x for x in stats if x.samples > 0]
    if not valid:
        return [], []

    # 덜 붐빔: avg_rank 큰 순(4가 최고)
    best = sorted(valid, key=lambda x: (-x.avg_rank, -x.samples))[: int(k)]
    # 붐빔: avg_rank 작은 순(1이 최악)
    worst = sorted(valid, key=lambda x: (x.avg_rank, -x.samples))[: int(k)]

    return [x.hour for x in best], [x.hour for x in worst]


def trend_hint(rows: List[CrowdingSnapshot]) -> str:
    # rows는 최신순이라고 가정
    if len(rows) < 3:
        return ""

    r0, r1, r2 = rows[0], rows[1], rows[2]
    a = int(r0.rank or 0)
    b = int(r1.rank or 0)
    c = int(r2.rank or 0)

    # rank: 4(여유) -> 1(붐빔). 즉 값이 내려가면 더 붐비는 방향
    if a < b < c:
        return "최근 빠르게 붐비는 방향(혼잡 증가)으로 변하고 있어요."
    if a > b > c:
        return "최근 빠르게 여유로워지는 방향(혼잡 감소)으로 변하고 있어요."
    if a < b:
        return "조금 붐비는 방향으로 바뀌는 중이에요."
    if a > b:
        return "조금 여유로워지는 중이에요."
    return "혼잡도가 크게 변하지 않고 비슷해요."


def build_insight_payload(db: Session, *, zone_code: str, days: int = 14) -> Dict[str, Any]:
    rows = load_recent_snapshots(db, zone_code=zone_code, days=days)
    if not rows:
        return {
            "zone_code": zone_code,
            "days": int(days),
            "total_samples": 0,
            "message": "",
            "best_hours": [],
            "worst_hours": [],
            "hourly_profile": [],
            "trend": "",
        }

    stats = hourly_profile(rows)
    best, worst = best_worst_hours(stats, k=3)

    latest = rows[0]
    msg = (latest.message or "").strip()

    return {
        "zone_code": zone_code,
        "days": int(days),
        "total_samples": len(rows),
        "latest": {
            "ts": _to_kst(latest.ts).isoformat(),
            "level": latest.level,
            "rank": int(latest.rank or 0),
            "message": msg,
            "updated_at_epoch": int(latest.updated_at_epoch or 0),
        },
        "best_hours": best,
        "worst_hours": worst,
        "hourly_profile": [
            {"hour": x.hour, "samples": x.samples, "avg_rank": round(x.avg_rank, 2)} for x in stats
        ],
        "trend": trend_hint(rows),
        # "왜 붐비는지"는 메시지 + 피크 시간대로 설명(데이터가 더 필요하면 확장)
        "why_busy_hint": (msg + (f" (피크 시간대: {worst})" if worst else "")).strip(),
    }
