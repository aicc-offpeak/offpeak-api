# app/routes/zones.py
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import CrowdingSnapshot as CrowdingSnapshotModel, Zone
from app.services.crowding import CrowdingService
from app.services.crowding_insight import build_insight_payload

router = APIRouter(prefix="/zones", tags=["zones"])
KST = ZoneInfo("Asia/Seoul")


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _get_latest_snapshot(db: Session, zone_code: str) -> CrowdingSnapshotModel | None:
    stmt = (
        select(CrowdingSnapshotModel)
        .where(CrowdingSnapshotModel.zone_code == zone_code)
        .order_by(desc(CrowdingSnapshotModel.ts))
        .limit(1)
    )
    return db.scalar(stmt)


def _is_fresh(row: CrowdingSnapshotModel, min_interval_s: int) -> bool:
    ts = row.ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
    return age_s < float(min_interval_s)


@router.get("/nearby")
def nearby_zones(
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    radius_m: int = Query(3000, ge=200, le=20000),
    top_k: int = Query(8, ge=1, le=30),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    내 위치 기준 가까운 Zone을 DB(zones)에서 찾고,
    혼잡도는 crowding_snapshots 최신 스냅샷을 재사용하거나 갱신한다.
    """
    min_interval_s = int(os.getenv("CROWDING_SNAPSHOT_MIN_INTERVAL_S", "600"))  # 10분 기본
    crowding = CrowdingService()

    zones: List[Zone] = list(db.scalars(select(Zone)).all())

    # 1) 거리 계산 + 반경 필터
    candidates: List[Tuple[Zone, float]] = []
    for z in zones:
        d = _haversine_m(lat, lng, float(z.lat), float(z.lng))
        if d <= float(radius_m):
            candidates.append((z, d))

    # 2) 가까운 순 정렬 후 top_k
    candidates.sort(key=lambda x: x[1])
    candidates = candidates[: int(top_k)]

    # 3) crowding 스냅샷 확보(재사용 or 새로 저장)
    out: List[Dict[str, Any]] = []
    wrote_any = False

    for z, dist in candidates:
        latest = _get_latest_snapshot(db, z.code)

        if latest and _is_fresh(latest, min_interval_s):
            snap_level = latest.level or ""
            snap_rank = int(latest.rank or 0)
            snap_color = (latest.raw or {}).get("color") or ""
            snap_msg = latest.message or ""
            snap_updated = int(latest.updated_at_epoch or 0)
        else:
            dto = crowding.get(area_name=z.name, area_code=z.code)
            snap_level = dto.level or ""
            snap_rank = int(dto.rank or 0)
            snap_color = dto.color or ""
            snap_msg = dto.message or ""
            snap_updated = int(dto.updated_at_epoch or 0)

            db.add(
                CrowdingSnapshotModel(
                    zone_code=z.code,
                    level=snap_level,
                    rank=snap_rank,
                    message=snap_msg,
                    updated_at_epoch=snap_updated,
                    # 앞으로 "왜 붐비는지"를 위해 seoul raw를 같이 저장
                    raw={"color": snap_color, "seoul": dto.raw},
                )
            )
            wrote_any = True

        out.append(
            {
                "code": z.code,
                "name": z.name,
                "lat": float(z.lat),
                "lng": float(z.lng),
                "distance_m": round(dist, 1),
                "crowding_level": snap_level,
                "crowding_rank": snap_rank,
                "crowding_color": snap_color or "unknown",
                "crowding_message": snap_msg,
                "crowding_updated_at": snap_updated,
            }
        )

    if wrote_any:
        db.commit()

    return {"items": out}


@router.get("/{code}/insight")
def zone_insight(
    code: str,
    days: int = Query(7, ge=1, le=30, description="최근 N일 스냅샷으로 시간대 분석"),
    top_n: int = Query(3, ge=1, le=8, description="추천 시간대 상위 N개"),
    min_samples_per_hour: int = Query(2, ge=1, le=50, description="시간대별 최소 샘플 수"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    crowding_snapshots(시계열)을 기반으로
    - 시간대별 평균 rank(여유=4 ... 붐빔=1)를 계산하고
    - 덜 붐비는 시간대 TOP N을 반환
    """
    zone = db.get(Zone, code)
    if not zone:
        raise HTTPException(status_code=404, detail=f"zone not found: {code}")

    since = datetime.now(timezone.utc) - timedelta(days=int(days))

    rows = db.execute(
        select(CrowdingSnapshotModel.ts, CrowdingSnapshotModel.level, CrowdingSnapshotModel.rank)
        .where(
            CrowdingSnapshotModel.zone_code == code,
            CrowdingSnapshotModel.ts >= since,
        )
        .order_by(CrowdingSnapshotModel.ts.asc())
    ).all()

    if not rows:
        return {
            "zone": {"code": zone.code, "name": zone.name, "lat": float(zone.lat), "lng": float(zone.lng)},
            "days": int(days),
            "samples": 0,
            "by_hour": [],
            "recommended_hours": [],
            "message": "최근 스냅샷이 없습니다. collect_crowding_snapshots를 먼저 실행해 데이터를 쌓아주세요.",
        }

    bucket = defaultdict(list)  # hour -> list[(rank, level)]
    for ts, level, rank in rows:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        kst = ts.astimezone(KST)
        hour = int(kst.hour)
        bucket[hour].append((int(rank or 0), (level or "").strip()))

    by_hour: list[dict[str, Any]] = []
    for h in range(24):
        items = bucket.get(h, [])
        cnt = len(items)
        if cnt == 0:
            by_hour.append({"hour": h, "count": 0, "avg_rank": 0.0, "dominant_level": ""})
            continue

        ranks = [r for r, _ in items]
        levels = [lv for _, lv in items if lv]
        avg_rank = round(sum(ranks) / cnt, 3)
        dominant = Counter(levels).most_common(1)[0][0] if levels else ""

        by_hour.append({"hour": h, "count": cnt, "avg_rank": avg_rank, "dominant_level": dominant})

    # 추천 시간대: avg_rank 높은 순 -> 표본 많은 순
    candidates = [x for x in by_hour if x["count"] >= int(min_samples_per_hour)]
    candidates.sort(key=lambda x: (x["avg_rank"], x["count"]), reverse=True)
    best = candidates[: int(top_n)]

    recommended_hours = [
        {
            "hour": x["hour"],
            "label": f"{x['hour']:02d}:00~{x['hour']:02d}:59",
            "avg_rank": x["avg_rank"],
            "samples": x["count"],
            "dominant_level": x["dominant_level"],
        }
        for x in best
    ]

    return {
        "zone": {"code": zone.code, "name": zone.name, "lat": float(zone.lat), "lng": float(zone.lng)},
        "days": int(days),
        "samples": len(rows),
        "by_hour": by_hour,
        "recommended_hours": recommended_hours,
        "note": "rank는 여유=4, 보통=3, 약간 붐빔=2, 붐빔=1 (최근 스냅샷 기반)",
    }


@router.get("/{code}/crowding/history")
def crowding_history(
    code: str,
    hours: int = Query(24, ge=1, le=24 * 14),
    limit: int = Query(200, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    특정 zone의 혼잡도 스냅샷 히스토리(JSON 직렬화 안전)
    """
    since = datetime.now(tz=KST) - timedelta(hours=int(hours))

    stmt = (
        select(CrowdingSnapshotModel)
        .where(CrowdingSnapshotModel.zone_code == code)
        .where(CrowdingSnapshotModel.ts >= since)
        .order_by(desc(CrowdingSnapshotModel.ts))
        .limit(int(limit))
    )

    rows = list(db.scalars(stmt).all())

    items: List[Dict[str, Any]] = []
    for r in rows:
        ts = r.ts
        if ts is None:
            ts_s = ""
        else:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=KST)
            ts_s = ts.astimezone(KST).isoformat()

        items.append(
            {
                "zone_code": r.zone_code,
                "ts": ts_s,
                "level": r.level or "",
                "rank": int(r.rank or 0),
                "message": r.message or "",
                "updated_at_epoch": int(r.updated_at_epoch or 0),
            }
        )

    return {
        "zone_code": code,
        "hours": int(hours),
        "limit": int(limit),
        "count": len(items),
        "items": items,
    }


@router.get("/{code}/crowding/insight")
def crowding_insight(
    code: str,
    days: int = Query(14, ge=1, le=60),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return build_insight_payload(db, zone_code=code, days=int(days))
