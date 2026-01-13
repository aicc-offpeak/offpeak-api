from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Zone(Base):
    __tablename__ = "zones"

    code: Mapped[str] = mapped_column(String(20), primary_key=True)  # POI013
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lng: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CrowdingSnapshot(Base):
    __tablename__ = "crowding_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    zone_code: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("zones.code", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)

    level: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    updated_at_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class PlaceCache(Base):
    __tablename__ = "place_cache"

    place_id: Mapped[str] = mapped_column(String(50), primary_key=True)  # Kakao place id

    name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    phone: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    address_name: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    road_address_name: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    place_url: Mapped[str] = mapped_column(String(500), nullable=False, default="")

    lat: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lng: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    category_name: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    category_group_code: Mapped[str] = mapped_column(String(10), nullable=False, default="")
    category_group_name: Mapped[str] = mapped_column(String(50), nullable=False, default="")

    last_fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class PlaceCrowdingSnapshot(Base):
    """
    "가게(place) 기준" 혼잡도 스냅샷.
    실제 혼잡도는 Zone의 혼잡도를 place에 매핑해서 기록한다.
    """
    __tablename__ = "place_crowding_snapshots"

    __table_args__ = (
        Index("ix_place_crowding_snapshots_place_id_ts", "place_id", "ts"),
        Index("ix_place_crowding_snapshots_zone_code_ts", "zone_code", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    place_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    place_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    category_group_code: Mapped[str | None] = mapped_column(String(10), nullable=True)

    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)

    zone_code: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("zones.code", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    zone_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    level: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    updated_at_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
