# scripts/seed_zones_db.py
from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Zone


def main() -> None:
    seed_path = os.getenv("ZONES_SEED_PATH", "app/resources/zones_seed.json")
    zones = json.loads(Path(seed_path).read_text(encoding="utf-8"))

    with SessionLocal() as db:
        count_new = 0
        count_upd = 0

        for z in zones:
            code = (z.get("code") or "").strip()
            if not code:
                continue

            name = (z.get("name") or "").strip()
            lat = float(z.get("lat") or 0.0)
            lng = float(z.get("lng") or 0.0)
            extra = z.get("extra") or {}
            category = (extra.get("CATEGORY") or "").strip() or None

            row = db.scalar(select(Zone).where(Zone.code == code))
            if row is None:
                db.add(
                    Zone(
                        code=code,
                        name=name,
                        lat=lat,
                        lng=lng,
                        category=category,
                        extra=extra,
                    )
                )
                count_new += 1
            else:
                row.name = name
                row.lat = lat
                row.lng = lng
                row.category = category
                row.extra = extra
                count_upd += 1

        db.commit()

    print(f"[OK] seeded zones: new={count_new}, updated={count_upd}")


if __name__ == "__main__":
    main()
