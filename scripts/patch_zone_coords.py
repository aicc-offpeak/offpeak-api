#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch a zone's coordinates inside zones_seed.json using Kakao keyword search.

Usage:
  set -a; source .env; set +a
  python scripts/patch_zone_coords.py --code POI007 --query "홍대입구역"
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import httpx

KAKAO_BASE_URL = "https://dapi.kakao.com"


def kakao_keyword_top(query: str, rest_api_key: str) -> Tuple[float, float, str]:
    url = f"{KAKAO_BASE_URL}/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {rest_api_key}"}
    params = {"query": query, "size": 1, "sort": "accuracy"}

    # trust_env=False: ignore HTTP_PROXY/HTTPS_PROXY env vars
    with httpx.Client(timeout=8.0, trust_env=False) as client:
        r = client.get(url, headers=headers, params=params)

    if r.status_code != 200:
        raise RuntimeError(f"Kakao keyword search failed: {r.status_code} {r.text}")

    data = r.json()
    docs = data.get("documents") or []
    if not docs:
        raise RuntimeError(f"No Kakao search result for query='{query}'")

    top = docs[0]
    lat = float(top["y"])
    lng = float(top["x"])
    place_name = str(top.get("place_name") or "")
    return lat, lng, place_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="app/resources/zones_seed.json")
    ap.add_argument("--code", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    kakao_key = os.getenv("KAKAO_REST_API_KEY")
    if not kakao_key:
        raise RuntimeError("KAKAO_REST_API_KEY is missing. Load .env or export it in this terminal.")

    with open(args.seed, "r", encoding="utf-8") as f:
        data = json.load(f)

    target: Optional[Dict[str, Any]] = None
    for z in data:
        if z.get("code") == args.code:
            target = z
            break
    if not target:
        raise RuntimeError(f"Zone code not found in seed: {args.code}")

    old_lat, old_lng = target.get("lat"), target.get("lng")
    lat, lng, place_name = kakao_keyword_top(args.query, kakao_key)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{args.seed}.bak.{ts}"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    target["lat"] = lat
    target["lng"] = lng

    with open(args.seed, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Patched {args.code}: {target.get('name')}")
    print(f"     old=({old_lat}, {old_lng})")
    print(f"     new=({lat}, {lng}) from Kakao='{place_name}' (query='{args.query}')")
    print(f"[OK] Backup: {backup_path}")


if __name__ == "__main__":
    main()
