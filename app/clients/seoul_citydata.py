# app/clients/seoul_citydata.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import httpx


class SeoulCityDataClient:
    """
    Seoul OpenAPI 'citydata_ppltn' client.

    Observed response shape (2026-01-07):
      {
        "SeoulRtd.citydata_ppltn": [
          {
            "AREA_NM": "...",
            "AREA_CD": "POI009",
            "AREA_CONGEST_LVL": "약간 붐빔",
            "PPLTN_TIME": "YYYY-MM-DD HH:MM",
            "FCST_YN": "Y",
            "FCST_PPLTN": [ ... ],
            "RESULT": {"RESULT.CODE":"INFO-000", "RESULT.MESSAGE":"정상 처리되었습니다."}
          }
        ]
      }

    URL pattern:
      http://openapi.seoul.go.kr:8088/{KEY}/json/citydata_ppltn/1/5/{AREA_NM}
    """

    BASE_URL = "http://openapi.seoul.go.kr:8088"

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 8.0) -> None:
        self.api_key = api_key or os.getenv("SEOUL_OPENAPI_KEY")
        if not self.api_key:
            raise RuntimeError("SEOUL_OPENAPI_KEY is missing in environment variables.")
        self.timeout_s = timeout_s

    async def fetch_area_crowding(self, *, area_name: str) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (crowding_level, raw_dict).
        crowding_level: '여유' | '보통' | '약간 붐빔' | '붐빔' | '정보없음'
        raw_dict: the first record dict (includes FCST_PPLTN, etc.)
        """
        # URL-encode Korean safely (handles '·' too)
        encoded = quote(area_name, safe="")
        url = f"{self.BASE_URL}/{self.api_key}/json/citydata_ppltn/1/5/{encoded}"

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Seoul OpenAPI HTTP {resp.status_code}: {resp.text}")
            data = resp.json()

        record = _extract_first_record(data)

        # Handle RESULT (some responses put RESULT inside record)
        result = record.get("RESULT")
        if isinstance(result, dict):
            code = result.get("RESULT.CODE") or result.get("CODE")
            if code and str(code) != "INFO-000":
                # still return record for debugging
                raise RuntimeError(f"Seoul OpenAPI RESULT not OK: {result}")

        crowding = (
            record.get("AREA_CONGEST_LVL")
            or record.get("AREA_CONGEST_LEVEL")
            or record.get("area_congest_lvl")
            or "정보없음"
        )
        return str(crowding), record


def _extract_first_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robustly find the first data record from various Seoul OpenAPI shapes:
    - {"SeoulRtd.citydata_ppltn": [ { ... } ]}
    - {"citydata_ppltn": {"row":[{...}]}}
    - {"something": {"row":[...]}}
    """
    # 1) most common in your output: a key that maps to a list of dicts
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v[0]

    # 2) sometimes nested dict with "row"
    for v in payload.values():
        if isinstance(v, dict) and isinstance(v.get("row"), list) and v["row"]:
            if isinstance(v["row"][0], dict):
                return v["row"][0]

    # 3) fallback: empty dict
    return {}
