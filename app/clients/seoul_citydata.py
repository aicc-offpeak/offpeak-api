# app/clients/seoul_citydata.py

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

# 단독 실행 테스트에서도 .env 읽게
load_dotenv()


@dataclass(frozen=True)
class SeoulCrowding:
    level: str
    message: str
    updated_at: int  # epoch seconds
    raw: Dict[str, Any]


def _parse_seoul_time_to_epoch(s: str) -> int:
    try:
        dt = datetime.strptime((s or "").strip(), "%Y-%m-%d %H:%M")
        return int(dt.timestamp())
    except Exception:
        return int(time.time())


def _extract_first_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    for v in (payload or {}).values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v[0]
    for v in (payload or {}).values():
        if isinstance(v, dict) and isinstance(v.get("row"), list) and v["row"]:
            if isinstance(v["row"][0], dict):
                return v["row"][0]
    return {}


class SeoulCityDataClient:
    BASE_URL = "http://openapi.seoul.go.kr:8088"
    SERVICE = "citydata_ppltn"

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 8.0) -> None:
        self.api_key = api_key or self._pick_api_key()
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _pick_api_key() -> str:
        # 표준: 이 키만 쓰는 걸 권장
        key = (os.getenv("SEOUL_OPENAPI_KEY") or "").strip()
        if key:
            return key

        # (호환용)
        for k in ("SEOUL_CITYDATA_API_KEY", "SEOUL_CITY_NOW_API_KEY", "SEOUL_CITY_AREA_API_KEY"):
            v = (os.getenv(k) or "").strip()
            if v:
                return v
        return ""

    def fetch_area_crowding(
        self,
        *,
        area_code: Optional[str] = None,
        area_name: Optional[str] = None,
    ) -> SeoulCrowding:
        if not self.api_key:
            return SeoulCrowding(level="", message="", updated_at=0, raw={})

        token = (area_code or "").strip() or (area_name or "").strip()
        if not token:
            return SeoulCrowding(level="", message="", updated_at=0, raw={})

        encoded = quote(token, safe="")
        url = f"{self.BASE_URL}/{self.api_key}/json/{self.SERVICE}/1/1/{encoded}"

        try:
            with httpx.Client(timeout=self.timeout_s, trust_env=False) as client:
                resp = client.get(url)
        except Exception:
            return SeoulCrowding(level="", message="", updated_at=0, raw={})

        if resp.status_code != 200:
            return SeoulCrowding(level="", message="", updated_at=0, raw={})

        try:
            data = resp.json()
        except Exception:
            return SeoulCrowding(level="", message="", updated_at=0, raw={})

        record = _extract_first_record(data)
        if not record:
            return SeoulCrowding(level="", message="", updated_at=0, raw={})

        result = record.get("RESULT")
        if isinstance(result, dict):
            code = result.get("RESULT.CODE") or result.get("CODE")
            if code and str(code) != "INFO-000":
                return SeoulCrowding(level="", message="", updated_at=0, raw=record)

        level = str(record.get("AREA_CONGEST_LVL") or record.get("AREA_CONGEST_LEVEL") or "").strip()
        message = str(record.get("AREA_CONGEST_MSG") or "").strip()
        t = str(record.get("PPLTN_TIME") or "").strip()
        updated_at = _parse_seoul_time_to_epoch(t) if t else int(time.time())

        return SeoulCrowding(level=level, message=message, updated_at=updated_at, raw=record)
