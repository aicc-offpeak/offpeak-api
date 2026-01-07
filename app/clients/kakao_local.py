# app/clients/kakao_local.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class KakaoLocalClient:
    """
    Kakao Local REST API client (Category Search).
    Docs endpoint: /v2/local/search/category.json
    """

    BASE_URL = "https://dapi.kakao.com"

    def __init__(self, rest_api_key: Optional[str] = None, timeout_s: float = 5.0) -> None:
        self.rest_api_key = rest_api_key or os.getenv("KAKAO_REST_API_KEY")
        if not self.rest_api_key:
            raise RuntimeError("KAKAO_REST_API_KEY is missing in environment variables.")
        self.timeout_s = timeout_s

    async def search_category(
        self,
        *,
        category_group_code: str,
        x: float,
        y: float,
        radius_m: int = 700,
        size: int = 15,
        page: int = 1,
        sort: str = "distance",
    ) -> List[Dict[str, Any]]:
        """
        Returns raw Kakao docs list.
        """
        url = f"{self.BASE_URL}/v2/local/search/category.json"
        headers = {"Authorization": f"KakaoAK {self.rest_api_key}"}
        params = {
            "category_group_code": category_group_code,  # CE7 (cafe), FD6 (food)
            "x": f"{x:.7f}",
            "y": f"{y:.7f}",
            "radius": int(max(0, min(radius_m, 20000))),  # Kakao radius max 20km
            "size": int(max(1, min(size, 15))),          # Kakao size max 15
            "page": int(max(1, min(page, 45))),          # Kakao page max 45
            "sort": sort,                                # distance | accuracy
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"Kakao Local API error {resp.status_code}: {resp.text}")
            data = resp.json()

        docs = data.get("documents", [])
        if not isinstance(docs, list):
            return []
        return docs
