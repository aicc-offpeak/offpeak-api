#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "[INFO] BASE_URL=$BASE_URL"

echo
echo "== / (expect 404 or 200) =="
curl --noproxy "*" -i --max-time 5 "$BASE_URL/" || true

echo
echo "== /docs (expect 200) =="
curl --noproxy "*" -i --max-time 5 "$BASE_URL/docs" | head -n 15 || true

echo
echo "== /health (expect 200) =="
curl --noproxy "*" -i --max-time 5 "$BASE_URL/health"

echo
echo "== /openapi.json (expect 200) =="
curl --noproxy "*" -i --max-time 5 "$BASE_URL/openapi.json" | head -n 20 || true

echo
echo "== /zones/nearby (expect 200) =="
curl --noproxy "*" -i --max-time 10 \
  "$BASE_URL/zones/nearby?lat=37.5665&lng=126.9780&radius_m=2000&top_k=5"

echo
echo "== /recommendations (cafe) (expect 200) =="
curl --noproxy "*" -i --max-time 20 \
  "$BASE_URL/recommendations?lat=37.5665&lng=126.9780&category=cafe&radius_m=3000&max_results=5"

echo
echo "== /places/search (expect 200) =="
# Git Bash 한글 인코딩 이슈 방지: python으로 UTF-8 percent-encoding
Q="$(python - <<'PY'
import urllib.parse
print(urllib.parse.quote("스타벅스"))
PY
)"
curl --noproxy "*" -i --max-time 15 \
  "$BASE_URL/places/search?query=$Q&lat=37.55687&lng=126.92378&size=5"

echo
echo "== /places/insight (expect 200, recommend_from=user) =="
# 안정성: 스타벅스가 0개면 대체 키워드로 fallback
python - <<'PY'
import os, json
import httpx

base = os.getenv("BASE_URL", "http://127.0.0.1:8000")
user_lat, user_lng = 37.55687, 126.92378

client = httpx.Client(timeout=20)

# 1) search (fallback queries)
queries = ["스타벅스", "카페", "커피", "투썸플레이스"]
selected = None
last = None

for q in queries:
    last = client.get(
        f"{base}/places/search",
        params={"query": q, "lat": user_lat, "lng": user_lng, "size": 5},
    ).json()
    items = last.get("items") or []
    if items:
        selected = items[0]
        break

assert selected is not None, f"places/search returned empty for all queries. last={last}"

# 2) insight (user 기준 추천)
payload = {
  "selected": selected,
  "user_lat": user_lat,
  "user_lng": user_lng,
  "recommend_from": "user",
  "radius_m": 1200,
  "max_candidates": 25,
  "max_alternatives": 3,
  "category": "cafe",
}
r = client.post(f"{base}/places/insight", json=payload)
print("status:", r.status_code)
out = r.json()
print(json.dumps(out, ensure_ascii=False, indent=2)[:1200])
assert r.status_code == 200, out
PY

echo
echo "[OK] smoke test done."
