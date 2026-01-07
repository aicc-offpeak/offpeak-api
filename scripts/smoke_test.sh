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
python - <<'PY'
import os, json, httpx

base = os.getenv("BASE_URL", "http://127.0.0.1:8000")
user_lat, user_lng = 37.55687, 126.92378

# 1) search
s = httpx.get(
    f"{base}/places/search",
    params={"query":"스타벅스","lat":user_lat,"lng":user_lng,"size":5},
    timeout=15,
).json()

items = s.get("items") or []
assert items, f"places/search returned empty: {s}"
selected = items[0]

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
r = httpx.post(f"{base}/places/insight", json=payload, timeout=25)
print("status:", r.status_code)
out = r.json()
print(json.dumps(out, ensure_ascii=False, indent=2)[:1200])
assert r.status_code == 200, out
PY

echo
echo "[OK] smoke test done."
