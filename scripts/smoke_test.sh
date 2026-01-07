#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "[INFO] BASE_URL=$BASE_URL"

echo
echo "== / (expect 404 or 200) =="
curl --noproxy "*" -i --max-time 5 "$BASE_URL/" | head -n 12

echo
echo "== /docs (expect 200) =="
curl --noproxy "*" -I --max-time 5 "$BASE_URL/docs" | head -n 20

echo
echo "== /health (expect 200) =="
curl --noproxy "*" -i --max-time 5 "$BASE_URL/health" | head -n 20

echo
echo "== /v1/zones/nearby (expect 200) =="
curl --noproxy "*" -i --max-time 10 \
"$BASE_URL/v1/zones/nearby?lat=37.5665&lng=126.9780&radius_m=3000&top_k=5" | head -n 40

echo
echo "== /v1/recommendations (cafe) (expect 200) =="
curl --noproxy "*" -i --max-time 20 \
"$BASE_URL/v1/recommendations?lat=37.5665&lng=126.9780&category=cafe&radius_m=3000&max_results=10" | head -n 60

echo
echo "[OK] smoke test done."
