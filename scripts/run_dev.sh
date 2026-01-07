#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 1) venv
if [ ! -f ".venv/Scripts/activate" ]; then
  echo "[ERR] .venv not found. Run: python -m venv .venv"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/Scripts/activate

# 2) .env load (simple)
if [ -f ".env" ]; then
  echo "[INFO] Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[WARN] .env not found. Use .env.example to create one."
fi

# 3) proxy safety for localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

# 4) defaults
PORT="${PORT:-8000}"
ZONES_SEED_PATH="${ZONES_SEED_PATH:-app/resources/zones_seed.json}"
export PORT
export ZONES_SEED_PATH

# 5) safe length prints (avoid bad substitution)
KAKAO_LEN=${#KAKAO_REST_API_KEY}
SEOUL_LEN=${#SEOUL_OPENAPI_KEY}

echo "[INFO] Python: $(python -V)"
echo "[INFO] PORT=$PORT"
echo "[INFO] ZONES_SEED_PATH=$ZONES_SEED_PATH"
echo "[INFO] KAKAO_REST_API_KEY length=$KAKAO_LEN"
echo "[INFO] SEOUL_OPENAPI_KEY length=$SEOUL_LEN"

# 6) quick validation
if [ "$KAKAO_LEN" -eq 0 ]; then
  echo "[ERR] KAKAO_REST_API_KEY is empty. Check your .env"
  exit 1
fi
if [ "$SEOUL_LEN" -eq 0 ]; then
  echo "[ERR] SEOUL_OPENAPI_KEY is empty. Check your .env"
  exit 1
fi

# 7) run
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port "$PORT"
