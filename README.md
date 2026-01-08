# OffPeak API (FastAPI)

서울 실시간 혼잡도(인구) 데이터를 기반으로 **주변 Zone의 혼잡도**를 확인하고,  
선택한 장소(카페/음식점)가 붐비는 경우 **덜 붐비는 대체 장소**를 추천하는 백엔드 API 입니다.

- 혼잡도 데이터: 서울 열린데이터광장 실시간 도시데이터(citydata_ppltn)
- 장소 검색/추천: 카카오 로컬 API(키워드/카테고리 검색)

> ✅ 현재 API는 `/v1` prefix 없이 동작합니다.  
> (예: `/recommendations`, `/places/search`)

---

## Requirements
- Python 3.11+
- Windows: Git Bash 또는 PowerShell

---

## Project Structure
- `app/` : FastAPI app, routes, services, clients
- `app/resources/zones_seed.json` : 서울 주요 장소(Zone) seed
- `scripts/` : seed 생성/테스트 스크립트

---

## 1) Setup

### Windows PowerShell
```powershell
cd C:\boom.b\offpeak-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Git Bash
```bash
cd /c/boom.b/offpeak-api
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### .env에 아래 값 입력:
- KAKAO_REST_API_KEY : 카카오 로컬(키워드/카테고리 검색) REST API 키
- SEOUL_OPENAPI_KEY : 서울 열린데이터광장 "실시간 인구/혼잡도(citydata_ppltn)" 호출 가능한 인증키
- ZONES_SEED_PATH : 기본값 사용 가능 (app/resources/zones_seed.json)

✅ .env는 커밋 금지입니다. (.gitignore에 포함)

### Optional env vars (tuning)
- CROWDING_CACHE_TTL_S : 혼잡도 캐시 TTL(초)
- TOP_ZONES : 주변 zone 후보 개수
- PER_ZONE : zone당 추천 POI 개수
- ZONE_SEARCH_RADIUS_M : zone 중심 기준 POI 검색 반경(m)

---

## 2) Run (Git Bash)

```bash
cd /c/boom.b/offpeak-api
source .venv/Scripts/activate
set -a; source .env; set +a
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 3) API Docs / Health
- Swagger: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- Health: http://127.0.0.1:8000/health

---

## 4) Quick Test (cURL)

⚠️ Git Bash에서 한글 query는 인코딩 이슈가 있을 수 있습니다.  
아래 예시는 python으로 URL 인코딩 후 호출합니다.

### 추천(카페)
```bash
curl --noproxy "*" -sS \
"http://127.0.0.1:8000/recommendations?lat=37.5665&lng=126.9780&category=cafe&radius_m=3000&max_results=10" \
| python -m json.tool
```

### 추천(음식점)
```bash
curl --noproxy "*" -sS \
"http://127.0.0.1:8000/recommendations?lat=37.5665&lng=126.9780&category=restaurant&radius_m=3000&max_results=10" \
| python -m json.tool
```

### 검색 리스트 (가까운 순 5개)
```bash
Q=$(python - <<'PY'
import urllib.parse
print(urllib.parse.quote("스타벅스"))
PY
)
curl --noproxy "*" -sS \
"http://127.0.0.1:8000/places/search?query=$Q&lat=37.55687&lng=126.92378&size=5" \
| python -m json.tool
```

---

## 5) Smoke Test (Git Bash)

서버가 켜진 상태에서 실행:
```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

Smoke test는 아래를 확인합니다:
- `/health` 200
- `/zones/nearby` 200
- `/recommendations` 200
- `/places/search` 200
- `/places/insight` 200

