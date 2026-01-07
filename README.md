# OffPeak API (FastAPI)

서울 실시간 혼잡도(인구) 데이터를 기반으로 "덜 붐비는 Zone"을 고르고, 그 주변의 카페/음식점을 카카오 로컬 API로 추천하는 백엔드 API 입니다.

---

## Requirements
- Python 3.11+
- Windows: Git Bash 또는 PowerShell

---

## Project Structure
- `app/` : FastAPI app, routes, services, clients
- `app/resources/zones_seed.json` : 서울 주요 장소(Zone) seed (120개)
- `scripts/` : seed 생성 스크립트 등 유틸

---

## 1. Setup

### Windows PowerShell
```powershell
cd C:\boom.b\offpeak-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
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

## 2. Run

### Git Bash
```bash
cd /c/boom.b/offpeak-api
source .venv/Scripts/activate
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### PowerShell
```powershell
cd C:\boom.b\offpeak-api
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 3. API Docs / Health
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 4. 테스트

### 카페 추천
```bash
curl --noproxy "*" "http://127.0.0.1:8000/v1/recommendations?lat=37.5665&lng=126.9780&category=cafe&radius_m=3000&max_results=10"
```

### 음식점 추천
```bash
curl --noproxy "*" "http://127.0.0.1:8000/v1/recommendations?lat=37.5665&lng=126.9780&category=restaurant&radius_m=3000&max_results=10"
```
