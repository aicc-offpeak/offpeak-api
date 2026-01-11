# OffPeak API (FastAPI)

서울 실시간 혼잡도(인구) 데이터를 기반으로 **주변 Zone의 혼잡도**를 확인하고,  
선택한 장소(카페/음식점)가 붐비는 경우 **덜 붐비는 대체 장소**를 추천하는 백엔드 API 입니다.

- 혼잡도 데이터: 서울 열린데이터광장 실시간 도시데이터(citydata_ppltn)
- 장소 검색/추천: 카카오 로컬 API(키워드/카테고리 검색)

---

## Requirements

- Windows 10/11
- Miniconda(또는 Anaconda) 설치
- Python 3.11 (conda env로 관리)
- (DB 사용 시) PostgreSQL 15+ 권장

---

## Project Structure

- `app/` : FastAPI app, routes, services, clients
- `app/resources/zones_seed.json` : 서울 주요 장소(Zone) seed
- `scripts/` : seed 생성/테스트 스크립트
- `alembic/` : DB migration

---

## 1) Setup (Conda)

### 0) (1회만) conda init

Git Bash에서 아래 실행 후 **Git Bash 창을 완전히 종료하고 새로 열어야** 적용됩니다.

```bash
conda init bash
```

## 새 창에서도 conda activate가 안 될 경우(bash)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

## 2) conda env 생성/활성화 (Python 3.11)

```bash
conda create -n offpeak-py311 python=3.11 -y
conda activate offpeak-py311
python -V
```

## 3) 패키지 설치

```bash
cd ~/boom.b/offpeak-api
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4) .env 설정 후 로드

```bash
set -a; source .env; set +a
```

Optional env vars (tuning)

- CROWDING_CACHE_TTL_S : 혼잡도 캐시 TTL(초)
- TOP_ZONES : 주변 zone 후보 개수
- PER_ZONE : zone당 추천 POI 개수
- ZONE_SEARCH_RADIUS_M : zone 중심 기준 POI 검색 반경(m)

## 4) Run (Git Bash)

```bash
cd ~/boom.b/offpeak-api
conda activate offpeak-py311
set -a; source .env; set +a

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 5) API Docs / Health

- Swagger: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- Health: http://127.0.0.1:8000/health

## 6) Smoke Test (Git Bash) - 서버가 켜진 상태에서 실행

chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh

## 7) Smoke Test Result

- /health 200
- /zones/nearby 200
- /recommendations 200
- /places/search 200
- /places/insight 200
