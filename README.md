# 🗺️ OffPeak API

실시간 혼잡도 기반 **장소 추천 서비스**의 백엔드 API입니다.

서울 실시간 혼잡도 데이터를 활용하여 **덜 붐비는 장소**와 **방문 최적 시간대**를 추천합니다.

---

## 🎯 주요 기능

- 🔍 **실시간 혼잡도 조회** – 서울시 데이터 기반 주변/특정 구역 현황
- 📍 **장소 검색 및 프로필** – 카카오 로컬 API 연동, 장소별 혼잡도 프로필
- 🏆 **지능형 추천 서비스** – 덜 붐비는 장소 및 방문 추천 시간대 제공
- 📊 **혼잡도 분석** – 시간대별/요일별 혼잡도 패턴 분석

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **Language** | Python 3.11 |
| **Framework** | FastAPI |
| **Database** | PostgreSQL 15+ |
| **ORM** | SQLAlchemy 2.0 + Alembic |
| **HTTP Client** | httpx |
| **Caching** | cachetools (TTLCache) |
| **External API** | 서울 열린데이터광장, Kakao Local API |
| **Env** | Conda, python-dotenv |

---

## 📁 디렉토리 구조

```
app/
├── clients/                    # 외부 API 클라이언트
│   ├── kakao_local.py          # 카카오 로컬 API
│   └── seoul_citydata.py       # 서울시 실시간 도시데이터 API
├── core/
│   └── config.py               # 설정
├── routes/                     # API 엔드포인트
│   ├── health.py               # 헬스체크
│   ├── zones.py                # 구역 혼잡도
│   ├── places.py               # 장소 검색/프로필
│   └── recommendations.py      # 추천
├── services/                   # 비즈니스 로직
│   ├── crowding.py             # 혼잡도 서비스 (TTL 캐싱)
│   ├── place_cache.py          # 장소 캐시 서비스
│   ├── place_profile.py        # 장소 프로필 분석
│   ├── place_crowding_snapshot.py  # 장소 혼잡도 스냅샷
│   ├── crowding_insight.py     # 혼잡도 인사이트
│   └── recommender.py          # 추천 로직
├── resources/
│   └── zones_seed.json         # 서울 주요 구역 시드 데이터
├── db.py                       # DB 연결
├── models.py                   # SQLAlchemy 모델
└── main.py                     # FastAPI 앱 진입점
├── alembic/                    # DB 마이그레이션
├── scripts/                    # 유틸리티 스크립트
└── tests/                      # 테스트
```

---

## 🗄️ 데이터베이스 모델

| 테이블 | 설명 |
|--------|------|
| `zones` | 서울시 주요 POI 구역 마스터 (코드, 이름, 위치) |
| `crowding_snapshots` | 구역별 실시간 혼잡도 이력 |
| `place_cache` | 카카오 API 장소 캐시 (TTL 24h) |
| `place_crowding_snapshots` | 장소별 혼잡도 이력 (주간 프로필용) |

### 혼잡도 레벨
```
여유 (rank: 4) → 보통 (rank: 3) → 약간 붐빔 (rank: 2) → 붐빔 (rank: 1)
  🟢 green        🟡 yellow        🟠 orange          🔴 red
```

---

## 🔌 API 엔드포인트

### Health
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 서버 상태 확인 |

### Zones (구역 혼잡도)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/zones/nearby` | 주변 혼잡 구역 조회 |
| GET | `/zones/{code}/insight` | 구역 시간대별 혼잡도 분석 |
| GET | `/zones/{code}/crowding/history` | 구역 혼잡도 이력 |
| GET | `/zones/{code}/crowding/insight` | 구역 혼잡도 인사이트 |

### Places (장소)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/places/search` | 장소 검색 (키워드/위치/카테고리) |
| POST | `/places/insight` | 장소 상세 + 대안 장소 추천 |
| GET | `/places/profile` | 장소 주간 혼잡도 프로필 (7x24) |
| GET | `/places/recommend_times` | 한산 시간대 추천 |

### Recommendations (추천)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/recommendations` | 덜 붐비는 장소 원샷 추천 |

---

## ⚙️ 환경 설정

### 필수 환경변수 (.env)
```env
# Database
DATABASE_URL=postgresql+psycopg://user:pass@host:port/dbname

# External APIs
KAKAO_REST_API_KEY=<your_kakao_key>
SEOUL_OPENAPI_KEY=<your_seoul_key>

# Server
PORT=8001
CORS_ORIGINS=*
```

### 선택 환경변수 (튜닝)
```env
# 캐시 TTL
CROWDING_CACHE_TTL_S=300          # 혼잡도 캐시 (기본 5분)
PLACE_CACHE_TTL_S=86400           # 장소 캐시 (기본 24시간)

# 추천 파라미터
TOP_ZONES=5                       # 주변 zone 후보 개수
PER_ZONE=7                        # zone당 추천 장소 개수
ZONE_SEARCH_RADIUS_M=700          # zone 중심 검색 반경(m)

# 데이터 수집
CROWDING_SNAPSHOT_MIN_INTERVAL_S=600  # 스냅샷 최소 간격 (10분)
```

---

## 🚀 실행 방법

### 1. 환경 구성 (Conda)
```bash
conda create -n offpeak-py311 python=3.11 -y
conda activate offpeak-py311
```

> **Note**: 새 창에서 `conda activate`가 안 될 경우 (bash)
> ```bash
> source ~/miniconda3/etc/profile.d/conda.sh
> ```

### 2. 패키지 설치
```bash
cd offpeak-api
pip install -r requirements.txt
```

### 3. 환경변수 로드
```bash
set -a; source .env; set +a
```

### 4. DB 마이그레이션
```bash
alembic upgrade head
```

### 5. Zone 시드 데이터 적재
```bash
python scripts/seed_zones_db.py
```

### 6. 서버 실행
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

---

## 📖 API 문서

- **Swagger UI**: http://127.0.0.1:8001/docs
- **OpenAPI JSON**: http://127.0.0.1:8001/openapi.json

---

## 🧪 API 예시

### 주변 혼잡 구역 조회
```bash
curl "http://localhost:8001/zones/nearby?lat=37.5665&lng=126.9780&radius_m=3000"
```

### 장소 검색
```bash
curl "http://localhost:8001/places/search?query=스타벅스&lat=37.5665&lng=126.9780&radius_m=3000"
```

### 덜 붐비는 장소 추천
```bash
curl "http://localhost:8001/recommendations?lat=37.5665&lng=126.9780&category=cafe&radius_m=3000"
```

### 한산 시간대 추천
```bash
curl "http://localhost:8001/places/recommend_times?place_id=12345678&days=7"
```

---

## 🔧 유틸리티 스크립트

| 스크립트 | 용도 |
|----------|------|
| `scripts/run_dev.sh` | 개발 서버 실행 |
| `scripts/smoke_test.sh` | API 스모크 테스트 |
| `scripts/seed_zones_db.py` | Zone 시드 데이터 적재 |
| `scripts/collect_crowding_snapshots.py` | 혼잡도 스냅샷 수집 (주기 실행용) |

### 스모크 테스트
```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

---

## 📊 시스템 구성

### 데이터 흐름
```
[Mobile App]
    ↓ HTTP Request
[FastAPI Backend]
    ↓ Query
[PostgreSQL DB] ← TTL 캐싱 (place_cache)
    ↓
[외부 API]
├── 서울 열린데이터광장 (실시간 혼잡도)
└── Kakao Local API (장소 검색)
```

### 캐싱 전략
| 대상 | TTL | 목적 |
|------|-----|------|
| 혼잡도 (메모리) | 5분 | 실시간 API 호출 최소화 |
| 장소 캐시 (DB) | 24시간 | Kakao API 호출 최소화 |

---

## 📋 체크리스트

- [ ] `.env` 환경변수 설정 완료
- [ ] PostgreSQL DB 연결 확인
- [ ] Kakao API Key 유효성 확인
- [ ] Seoul OpenAPI Key 유효성 확인
- [ ] Zone 시드 데이터 적재 완료

---

## 📄 라이선스

본 프로젝트의 저작권은 **OffPeak Team**에 있으며,
상용 및 배포 정책은 별도 라이선스 조항을 따릅니다.

---

## 👥 팀 정보

**OffPeak Team**
실시간 혼잡도 기반 장소 추천 서비스
