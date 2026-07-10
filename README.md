# MixMaster AI (LangChain 기반 RAG 믹싱 에이전트)

본 프로젝트는 오디오 엔지니어링 및 믹싱에 입문하거나 어려움을 겪는 사용자들을 위해, Wikipedia의 공신력 있는 음향 자료들과 실용적인 오디오 계산기 도구를 결합하여 맞춤형 사운드 믹싱 가이드를 제공하는 자율형 LangGraph AI 에이전트입니다.

**[OCI Cloud 배포 버전]** 이제 로컬 개발뿐 아니라 Oracle Cloud Infrastructure(OCI) 위에서도 운영 가능합니다. 데이터 수집-저장-가공-제공의 완전한 파이프라인과 통계 대시보드 API를 포함합니다.

---

## 1. 서비스 소개 및 사용 시나리오

### 💡 서비스 소개

사용자가 직면한 음향 문제(예: "보컬이 너무 먹먹해요", "드럼 드럼 소리가 뒤섞여요")를 자연어로 질문하면, 에이전트가 이를 분석해 필요한 정보를 RAG(검색) 혹은 연산 도구(BPM 딜레이 계산기)를 통해 수집하고 구조화된 전문 조언을 제공합니다.

### 🎬 주요 사용 시나리오

1. **음향 문제 해결 (Troubleshooting)**
    - "보컬 소리가 너무 웅웅거려요. 어떤 주파수를 깎아야 하나요?"
    - 에이전트가 음향 지식 베이스(Wikipedia: Audio Mixing, Equalization)를 조회하여 200Hz-500Hz 대역의 로우-컷(High-pass Filter) 및 벨 커브 EQ 감쇠 방안을 제시합니다.
2. **딜레이 시간 연산 (BPM Delay Calculator)**
    - "120 BPM 하우스 장르의 노래에서 1/8 박자 딜레이와 적정 리버브 프리딜레이 값을 계산해 줘."
    - 에이전트가 템포(BPM)를 인지하고 `calculate_delay_time` 도구를 자율 실행하여 정확한 밀리초(ms) 단위 설정값을 반환합니다.
3. **멀티턴 상세 질문 (Multi-turn Memory)**
    - (이전 질문에 이어) "방금 알려준 설정에서 템포를 80 BPM으로 늦추면 어떻게 설정값이 달라져?"
    - 에이전트가 세션(Thread ID) 메모리를 기반으로 이전 템포 설정을 기억한 채 변동된 연산 결과를 연속적으로 대답합니다.

---

## 2. 전체 아키텍처 및 흐름도 (Workflow 다이어그램)

에이전트는 **LangGraph의 StateGraph**를 기반으로 자율 의사 결정을 하도록 설계되었습니다.

```mermaid
graph TD
    %% 노드 정의
    Start([시작]) --> Node_Analyze[analyze_query]
    Node_Analyze --> Cond_Route{route_after_analysis}

    %% 첫 번째 분기
    Cond_Route -- "정보 부족 (clarification_needed == True)" --> Node_Ask[ask_clarification]
    Cond_Route -- "정상 질문 (clarification_needed == False)" --> Node_Call[call_model]

    %% 대화 종료
    Node_Ask --> End([종료])

    %% 모델 호출 및 도구 실행 루프
    Node_Call --> Cond_Tools{tools_condition}
    Cond_Tools -- "도구 실행 필요 (tool_calls)" --> Node_Tools[tools Node]
    Node_Tools --> Node_Call

    Cond_Tools -- "최종 응답 준비 완료" --> End

    %% 스타일 지정
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#9f9,stroke:#333,stroke-width:2px
    style Cond_Route fill:#ff9,stroke:#333,stroke-width:2px
    style Cond_Tools fill:#ff9,stroke:#333,stroke-width:2px
```

### ⚙️ 흐름 설명

1. **`analyze_query`**: 사용자 쿼리를 받아 카테고리를 분류하고, 역질문이 필요한 모호한 질문인지 Pydantic으로 분석합니다.
2. **`route_after_analysis`**: 분석 결과를 기반으로 조건부 분기(Conditional Edge)를 통해 역질문을 던질지(`ask_clarification`) 모델 메인 루프(`call_model`)로 갈지 결정합니다.
3. **`call_model`**: 필요한 경우 도구를 호출하도록 모델을 호출하고, `tools_condition`을 통해 도구 실행 노드로 빠집니다.
4. **`tools`**: RAG 검색(`search_audio_knowledge`) 또는 딜레이 연산(`calculate_delay_time`) 도구를 자율적으로 선택하여 실행하고, 결과를 모델에 다시 넘깁니다.

---

## 3. 핵심 기술 요소 설명 (Tool / RAG / Memory / Middleware)

### 🛠️ 자율 선택 Tool

- **`search_audio_knowledge(query)`**: Chroma 벡터 DB(음향 Wikipedia 기반)로부터 가장 연관성 높은 3개의 문서 청크를 수집하는 RAG 도구입니다.
- **`calculate_delay_time(bpm)`**: 입력받은 BPM을 바탕으로 `60,000 / BPM` 수식을 활용하여 1/2, 1/4, 1/8, 1/16, 1/32 음표의 길이(ms)와 추천 프리딜레이 길이를 반환하는 산술 연산 도구입니다.

### 📚 RAG 파이프라인

- **데이터 인프라**: Wikipedia API를 이용해 Equalization, Compression, Reverb, Limiter 등 50여 개 이상의 전문 음향 기술 문서 수집.
- **임베딩 및 저장소**: `gemini-embedding-2` 모델을 통해 벡터화 후 `Chroma DB` 로컬 저장소에 영구 보존 및 유사도 매칭 수행.

### 🧠 대화 이력 (Memory)

- `langgraph.checkpoint.memory.MemorySaver`를 도입하여, FastAPI 요청 시 전달되는 `thread_id`별 세션 정보를 추적합니다.
- 사용자와 모델 간 주고받은 `HumanMessage`, `AIMessage`, `ToolMessage`의 누적 히스토리가 제미나이의 역할(Role) 순서 제약 조건에 어긋나지 않도록 엄격히 보장하여 멀티턴 대화의 안정성을 높였습니다.

### 🛡️ Middleware 및 안정성 적용

- **Audit 미들웨어 (FastAPI)**: HTTP 통신의 응답 소요 시간을 밀리초 단위로 측정하여 로그에 기록하며, 예기치 못한 API 서버 예외 발생 시 에러 핸들링 역할을 함께 수행합니다.
- **입력 검증 가드레일**: `BAD_WORDS` 필터를 구축하여 부적절한 단어나 비속어 쿼리가 인입될 시 사전에 차단하고 400 Bad Request 에러를 즉각 반환합니다.
- **Pydantic OutputParser**: 사용자의 최초 쿼리 분석 시 일관된 구조화(JSON) 출력을 강제하여 안정적인 라우팅 제어 흐름을 달성했습니다.

---

## 4. 설치 및 실행 방법

### 로컬 개발 환경 (Docker 없음, 기존 방식)

#### 1) 환경 변수 설정

프로젝트 루트 폴더에 `.env` 파일을 생성합니다.

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 필수 값 입력:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# 로컬 개발 시 (선택사항, 기본값 사용 가능)
# DB_DIR=./chroma_db
```

#### 2) 패키지 설치

```bash
pip install -r requirements.txt
```

#### 3) API 서버 실행

```bash
python3 -m uvicorn src.app:app --reload --port 8000
```

웹 브라우저를 열고 `http://localhost:8000`에 접속하면 시각적인 웹 UI 프론트엔드를 이용할 수 있습니다.

---

### OCI Cloud 배포 (Oracle Cloud Infrastructure)

OCI VM(Oracle Linux 8.10) 위에서 실행하는 경우의 상세 절차입니다.

#### 사전 요구사항 (VM에 이미 구성됨)

- Oracle Linux 8.10 + Python 3.11.13
- MySQL 8.0.46 (DB `mixmaster`, 유저 `mixmaster`)
- Block Volume 50GB (`/mnt/mixmaster-data`)
- firewalld 8000/tcp 개방
- OCI CLI 인증 완료

#### 설치 절차 (처음 1회)

1. **프로젝트 클론** (로컬에서 git으로 관리하지 않는 경우 수동 배포)
   ```bash
   cd /home/opc
   # git clone 또는 수동 배포
   ```

2. **패키지 설치**
   ```bash
   cd /home/opc/MixMaster
   python3.11 -m pip install -r requirements.txt
   ```

3. **환경변수 설정**
   ```bash
   cp .env.example .env
   # .env 파일 편집: MYSQL_PASSWORD 등 실제 값 입력
   vi .env
   ```

4. **로그 디렉토리 생성**
   ```bash
   mkdir -p /mnt/mixmaster-data/{chroma_db,logs}
   chmod 755 /mnt/mixmaster-data/logs
   ```

5. **MySQL 테이블 생성**
   ```bash
   python3.11 scripts/init_db.py
   # 출력: "테이블 생성 완료."
   ```

6. **Wikipedia 데이터 수집 및 Vector DB 구축** (초기 1회, ~수 분 소요)
   ```bash
   cd /home/opc/MixMaster
   python3.11 src/ingester.py
   # 출력: "'Audio mixing' 수집 시작..." 등의 진행 메시지
   # 최종: "성공적으로 DB를 구축하고 '/mnt/mixmaster-data/chroma_db'에 저장했습니다."
   ```

7. **Systemd 서비스 등록** (FastAPI 상시 실행)
   ```bash
   sudo cp deploy/mixmaster.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl start mixmaster
   sudo systemctl enable mixmaster
   sudo systemctl status mixmaster
   ```

8. **Cron 등록** (매일 새벽 1시 자동 수집)
   ```bash
   crontab -e
   # 아래 라인 추가:
   # 0 1 * * * cd /home/opc/MixMaster && /usr/bin/python3.11 src/ingester.py >> /mnt/mixmaster-data/logs/ingester.log 2>&1
   ```

#### 동작 확인

```bash
# FastAPI 서버 정상 작동
curl http://localhost:8000/health
# 출력: {"status":"healthy","model":"gemini-2.5-flash","embedding":"gemini-embedding-2"}

# 채팅 API 테스트
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"보컬이 너무 웅웅거려요","thread_id":"test_session"}'

# 통계 조회
curl http://localhost:8000/api/stats/chat-summary
# 출력: {"total_chats":1,"by_category":{"general":1}}

curl http://localhost:8000/api/stats/documents
# 출력: {"total_documents":60,"by_category":{"theory":17,"manufacturer":27,"software":11}}
```

---

## 5. 데이터 파이프라인 (수집-저장-가공-제공)

### OCI 클라우드 환경에서의 4단계 파이프라인

| 단계               | 담당 모듈                         | 입력               | 출력                 | 저장소                              |
| ------------------ | --------------------------------- | ------------------ | -------------------- | ----------------------------------- |
| **수집 (Collect)** | `src/ingester.py`                 | Wikipedia API      | 원본 텍스트          | Object Storage (`raw/wikipedia/`)   |
| **저장 (Store)**   | `src/oci_storage.py`, `src/db.py` | 수집된 텍스트/메타 | 비정형 + 정형 데이터 | Object Storage, MySQL, Block Volume |
| **가공 (Process)** | `src/ingester.py` (내부)          | 원본 텍스트        | 임베딩 벡터          | Chroma DB (Block Volume)            |
| **제공 (Serve)**   | `src/app.py`                      | 사용자 쿼리        | RAG 답변 + 통계      | 사용자 API, 통계 API                |

**사용 OCI 리소스:**

- **VM Instance**: `vnic-manuel` (4 vCPU/15GB, Oracle Linux 8.10) - 코드 실행 및 FastAPI 서비스
- **Block Volume**: 50GB (`/mnt/mixmaster-data`) - Chroma DB 및 로그 저장
- **Object Storage**: 버킷 `mixmaster-datalake` (Namespace: `cn5brhz58dgr`) - Wikipedia 원본 텍스트 저장
- **MySQL**: 8.0.46 (VM 내 설치) - 문서 메타데이터 및 대화 로그 저장

**추가 통계 API:**

- `/api/stats/chat-summary`: 대화 카테고리별 집계
- `/api/stats/documents`: 수집 문서 카테고리별 통계

---

## 6. 테스트 방법

서버가 실행 중인 상태(`http://localhost:8000`)를 기준으로 아래 3가지 방식으로 검증할 수 있습니다.

### 1) CLI 통합 테스트 (가장 빠른 검증)

백엔드 LangGraph 노드의 자율적 도구 호출 및 멀티턴 메모리를 서버 실행 없이 바로 확인할 수 있습니다.

```bash
python3 src/run_agent.py
```

- BPM 딜레이 질문 시 `calculate_delay_time` 도구가 자율 실행되는지 확인
- 이전 대화(템포 변경) 질문 시 멀티턴 메모리가 유지되는지 확인
- 먹먹한 보컬 EQ 질문 시 Chroma DB RAG 검색 결과가 반영되는지 확인

### 2) API 엔드포인트 curl 테스트

```bash
python3 -m uvicorn app:app --app-dir src --reload --port 8000
```

**딜레이 계산 도구 테스트**

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "135 BPM 노래의 1/16 박자 딜레이 시간을 계산해 주세요.", "thread_id": "curl_session_001"}'
```

**RAG 지식 검색 테스트**

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "보컬 믹싱할 때 컴프레서 래티오(ratio)는 보통 몇 대 몇으로 세팅하나요?", "thread_id": "curl_session_002"}'
```

**가드레일(비속어 필터) 차단 테스트**

```bash
curl -i -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "너는 바보 에이전트니?", "thread_id": "curl_session_003"}'
```

예상 결과: `HTTP/1.1 400 Bad Request`와 함께 `"가드레일 감지: 부적절한 단어('바보')가 포함되어 있습니다."` 메시지 반환.

**통계 API 테스트**

```bash
curl "http://127.0.0.1:8000/api/stats/chat-summary"
curl "http://127.0.0.1:8000/api/stats/documents"
curl "http://127.0.0.1:8000/health"
```

### 3) 웹 프론트엔드 수동 시나리오 테스트

1. 브라우저에서 `http://localhost:8000` 접속
2. 아래 순서로 대화하며 세션 기억 및 도구 활용을 관찰

   - **1단계 (BPM 질문)**: "이번 작업 중인 일렉트로니카 트랙 템포가 128 BPM인데 1/8 박자 딜레이랑 보컬 리버브 프리딜레이 계산해 줘." → 딜레이 연산 값이 정확히 출력되는지 확인
   - **2단계 (멀티턴 질문)**: "템포를 100 BPM으로 떨어뜨리면 1/16 딜레이는 아까보다 얼마나 길어져?" → 이전 128 BPM 컨텍스트를 기억해 비교 답변하는지 확인
   - **3단계 (보컬 RAG 질문)**: "보컬에 치찰음이 너무 심해서 쉭쉭 소리가 나는데 어떻게 고쳐?" → 디에서(De-essing) 관련 RAG 지식이 반영되는지 확인

---

## 7. 한계점 및 향후 개선 방향

### 기존 한계점

- **RAG 컨텍스트의 정교성**: Wikipedia 영문 문서 위주로 색인되어 있어, 한국어 질문에 대한 번역 질의 확장을 추가하거나 한국어로 쓰인 오디오 엔지니어링 강의 문서를 벡터 DB에 추가 통합하면 보다 신뢰성 있는 현업 중심의 지식 응답이 가능할 것입니다.
- **웹 검색 연동**: 사내 로컬 RAG DB 외에도 최신 상용 플러그인(FabFilter, Waves, Soundtoys 등)의 릴리즈 노트와 트렌드를 실시간으로 리트리브할 수 있는 `Tavily` 웹 검색 기능과의 결합이 권장됩니다.

### OCI 클라우드 운영 시 한계점 (향후 개선)

**현재 한계점**

- **MySQL 설치형**: VM 내 직접 설치. 관리형 OCI MySQL Database Service(HeatWave)로 전환 시 자동 백업/고가용성 확보 가능
- **단일 ingester 프로세스**: cron으로 매일 1회 수집. 실시간 수집 불가, 대규모 데이터 처리 시 bottleneck
- **로컬 Chroma**: 단일 서버에 종속. 분산 벡터 DB(Weaviate, Milvus 등)로 대체 시 확장성 증대
- **HTTPS 미지원**: 로컬/데모용으로만 HTTP 사용. 프로덕션 배포 시 OCI Load Balancer + SSL 인증서 필수
- **Object Storage 백업 청크**: 현재 원문만 저장. 청킹된 문서도 백업하면 재구축 시간 단축 가능

**향후 개선 방향 (우선순위)**

1. **OCI MySQL Database Service 마이그레이션** (높음) - SQLAlchemy 연결 문자열만 변경(현재 설계는 이미 호환), 자동 백업·고가용성(Replica) 활용
2. **분산 Vector DB 도입** (중간) - Weaviate, Milvus, Qdrant 등 검토, 대규모 임베딩 벡터(수백만+) 처리 필요 시
3. **Event-driven 실시간 수집** (중간) - OCI Events + Functions로 Wikipedia 업데이트 감지 → 자동 수집, 현재 cron 기반 배치 → 스트림 처리로 전환
4. **대시보드 프론트엔드 강화** (낮음) - `/api/stats/*` 결과를 시각화하는 웹 UI (차트, 테이블) 추가, 현재 기본 프론트엔드는 RAG 챗봇 UI에 집중
5. **HTTPS + 사용자 인증** (낮음) - OCI Load Balancer 앞단 배치, SSL/TLS 적용, JWT 토큰 기반 API 인증 추가
