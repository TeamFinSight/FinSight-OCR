# FinSight-OCR 개발 가이드

## 목차
- [🏗️ 시스템 아키텍처](#️-시스템-아키텍처)
- [⚙️ 환경 설정 및 설치](#️-환경-설정-및-설치)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🤖 AI 모델 아키텍처](#-ai-모델-아키텍처)
- [💡 박스 라벨링 알고리즘](#-박스-라벨링-알고리즘)
- [🔧 개발 워크플로우](#-개발-워크플로우)
- [📡 API 상세 문서](#-api-상세-문서)
- [🧪 테스트 및 디버깅](#-테스트-및-디버깅)
- [🚀 배포 및 성능 최적화](#-배포-및-성능-최적화)

## 🏗️ 시스템 아키텍처

### 전체 시스템 구조
```
┌─────────────────────┐     ┌─────────────────────┐
│   Frontend (React)  │────▶│  Backend (FastAPI)  │
│                     │     │                     │
│ - File Upload UI    │     │ - OCR API Endpoints │
│ - Document Selector │     │ - Model Pipeline    │
│ - Results Display   │     │ - Box Labeling      │
│ - Data Export       │     │ - JSON Response     │
└─────────────────────┘     └─────────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │    AI Models        │
                            │                     │
                            │ - DBNet (Detection) │
                            │ - CRNN (Recognition)│
                            │ - Pre-trained .pth  │
                            └─────────────────────┘
```

### 2-Stage OCR 파이프라인
```
[ 문서 이미지 입력 ]
        │
        ▼
┌────────────────────────────────────┐
│  Stage 1: 텍스트 검출 (Detection)  │
│  - 모델: DBNet(ResNet-18)          │
│  - 출력: 문서 내 텍스트 영역의     │
│          좌표(Polygon) 정보        │
└────────────────────────────────────┘
        │
        │ (검출된 텍스트 영역 좌표)
        ▼
┌────────────────────────────────────┐
│      이미지 자르기 (Crop)        │
└────────────────────────────────────┘
        │
        │ (자른 개별 텍스트 이미지들)
        ▼
┌────────────────────────────────────┐
│  Stage 2: 텍스트 인식 (Recognition)  │
│  - 모델: CRNN(EfficientNet-B3)       │
│  - 출력: 각 이미지별 텍스트 내용  │
└────────────────────────────────────┘
        │
        ▼
[ 최종 결과 (JSON 형태 텍스트와 좌표 매핑) ]
```

## ⚙️ 환경 설정 및 설치

### Prerequisites
- **Node.js**: 18+ (프론트엔드 개발)
- **Python**: 3.9+ (백엔드 및 AI 모델)
- **CUDA**: 11.8+ (GPU 사용 시, 권장)
- **Git**: 최신 버전

### 1단계: 저장소 클론 및 모델 파일 준비
```bash
# 저장소 클론
git clone https://github.com/TeamFinSight/FinSight-OCR.git
cd FinSight-OCR
```

> ⚠️ **중요**: OCR 처리를 위해서는 `backend/modelrun/saved_models/` 디렉터리에 다음 `.pth` 파일들이 필수적으로 있어야 합니다:
> - `dbnet_a100_best.pth` (텍스트 검출 모델, ~146MB)
> - `robust_korean_recognition_best.pth` (텍스트 인식 모델, ~163MB)

### 2단계: 환경 설정

#### 방법 1: Conda 환경 파일 사용 (권장)
```bash
# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate finsight-ocr
```

#### 방법 2: 수동 설정
```bash
# 1. Conda 환경 생성
conda create -n finsight-ocr python=3.9
conda activate finsight-ocr

# 2. PyTorch 설치 (CUDA 지원)
pip install torch==2.1.2+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 3. 백엔드 의존성 설치
cd backend
pip install -r requirements.txt
```

#### 3단계: 프론트엔드 설정
```bash
# 프론트엔드 디렉터리로 이동
cd frontend

# 의존성 설치
npm install
```

### 환경 관리 명령어
```bash
# 환경 활성화/비활성화
conda activate finsight-ocr
conda deactivate

# 환경 삭제 (필요시)
conda env remove -n finsight-ocr

# 설치된 패키지 확인
conda list

# 환경 정보 내보내기
conda env export > environment.yml --no-builds
```

## 📁 프로젝트 구조

```
FinSight-OCR/
├── frontend/                          # React TypeScript 프론트엔드
│   ├── src/
│   │   ├── App.tsx                   # 메인 애플리케이션 컴포넌트
│   │   ├── components/               # UI 컴포넌트
│   │   │   ├── ui/                   # shadcn/ui 기본 컴포넌트
│   │   │   ├── DocumentTypeSelector.tsx  # 문서 타입 선택
│   │   │   ├── ImageUpload.tsx           # 드래그앤드롭 업로드
│   │   │   ├── OCRProcessor.tsx          # OCR 처리 및 결과 표시
│   │   │   ├── GenericTable.tsx          # 편집 가능한 테이블
│   │   │   ├── ResultExporter.tsx        # Excel/CSV/JSON 내보내기
│   │   │   └── DropdownInfo.tsx          # 문서 타입 정보 표시
│   │   ├── services/                 # API 통신 서비스
│   │   │   ├── api.ts               # 공통 API 클라이언트
│   │   │   └── ocrService.ts        # OCR 전용 서비스
│   │   ├── hooks/                   # React 커스텀 훅
│   │   │   ├── useOCR.ts           # OCR 처리 훅
│   │   │   └── useErrorHandler.ts   # 에러 핸들링 훅
│   │   ├── types/                   # TypeScript 타입 정의
│   │   ├── constants/               # 상수 및 더미 데이터
│   │   └── config/                  # 애플리케이션 설정
│   ├── package.json                 # Node.js 의존성
│   ├── vite.config.ts              # Vite 빌드 설정
│   └── dist/                       # 빌드 결과물
├── backend/                            # FastAPI Python 백엔드
│   ├── main.py                     # FastAPI 메인 애플리케이션
│   ├── boxLabel.py                 # 박스 라벨링 알고리즘 모듈
│   ├── document/                   # 문서 타입 및 라벨링 설정
│   │   ├── document_types.json     # 지원 문서 타입 정의
│   │   ├── labelings.json          # 문서별 좌표 정보 (13종 문서)
│   │   └── boxlabeling.txt         # 박스 라벨링 알고리즘 문서
│   ├── modelrun/                   # AI 모델 실행 환경
│   │   ├── scripts/                # 모델 학습 및 실행 스크립트
│   │   │   ├── detection/          # DBNet 텍스트 검출 모델
│   │   │   │   ├── dbnet_a100_optimized.py      # A100 최적화 학습
│   │   │   │   ├── fix_doctype_preprocessing.py  # 전처리 수정
│   │   │   │   └── train_final_detector.py       # 최종 검출기 학습
│   │   │   ├── recognition/        # CRNN 텍스트 인식 모델
│   │   │   │   ├── preprocess.py                 # 데이터 전처리
│   │   │   │   ├── final.py                      # 최종 인식 모델
│   │   │   │   ├── recognition_a100_optimized.py # A100 최적화 인식
│   │   │   │   └── train_korean_handwritten_recognition_enhanced.py
│   │   │   └── tba/                # 통합 OCR 파이프라인
│   │   │       └── run_ocr.py      # OCR 파이프라인 실행
│   │   ├── configs/                # 모델 설정 파일
│   │   │   └── recognition/
│   │   │       ├── sweep_config.yaml      # WandB 하이퍼파라미터 설정
│   │   │       └── korean_char_map.txt    # 한국어 문자 맵
│   │   ├── saved_models/           # 학습된 모델 저장소 (⚠️ .pth 파일 필수)
│   │   │   ├── dbnet_a100_best.pth              # DBNet 텍스트 검출 모델
│   │   │   ├── robust_korean_recognition_best.pth # CRNN 텍스트 인식 모델
│   │   │   └── README.md                        # 모델 파일 설명
│   │   └── output/                 # 모델 실행 결과 및 시각화
│   ├── output/                     # API 처리 결과 저장소
│   ├── requirements.txt            # Python 의존성
│   └── Dockerfile                  # Docker 빌드 설정
├── docker-compose.yml              # Docker 컨테이너 구성
├── docker-compose.dev.yml          # 개발용 Docker 구성
├── environment.yml                 # Conda 환경 설정
├── requirements.txt                # 전체 프로젝트 Python 의존성
├── DOCKER.md                      # Docker 설정 가이드
├── README.md                      # 프로젝트 개요
└── README_DEV.md                  # 개발 가이드 (이 파일)
```

## 🤖 AI 모델 아키텍처

### 1. 텍스트 검출 모델: DBNet
- **Backbone**: ResNet-18 (경량화 및 속도 최적화)
- **FPN (Feature Pyramid Network)**: Multi-scale 특징 추출로 다양한 크기 텍스트 검출
- **DB Head**: Differentiable Binarization으로 정밀한 경계 검출
- **입력**: RGB 이미지 (임의 크기)
- **출력**: 텍스트 영역 polygon 좌표

### 2. 텍스트 인식 모델: CRNN
- **CNN Backbone**: EfficientNet-B3 (정확도와 효율성의 균형)
- **RNN Encoder**: Bidirectional LSTM (시퀀스 데이터의 문맥 정보 파악)
- **Decoder**: CTC (Connectionist Temporal Classification - 길이 가변 텍스트 시퀀스 인식)
- **입력**: 검출된 텍스트 영역 이미지 (64px 높이로 정규화)
- **출력**: 한국어 텍스트 문자열

### 3. 주요 개선사항
- **GPU 기반 데이터 증강**: Kornia 라이브러리 활용
- **혼합 정밀도 학습**: AMP (Automatic Mixed Precision)
- **고급 데이터 증강**: RandomErasing, RandomPerspective, ColorJitter 등
- **WandB 연동**: 실험 관리 및 하이퍼파라미터 튜닝

### 4. 훈련 데이터
- **출처**: AI Hub 한글 금융 문서 OCR 데이터셋
- **문서 유형**: 은행 신고서, 보험 서류, 증권 서류 등 13종
- **언어**: 한국어 + 숫자 + 특수문자
- **데이터 크기**: 검출용 약 10만장, 인식용 약 50만 텍스트 이미지

## 💡 박스 라벨링 알고리즘

### 알고리즘 개요
FinSight-OCR의 핵심 기술 중 하나인 **지능형 박스 라벨링 알고리즘**은 OCR로 검출된 텍스트 영역에 의미있는 라벨을 자동으로 할당합니다.

### 작동 원리
1. **기준 데이터 (A)**: `labelings.json`에 저장된 라벨링된 좌표와 라벨 정보
2. **대상 데이터 (B)**: OCR로 검출된 텍스트 영역의 좌표값
3. **매칭 과정**: B의 중심점 좌표와 가장 가까운 A를 찾아 라벨 매칭

### 핵심 기능
- **중심점 거리 계산**: 유클리드 거리 기반 최단거리 매칭
- **이미지 크기 정규화**: 서로 다른 이미지 크기에 대응하는 비율 계산
- **Y축 우선 탐색**: 문서의 세로 배치를 고려한 매칭 정확도 향상

### 매칭 결과 예시
```
기준점 [2097, 442, 'bs_register_num'] → 검출점 [2098, 444, 'bs_register_num']
기준점 [699, 533, 'bs_name']          → 검출점 [700, 532, 'bs_name']
기준점 [1096, 529, 'bs_name']         → 검출점 [1097, 532, 'bs_name']
```

### 라벨 약어 체계
- **bs**: 사업자 (business)
- **co**: 영업 (company)  
- **ceo**: 대표자
- **pay**: 결제
- **deposit**: 예금
- **cust**: 고객 (customer)
- **account**: 계좌

## 🔧 개발 워크플로우

### 프론트엔드 개발

#### 개발 서버 실행
```bash
cd frontend
npm run dev      # http://localhost:5173 (개발 서버)
npm run build    # 프로덕션 빌드
npm run preview  # 빌드 미리보기
```

#### 환경 변수 설정 (.env)
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MOCK_API=false
VITE_DEBUG_MODE=true
```

#### 새로운 문서 유형 추가
1. `DocumentTypeSelector.tsx`의 `documentCategories`에 옵션 추가
2. `backend/document/labelings.json`에 좌표 정보 추가
3. 백엔드에서 새 타입 처리 로직 구현

### 백엔드 개발

#### 개발 서버 실행
```bash
cd backend
conda activate finsight-ocr
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 새로운 문서 타입 모델 추가
`backend/document/labelings.json`에 다음 형태로 추가:
```json
{
  "type_name": "new_document_type",
  "bin_box": [
    [x_coordinate, y_coordinate, "field_name"],
    [350, 150, "customer_name"],
    [500, 200, "account_number"]
  ]
}
```

### AI 모델 학습 워크플로우

#### 1단계: 데이터 전처리
```bash
python backend/modelrun/scripts/recognition/preprocess.py \
  --raw_data_dir ./data \
  --output_dir ./processed_data/recognition \
  --val_split_ratio 0.1
```

#### 2단계: 모델 학습
```bash
# WandB Sweeps를 통한 하이퍼파라미터 튜닝
wandb sweep backend/modelrun/configs/recognition/sweep_config.yaml
wandb agent <SWEEP_ID>

# 직접 학습
python backend/modelrun/scripts/recognition/train_korean_handwritten_recognition_enhanced.py \
  --train_data_dir "processed_data/recognition/train_images" \
  --val_data_dir "processed_data/recognition/val_images" \
  --save_dir "saved_models/recognition" \
  --epochs 15 --batch_size 64 --lr 0.002
```

#### 3단계: 모델 평가
```bash
python backend/modelrun/scripts/recognition/final.py \
  --weights "saved_models/robust_korean_recognition_best.pth" \
  --source "test_image.png" \
  --img_h 64
```

#### 4단계: 통합 OCR 실행
```bash
python backend/modelrun/scripts/tba/run_ocr.py \
  --det_weights "saved_models/dbnet_a100_best.pth" \
  --rec_weights "saved_models/robust_korean_recognition_best.pth" \
  --source "sample_image.png" \
  --char_map "configs/recognition/korean_char_map.txt"
```

## 📡 API 상세 문서

### POST /insert
OCR 처리를 위한 메인 엔드포인트

**요청 형식:**
```bash
curl -X POST "http://localhost:8000/insert" \
  -F "filename=@document.png" \
  -F "doc_type=auto_transfer"
```

**응답 형식:**
```json
{
  "metadata": {
    "source_image": "document.png",
    "processed_at": "2025-09-19T10:00:23.755Z",
    "total_detections": 12,
    "model_info": {
      "detection_model": "DBNet (ResNet18)",
      "recognition_model": "RobustKoreanCRNN (EfficientNet-B3 + BiLSTM + CTC)"
    }
  },
  "document_info": {
    "width": 2481,
    "height": 3507,
    "document_type": "auto_transfer"
  },
  "fields": [
    {
      "id": 1,
      "labels": "고객명",
      "rotation": 0.0,
      "value_text": "홍길동",
      "confidence": 0.95,
      "value_box": {
        "x": [100, 200, 200, 100],
        "y": [50, 50, 80, 80],
        "type": "polygon"
      }
    }
  ]
}
```

### GET /api/v1/health
시스템 상태 확인 엔드포인트

**응답:**
```json
{
  "status": "ok",
  "timestamp": "2025-09-19T10:00:23.755Z"
}
```

### 지원 문서 유형 (13종)

#### 은행 업무 (4종)
- `general_form`: 일반 양식 (제신고서)
- `identity_delegation`: 신분위임장 (실명확인 위임장)
- `auto_transfer`: 자동이체 신청서
- `account_opening`: 계좌 개설 신청 (명의변경 신청서)

#### 보험 업무 (4종)
- `insurance_contract`: 보험계약대출 승계 동의서
- `insurance_claim`: 보험 청구 관련 서류
- `auto_transfer_approval`: 자동이체 승인서
- `delegation_insurance`: 위임장 (보험용)

#### 기타 업무 (5종)
- `virtual_account`: 기업인터넷뱅킹 가상계좌발급 목적확인서
- `delegation_general`: 위임장 (일반용)
- `auto_transfer_insurance`: 자동이체신청서 (보험용)
- `nursing_service`: 간병인 지원 서비스 신청서
- `theft_damage_report`: 도난/파손 사실 확인서

## 🧪 테스트 및 디버깅

### 프론트엔드 테스트
```bash
cd frontend
npm run test         # 유닛 테스트 (Jest + Testing Library)
npm run test:e2e     # E2E 테스트 (Cypress)
npm run test:coverage # 커버리지 리포트
```

### 백엔드 테스트
```bash
cd backend
pytest                      # 유닛 테스트
pytest --cov=. --cov-report=html  # 커버리지 포함
python -m pytest tests/ -v  # 상세 출력
```

### API 테스트
```bash
# FastAPI 자동 문서 UI
http://localhost:8000/docs

# 직접 API 테스트
curl -X POST "http://localhost:8000/insert" \
  -F "filename=@test_document.png" \
  -F "doc_type=auto_transfer"
```

### 디버깅 가이드

#### 로그 확인
```bash
# 백엔드 로그
tail -f backend/logs/app.log

# Docker 로그
docker-compose logs -f backend
docker-compose logs -f frontend
```

#### 일반적인 문제 해결

1. **모델 로딩 실패**
   - 모델 파일 존재 확인: `backend/modelrun/saved_models/`
   - CUDA 메모리 부족: 배치 사이즈 조정 또는 CPU 모드로 전환
   - 모델 파일 크기 확인: dbnet_a100_best.pth (~146MB), robust_korean_recognition_best.pth (~163MB)

2. **API 연결 실패**
   - CORS 설정 확인: `main.py`의 origins 리스트
   - 포트 충돌 확인: 8000번 포트 사용 여부
   - 방화벽 설정 확인

3. **OCR 정확도 저하**
   - 이미지 품질 확인: 해상도, 노이즈, 기울기
   - 문서 타입 정확성: 올바른 doc_type 선택
   - 조명 조건 확인: 그림자, 반사광 최소화

## 🚀 배포 및 성능 최적화

### Docker 배포
```bash
# 전체 스택 빌드 및 실행
docker-compose up --build

# 개별 서비스 빌드
docker build -t finsight-frontend ./frontend
docker build -t finsight-backend ./backend
```

### 수동 배포
```bash
# 프론트엔드 빌드 및 배포
cd frontend
npm run build
# dist/ 폴더를 웹서버에 배포

# 백엔드 배포
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 성능 최적화 타겟

#### 프론트엔드
- **번들 크기**: 최대 2MB (gzip 압축 후)
- **로딩 시간**: 3G 환경 3초 이내, WiFi 환경 1초 이내
- **접근성**: WCAG 2.1 AA 수준 준수

#### 백엔드
- **API 응답시간**: 평균 200ms 이내
- **OCR 정확도**: 87-98% (문서 유형별)
- **박스 라벨링 정확도**: 95% 이상
- **동시 처리**: 20개 이상 요청 병렬 처리

### WandB 성능 모니터링
```bash
# WandB 로그인 및 프로젝트 설정
wandb login
wandb init --project "FinSight-OCR"

# 학습 메트릭 추적
python train_model.py --wandb_project "FinSight-OCR"
```

### 모니터링 대시보드
- **학습 메트릭**: 손실, 정확도, 학습률, 검증 성능
- **시스템 메트릭**: GPU 사용률, 메모리 사용량, 디스크 I/O
- **비즈니스 메트릭**: 처리 시간, 성공률, 사용자 만족도

---

## 🛡️ 보안 고려사항

- **데이터 보호**: 모든 OCR 처리를 On-Premise 환경에서 수행
- **파일 검증**: 업로드 파일 타입 및 크기 제한 (최대 10MB)
- **CORS 설정**: 허용된 도메인에서만 API 접근 가능
- **입력 검증**: SQL 인젝션 및 XSS 공격 방지
- **로깅**: 민감한 정보 로그 제외, 접근 기록 추적

더 자세한 정보는 각 모듈별 코드 주석을 참고하세요.