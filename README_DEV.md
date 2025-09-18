# FinSight-OCR 개발 가이드

## 목차
- [🏗️ 시스템 아키텍처](#️-시스템-아키텍처)
- [📦 상세 설치 가이드](#-상세-설치-가이드)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🤖 AI 모델 상세](#-ai-모델-상세)
- [🔧 모델 학습 가이드](#-모델-학습-가이드)
- [💡 박스 라벨링 알고리즘](#-박스-라벨링-알고리즘)
- [📡 API 문서](#-api-문서)
- [🚀 배포 가이드](#-배포-가이드)
- [⚠️ 보안 고려사항](#️-보안-고려사항)

## 🏗️ 시스템 아키텍처

### 전체 시스템 구조
```
Frontend (React + Vite)    Backend (FastAPI + PyTorch)

 - File Upload UI         - OCR API Endpoints
 - Document Type Select   - Model Pipeline
 - Results Display        - Box Labeling
 - Data Export            - JSON Response

                     ↕

                               AI Models
                               - DBNet (Detection)
                               - CRNN (Recognition)
                               - Pre-trained

```

### 2-Stage OCR 파이프라인
```
[ 문서 이미지 입력 ]
        |
        v
+------------------------------------+
|  Stage 1: 텍스트 검출 (Detection)  |
|  - 모델: DBNet(ResNet-18)          |
|  - 출력: 문서 내 텍스트 영역의     |
|          좌표(Polygon) 정보        |
+------------------------------------+
        |
        | (검출된 텍스트 영역 좌표)
        v
+------------------------------------+
|      이미지 자르기 (Crop)        |
+------------------------------------+
        |
        | (자른 개별 텍스트 이미지들)
        v
+------------------------------------+
|  Stage 2: 텍스트 인식 (Recognition)  |
|  - 모델: CRNN(EfficientNet-b4)       |
|  - 출력: 각 이미지별 텍스트 내용  |
+------------------------------------+
        |
        v
[ 최종 결과 (JSON 형태 텍스트와 좌표 매핑) ]
```

## 📦 상세 설치 가이드

### Prerequisites
- Node.js 18+
- Python 3.9+
- CUDA 11.8+ (GPU 사용 시)

### 환경 설정

#### 방법 1: environment.yml 사용 (권장)
```bash
# Conda 환경 파일로 한 번에 설정
conda env create -f environment.yml
conda activate finsight-ocr
```

#### 방법 2: 수동 설정
```bash
# Conda 환경 생성
conda create -n finsight-ocr python=3.9
conda activate finsight-ocr

# PyTorch 설치 (CUDA 지원)
pip install torch==2.1.2+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 백엔드 의존성 설치
pip install -r requirements.txt
```

### 환경 파일 상세 정보

#### environment.yml 구성 요소
- **환경명**: `finsight-ocr`
- **Python 버전**: 3.9
- **채널**: conda-forge, defaults
- **주요 패키지**:
  - **딥러닝**: PyTorch 2.1.2+cu121, torchvision, torchaudio
  - **컴퓨터 비전**: OpenCV, albumentations, scikit-image
  - **웹 프레임워크**: FastAPI, uvicorn
  - **OCR**: EasyOCR 1.7.0
  - **데이터 처리**: pandas, numpy, scipy
  - **실험 관리**: WandB 0.21.3

#### 환경 관리 명령어
```bash
# 환경 활성화
conda activate finsight-ocr

# 환경 비활성화
conda deactivate

# 환경 삭제 (필요시)
conda env remove -n finsight-ocr

# 설치된 패키지 확인
conda list

# 환경 정보 내보내기 (업데이트된 패키지 정보 저장)
conda env export > environment.yml --no-builds
```

## 📁 프로젝트 구조

```
FinSight-OCR/
├─ frontend/                 # React TypeScript 프론트엔드
│  ├─ src/
│  │  ├─ components/      # UI 컴포넌트
│  │  │  ├─ ui/         # shadcn/ui 컴포넌트
│  │  │  ├─ DocumentTypeSelector.tsx
│  │  │  ├─ ImageUpload.tsx
│  │  │  ├─ OCRProcessor.tsx
│  │  │  ├─ GenericTable.tsx
│  │  │  └─ ResultExporter.tsx
│  │  ├─ services/        # API 서비스
│  │  │  ├─ api.ts      # 공통 API 클라이언트
│  │  │  └─ ocrService.ts # OCR 전용 서비스
│  │  ├─ hooks/          # 커스텀 훅
│  │  │  ├─ useOCR.ts
│  │  │  └─ useErrorHandler.ts
│  │  ├─ types/          # TypeScript 타입 정의
│  │  ├─ constants/      # 상수 정의
│  │  └─ config/         # 설정 파일
│  ├─ package.json
│  └─ vite.config.ts
├─ backend/                 # FastAPI Python 백엔드
│  ├─ main.py             # FastAPI 메인 애플리케이션
│  ├─ boxLabel.py         # 박스 라벨링 알고리즘 모듈
│  ├─ document/           # 문서 모델 설정
│  │  ├─ labelings.json  # 문서 타입별 좌표 정보 (13종 문서)
│  │  └─ boxlabeling.txt # 박스 라벨링 알고리즘 문서
│  └─ modelrun/           # AI 모델 실행 환경
│     ├─ scripts/        # 훈련/추론 스크립트
│     │  ├─ detection/  # 텍스트 검출 모델 (DBNet)
│     │  │  ├─ preprocess.py
│     │  │  └─ train_DBNet.py
│     │  ├─ recognition/ # 텍스트 인식 모델 (EfficientNet-B3 + CRNN)
│     │  │  ├─ preprocess.py
│     │  │  ├─ train_efficientnet_e3_ctc.py
│     │  │  ├─ eval.py
│     │  │  └─ recognition_pipeline.py
│     │  └─ tba/        # 통합 OCR 파이프라인
│     │     └─ run_ocr.py
│     ├─ configs/        # 모델 설정
│     │  ├─ detection/
│     │  └─ recognition/
│     │     ├─ sweep_config.yaml   # WandB 하이퍼파라미터 스윕 설정
│     │     └─ korean_char_map.txt # 한국어 문자 맵 파일
│     ├─ saved_models/   # 훈련된 모델 가중치
│     │  ├─ detection/
│     │  └─ recognition/
│     └─ output/         # 추론 결과, 시각화 등 출력
├─ output/             # 처리 결과 저장소
├─ environment.yml          # Conda 환경 설정
├─ requirements.txt         # Python 의존성
├─ README.md               # 프로젝트 개요
└─ README_DEV.md           # 개발 가이드 (이 파일)
```

## 🤖 AI 모델 상세

### 1. 텍스트 검출 모델: DBNet
- **Backbone**: ResNet-18 (경량화 및 속도 최적화)
- **FPN**: Multi-scale 특징 추출 (다양한 크기 텍스트 검출)
- **DB Head**: Differentiable Binarization (정밀한 경계 검출)

### 2. 텍스트 인식 모델: CRNN
- **CNN Backbone**: EfficientNet-B3 (정확도와 효율성의 균형)
- **RNN Encoder**: Bidirectional LSTM (시퀀스 데이터의 문맥 정보 파악)
- **Decoder**: CTC (Connectionist Temporal Classification - 길이 가변 텍스트 시퀀스 인식)
- **주요 개선사항**:
  - GPU 기반 데이터 증강 (Kornia 활용)
  - 혼합 정밀도 학습 (AMP)
  - 고급 데이터 증강 (RandomErasing, RandomPerspective 등)

### 3. 훈련 데이터
- **출처**: AI Hub OCR 데이터 (금융 및 물류)
- **문서 유형**: 은행 신고서, 보험 서류, 증권 서류 등
- **언어**: 한국어 + 숫자 + 특수문자
- **모니터링**: WandB를 통한 실험 관리 및 추적

## 🔧 모델 학습 가이드

### 1단계: 데이터 전처리

#### 텍스트 인식 모델용 데이터 전처리
```bash
python backend/modelrun/scripts/recognition/preprocess.py \
  --raw_data_dir ./data \
  --output_dir ./processed_data/recognition \
  --val_split_ratio 0.1
```

**작업 내용**: AI Hub 데이터셋의 JSON 형식 주석을 처리하여 cropped image와 `labels.csv` 파일을 생성합니다.

**출력 결과**:
- `processed_data/recognition/train_images/`: 학습용 이미지 및 라벨
- `processed_data/recognition/val_images/`: 검증용 이미지 및 라벨

### 2단계: 모델 학습 (WandB 연동)

#### 방법 1: WandB Sweeps (하이퍼파라미터 자동 튜닝)
```bash
# 1. 스윕 ID 생성
wandb sweep backend/modelrun/configs/recognition/sweep_config.yaml

# 2. 스윕 에이전트 실행 (출력된 SWEEP_ID 사용)
wandb agent <SWEEP_ID>
```

#### 방법 2: 직접 학습
```bash
python backend/modelrun/scripts/recognition/train_efficientnet_e3_ctc.py \
  --train_data_dir "processed_data/recognition/train_images" \
  --val_data_dir "processed_data/recognition/val_images" \
  --save_dir "saved_models/recognition/efficientnet_e3_ctc" \
  --epochs 15 --batch_size 64 --lr 0.002 \
  --wandb_project "FinSight-OCR-Recognition"
```

#### 텍스트 검출 모델 학습 (DBNet)
```bash
python backend/modelrun/scripts/detection/train_DBNet.py \
  --data_dir "processed_data/detection" \
  --save_dir "saved_models/detection" \
  --backbone "resnet18" \
  --epochs 200 --batch_size 16 \
  --wandb_project "FinSight-OCR-Detection"
```

### 3단계: 모델 평가

#### 단일 이미지 테스트
```bash
python backend/modelrun/scripts/recognition/eval.py \
  --weights "saved_models/recognition/efficientnet_e3_ctc/best_model.pth" \
  --source "test_image.png" \
  --img_h 64 --lstm_layers 2 --lstm_hidden_size 256
```

#### 전체 데이터셋 평가
```bash
python backend/modelrun/scripts/recognition/eval.py \
  --weights "saved_models/recognition/efficientnet_e3_ctc/best_model.pth" \
  --source "processed_data/recognition/val_images" \
  --label_csv "processed_data/recognition/val_images/labels.csv" \
  --img_h 64 --lstm_layers 2 --lstm_hidden_size 256
```

**평가 지표**:
- **정확도 (Sequence Accuracy)**: 예측된 문장과 정답 문장의 완전 일치율
- **문자 에러율 (CER)**: 문자 단위 에러 비율 (낮을수록 좋음)

### 4. 모델 실행
```bash
# 통합 OCR 파이프라인 실행
python backend/modelrun/scripts/tba/run_ocr.py \
  --det_weights "saved_models/detection/dbnet_resnet18_best_RUN-ID.pth" \
  --rec_weights "saved_models/recognition/efficientnet_e3_ctc/best_model.pth" \
  --source "sample_image.png" \
  --char_map "configs/recognition/korean_char_map.txt" \
  --img_h 64 --lstm_layers 2 --lstm_hidden_size 256
```

## 💡 박스 라벨링 알고리즘

본 시스템의 핵심 기술 중 하나인 **지능형 박스 라벨링 알고리즘**은 OCR로 검출된 텍스트 영역에 의미있는 라벨을 자동으로 할당합니다.

### 알고리즘 원리

1. **기준 데이터 (A)**: 라벨링된 비교군 - `labelings.json`의 좌표와 라벨 정보
2. **대상 데이터 (B)**: OCR로 검출된 텍스트 영역의 좌표값만 있는 데이터
3. **매칭 과정**: B의 중심점 좌표와 가장 가까운 A를 찾아 라벨을 매칭

### 주요 특징

- **중심점 거리 계산**: 각 텍스트 박스의 중심 좌표를 계산하여 최단거리 매칭
- **이미지 크기 대응**: 이미지 사이즈가 다를 경우 비율 계산으로 좌표 정규화
- **Y축 우선 탐색**: X좌표를 임의값 N(4)만큼 접어 Y좌표 위주로 검색하여 정확도 향상

### 매칭 결과 예시

```
A 중심 [2097, 442, 'bs_register_num']  →  B 중심 [2098, 444, 'bs_register_num']
A 중심 [699, 533, 'bs_name']           →  B 중심 [700, 532, 'bs_name']
A 중심 [1096, 529, 'bs_name']          →  B 중심 [1097, 532, 'bs_name']
```

### 라벨 약어 체계

- **bs**: 사업자 (business)
- **co**: 영업 (company)
- **ceo**: 대표자
- **pay**: 결제
- **deposit**: 예금

## 📡 API 문서

### POST /insert
OCR 처리를 위한 이미지 업로드 엔드포인트

**Request:**
```bash
curl -X POST "http://localhost:8000/insert" \
  -F "filename=@document.png" \
  -F "doc_type=auto_transfer"
```

**Response:**
```json
{
  "metadata": {
    "source_image": "document.png",
    "processed_at": "2025-09-17T10:00:23.755Z",
    "total_detections": 12,
    "model_info": {
      "detection_model": "DBNet (ResNet18)",
      "recognition_model": "RobustKoreanCRNN (EfficientNet-B2 + BiLSTM + CTC)"
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

### GET /health
헬스체크 엔드포인트

**Response:**
```json
{
  "status": "healthy",
  "service": "FinSight-OCR Backend"
}
```

## 📝 지원 문서 유형 상세

시스템은 다음과 같은 13종의 금융 문서 유형을 지원하며, 각 문서별 특화된 필드 추출이 가능합니다:

### 은행 업무 (4종)
- **`general_form`**: 일반 양식 (제신고서)
  - 주요 필드: 계좌번호, 고객명, 변경 전/후 구분, 신청일자, 고객주소, 전화번호 등
- **`identity_delegation`**: 신분위임장 (실명확인 위임장)
  - 주요 필드: 은행명, 예금종류, 계좌번호, 대리인 정보, 위임인 정보 등
- **`auto_transfer`**: 자동이체 신청서
  - 주요 필드: 사업자명, 사업자등록번호, 주소, 대표자 정보, 은행정보 등
- **`account_opening`**: 계좌 개설 신청 (명의변경 신청서)
  - 주요 필드: 은행명, 변경사유, 예금종류, 계좌번호, 신청인 정보 등

### 보험 업무 (4종)
- **`insurance_contract`**: 보험계약대출 승계 동의서
  - 주요 필드: 보험종류, 증권번호, 계약자명, 계좌정보 등
- **`insurance_claim`**: 보험 청구 관련 서류
  - 간병인 지원 서비스 신청서: 피보험자 정보, 병원 정보, 진단명 등
  - 도난/파손 사실 확인서: 사고 정보, 분실 물품 정보 등
- **`auto_transfer_approval`**: 자동이체 승인서
  - 주요 필드: 지로계좌, 사업자 정보, 은행 정보, 담당자 정보 등
- **`delegation_insurance`**: 위임장 (보험용)
  - 주요 필드: 피보험자명, 사고일자, 증권번호, 위임인/수임인 정보 등

### 기타 업무 (5종)
- **`virtual_account`**: 기업인터넷뱅킹 가상계좌발급 목적확인서
  - 주요 필드: 사업자 정보, 계좌정보, 예상 가상계좌 수, 이용목적 등
- **`delegation_general`**: 위임장 (일반용)
  - 주요 필드: 대리인 정보, 위임인 정보, 위임 사항 등
- **`auto_transfer_insurance`**: 자동이체신청서 (보험용)
  - 주요 필드: 증권번호, 상품명, 예금주 정보, 신청인 정보 등
- **`nursing_service`**: 간병인 지원 서비스 신청서
  - 주요 필드: 피보험자 정보, 병원 정보, 서비스 신청 정보 등
- **`theft_damage_report`**: 도난/파손 사실 확인서
  - 주요 필드: 신고인 정보, 사고 정보, 분실 물품 상세 정보 등

각 문서 유형별로 정확한 좌표 정보와 필드명이 `backend/document/labelings.json`에 정의되어 있으며, 박스 라벨링 알고리즘을 통해 자동으로 매칭됩니다.

## 🔧 개발 가이드

### 프론트엔드 개발

#### 스크립트
```bash
npm run dev      # 개발 서버 (포트 3000)
npm run build    # 프로덕션 빌드
npm run preview  # 빌드 미리보기
```

#### 환경 변수 (.env)
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MOCK_API=false
VITE_DEBUG_MODE=true
```

#### 새로운 문서 유형 추가 방법
1. `DocumentTypeSelector.tsx`의 `documentCategories`에 옵션 추가
2. `backend/document/labelings.json`에 좌표 정보 추가
3. 백엔드 라우터에서 새 타입 처리 로직 구현

### 백엔드 개발

#### 새로운 문서 모델 추가
`backend/document/labelings.json`에 다음과 같은 형태로 추가:
```json
{
  "document_type_name": {
    "bin_box": [
      [x_coordinate, y_coordinate, "field_name"],
      [350, 150, "customer_name"],
      [500, 200, "account_number"]
    ]
  }
}
```

## 🚀 배포 가이드

### Docker를 활용한 배포
```bash
# 전체 스택 빌드 및 실행
docker-compose up --build

# 프론트엔드만 빌드
docker build -t finsight-frontend ./frontend

# 백엔드만 빌드
docker build -t finsight-backend ./backend
```

### 수동 배포
```bash
# 프론트엔드 빌드
cd frontend
npm run build
# dist/ 폴더를 웹서버에 배포

# 백엔드 실행
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 성능 최적화

#### 프론트엔드
- **파일 크기**: 이미지 500KB 이하, 총 2MB 이하
- **로딩 시간**: 3G 환경 3초 이내, WiFi 환경 1초 이내
- **접근성**: WCAG 2.1 AA 수준 준수

#### 백엔드
- **API 응답시간**: 200ms 이내 (평균 문서)
- **OCR 정확도**: 87-98% (문서 유형별)
- **박스 라벨링 정확도**: 95% 이상 (동일 폼 기준)
- **동시 처리**: 다중 요청 병렬 처리

## ⚠️ 보안 고려사항

- **데이터 보호**: 모든 처리를 On-Premise 환경에서 수행
- **파일 검증**: 악성코드 파일 업로드 방지 검사 (최대 10MB)
- **CORS 설정**: 허용된 도메인에서만 API 접근 가능
- **문자 인코딩**: UTF-8 기반 파이프라인 처리 안정성

## 📊 성능 벤치마크

| 구분 | 메트릭 | 목표값 | 현재값 | 상태 |
|------|-------|-------|-------|------|
| **OCR 성능** | 정확도 | >90% | 87-98% | ✅ |
| **API 성능** | 응답시간 | <500ms | <200ms | ✅ |
| **UI 성능** | 로딩시간(3G) | <3s | <1s | ✅ |
| **UI 성능** | 로딩시간(WiFi) | <1s | <500ms | ✅ |
| **접근성** | WCAG 준수 | AA | AA | ✅ |
| **동시처리** | 요청수 | >10 | >20 | ✅ |

## 🧪 테스트 가이드

### 프론트엔드 테스트
```bash
cd frontend
npm run test        # 유닛 테스트
npm run test:e2e    # E2E 테스트
npm run test:coverage # 커버리지 리포트
```

### 백엔드 테스트
```bash
cd backend
pytest              # 유닛 테스트
pytest --cov=.      # 커버리지 포함
python -m pytest tests/ -v  # 상세 출력
```

### API 테스트
```bash
# FastAPI 테스트 UI 접속
http://localhost:8000/docs

# 또는 curl을 통한 API 테스트
curl -X POST "http://localhost:8000/insert" \
  -F "filename=@test_document.png" \
  -F "doc_type=auto_transfer"
```

## 🔍 디버깅 가이드

### 로그 확인
```bash
# 백엔드 로그
tail -f backend/logs/app.log

# Docker 로그
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 일반적인 문제 해결

1. **모델 로딩 실패**
   - 모델 파일 경로 확인: `backend/modelrun/saved_models/`
   - CUDA 메모리 부족: 배치 사이즈 조정

2. **API 연결 실패**
   - CORS 설정 확인: `main.py`의 origins 리스트
   - 포트 충돌 확인: 8000번 포트 사용 여부

3. **OCR 정확도 저하**
   - 이미지 품질 확인: 해상도, 노이즈, 기울기
   - 문서 타입 정확성: 올바른 doc_type 선택

## 🚀 성능 모니터링

### WandB 연동
```bash
# WandB 로그인
wandb login

# 실험 추적 시작
python train_model.py --wandb_project "FinSight-OCR"
```

### 메트릭 대시보드
- **학습 메트릭**: 손실, 정확도, 학습률
- **시스템 메트릭**: GPU 사용률, 메모리 사용량
- **비즈니스 메트릭**: 처리 시간, 성공률

---

더 자세한 정보는 각 모듈별 코드 주석과 docstring을 참고하세요.