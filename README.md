# FinSight-OCR

금융 문서 전용 AI 기반 OCR 솔루션

![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=flat-square&logo=typescript)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch)

## 📋 프로젝트 소개

FinSight-OCR은 **AI Hub 한글 금융 문서 데이터셋**을 활용한 전문 OCR 솔루션입니다. 2단계 파이프라인(텍스트 검출 + 텍스트 인식)을 통해 은행, 보험, 증권 등 다양한 금융 문서에서 높은 정확도로 한글 텍스트를 자동 추출합니다.

### 핵심 기능
- **2단계 OCR 파이프라인**: DBNet 텍스트 검출 + CRNN 텍스트 인식
- **한글 금융 문서 특화**: 13종 금융 문서 유형 지원
- **지능형 박스 라벨링**: 자동 필드 매칭 시스템
- **실용적 성능**: 87-98% OCR 정확도 달성

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/TeamFinSight/FinSight-OCR.git
cd FinSight-OCR
```

### 2. 모델 파일 준비 (필수)
> ⚠️ **중요**: OCR 처리를 위해서는 `backend/modelrun/saved_models/` 디렉터리에 다음 `.pth` 파일들이 필수적으로 있어야 합니다:
> - `dbnet_a100_best.pth` (텍스트 검출 모델)
> - `robust_korean_recognition_best.pth` (텍스트 인식 모델)

### 3. Docker로 실행 (권장)
```bash
# 전체 스택 빌드 및 실행
docker-compose up --build
```

### 4. 개별 실행
#### 백엔드
```bash
cd backend

# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate finsight-ocr

# 서버 실행
uvicorn main:app --reload
```

#### 프론트엔드
```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

### 5. 접속 확인
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 🛠️ 기술 스택

### Frontend
- React 18 + TypeScript + Vite
- shadcn/ui + Tailwind CSS
- 드래그 앤 드롭 파일 업로드
- 다양한 출력 형식 (Excel, CSV, JSON)

### Backend
- FastAPI + Python 3.9+
- PyTorch 2.1+ 기반 AI 모델
- DBNet (텍스트 검출) + CRNN (텍스트 인식)
- 13종 금융 문서 지원

## 📝 지원 문서 유형

| 카테고리 | 문서 유형 |
|----------|-----------|
| **은행 업무** | 일반 양식, 신분위임장, 자동이체 신청서, 계좌개설 신청서 |
| **보험 업무** | 보험계약대출 승계 동의서, 보험 청구 서류, 자동이체 승인서, 위임장 |
| **기타 업무** | 가상계좌발급 목적확인서, 일반 위임장, 간병인 지원 서비스, 도난/파손 확인서 |

## 📊 성능 지표

| 항목 | 목표 | 현재 달성률 |
|------|------|-----------|
| OCR 정확도 | >90% | 87-98% |
| API 응답 시간 | <500ms | <200ms |
| 프론트엔드 로딩 | <3초 (3G) | <1초 |
| 박스 라벨링 정확도 | >90% | 95%+ |

## 추가 기능
- ✅ **실시간 편집**: 테이블 셀 클릭으로 즉시 수정 가능
- ✅ **행/열 추가**: 동적으로 데이터 확장 가능
- ✅ **다양한 내보내기**: Excel, CSV, JSON 형식 지원

## 📁 프로젝트 구조

```
FinSight-OCR/
├── backend/                    # FastAPI 백엔드 서버
│   ├── main.py                # FastAPI 메인 애플리케이션
│   ├── boxLabel.py            # 박스 라벨링 및 필드 매칭 알고리즘
│   ├── document/              # 문서 타입 및 라벨링 설정
│   │   ├── document_types.json
│   │   ├── labelings.json
│   │   └── boxlabeling.txt
│   ├── modelrun/              # AI 모델 실행 및 관리
│   │   ├── scripts/           # 모델 학습 및 실행 스크립트
│   │   │   ├── detection/     # DBNet 텍스트 검출 모델
│   │   │   ├── recognition/   # CRNN 텍스트 인식 모델
│   │   │   └── tba/          # OCR 파이프라인 통합
│   │   ├── configs/           # 모델 설정 파일
│   │   ├── saved_models/      # 학습된 모델 저장소 (⚠️ .pth 파일 필수)
│   │   │   ├── dbnet_a100_best.pth          # DBNet 텍스트 검출 모델
│   │   │   ├── robust_korean_recognition_best.pth  # CRNN 텍스트 인식 모델
│   │   │   └── README.md                    # 모델 파일 설명
│   │   └── output/           # 모델 실행 결과
│   ├── output/               # API 처리 결과 저장
│   └── requirements.txt      # Python 의존성
├── frontend/                  # React 프론트엔드
│   ├── src/
│   │   ├── App.tsx           # 메인 애플리케이션 컴포넌트
│   │   ├── components/       # UI 컴포넌트
│   │   │   ├── DocumentTypeSelector.tsx  # 문서 타입 선택
│   │   │   ├── ImageUpload.tsx           # 이미지 업로드
│   │   │   ├── OCRProcessor.tsx          # OCR 처리 결과 표시
│   │   │   ├── GenericTable.tsx          # 편집 가능한 테이블
│   │   │   ├── ResultExporter.tsx        # 결과 내보내기
│   │   │   └── ui/                       # shadcn/ui 컴포넌트
│   │   ├── services/         # API 통신 서비스
│   │   ├── types/           # TypeScript 타입 정의
│   │   ├── hooks/           # React 커스텀 훅
│   │   ├── constants/       # 상수 정의
│   │   └── config/          # 애플리케이션 설정
│   ├── package.json         # Node.js 의존성
│   └── dist/               # 빌드 결과물
├── docker-compose.yml       # Docker 컨테이너 구성
├── docker-compose.dev.yml   # 개발용 Docker 구성
├── environment.yml          # Conda 환경 설정
├── requirements.txt         # 전체 프로젝트 Python 의존성
├── DOCKER.md               # Docker 설정 가이드
└── README_DEV.md           # 상세 개발 가이드
```

## 📚 추가 문서

- **[개발 가이드](README_DEV.md)**: 상세한 개발 환경 설정 및 모델 학습 가이드
- **[Docker 가이드](DOCKER.md)**: Docker 컨테이너 설정 및 배포 가이드
- **[API 문서](http://localhost:8000/docs)**: FastAPI 자동 생성 API 문서

## 👥 개발팀

### 팀명: FinSight
| 역할 | 담당자 | 주요 업무 |
|------|--------|----------|
| **팀 리더** | 김태식 | AI 모델링 및 시스템 아키텍처 총괄 |
| **프론트엔드** | 강성룡 | React 개발 및 GitHub 리포지터리 관리, OCR 파이프라인 및 모델 통합  |
| **시스템 구축** | 경준오 | 프로젝트 전체 관리, 모델 탐색 및 학습 |
| **백엔드** | 김선우 | FastAPI 백엔드 개발 총괄 |
| **백엔드** | 백승빈 | 데이터 처리 및 박스 라벨링 알고리즘, 백엔드 개발 |

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

---

<div align="center">
  <strong>FinSight-OCR</strong> - 금융 문서 특화 AI OCR 솔루션<br>
  Made with ❤️ by Team FinSight
</div>