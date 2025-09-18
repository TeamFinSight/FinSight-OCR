# FinSight-OCR

금융 문서 전용 AI 기반 OCR 솔루션

![FinSight OCR Logo](https://img.shields.io/badge/FinSight-OCR-blue?style=for-the-badge)
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

### 2. Docker로 실행 (권장)
```bash
# 전체 스택 빌드 및 실행
docker-compose up --build
```

### 3. 개별 실행
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

### 4. 접속 확인
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

## 📋 OCR 결과 예시

### 입력: 자동이체 신청서 이미지

### 출력: 구조화된 데이터 테이블

| 필드명 | 추출된 값 | 신뢰도 |
|--------|-----------|--------|
| **사업자명** | (주)핀사이트테크놀로지 | 98% |
| **사업자등록번호** | 123-45-67890 | 95% |
| **대표자명** | 김태식 | 97% |
| **사업장주소** | 서울특별시 강남구 테헤란로 123 | 92% |
| **연락처** | 02-1234-5678 | 96% |
| **팩스번호** | 02-1234-5679 | 89% |
| **은행명** | 국민은행 | 99% |
| **계좌번호** | 123456-78-901234 | 94% |
| **예금주** | (주)핀사이트테크놀로지 | 96% |

### 추가 기능
- ✅ **실시간 편집**: 테이블 셀 클릭으로 즉시 수정 가능
- ✅ **행/열 추가**: 동적으로 데이터 확장 가능
- ✅ **다양한 내보내기**: Excel, CSV, JSON 형식 지원
- ✅ **신뢰도 표시**: 각 필드별 OCR 정확도 확인

## 📚 추가 문서

- **[개발 가이드](README_DEV.md)**: 상세한 개발 환경 설정 및 모델 학습 가이드
- **[API 문서](http://localhost:8000/docs)**: FastAPI 자동 생성 API 문서
- **[프로젝트 구조](README_DEV.md#-프로젝트-구조)**: 상세한 디렉터리 구조 설명

## 👥 개발팀

### 팀명: FinSight
| 역할 | 담당자 | 주요 업무 |
|------|--------|----------|
| **팀 리더** | 김태식 | AI 모델링 및 시스템 아키텍처 총괄 |
| **프론트엔드** | 강성룡 | React 개발 및 GitHub 리포지터리 관리 |
| **백엔드** | 경준오 | FastAPI 서버 개발 및 API 설계 |
| **백엔드** | 김선우 | OCR 파이프라인 및 모델 통합 |
| **백엔드** | 백승빈 | 데이터 처리 및 박스 라벨링 알고리즘 |

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  <strong>FinSight-OCR</strong> - 금융 문서 특화 AI OCR 솔루션<br>
  Made with ❤️ by Team FinSight
</div>