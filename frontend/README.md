# FinSight OCR

한국 금융 서류를 위한 AI 기반 OCR 솔루션

## 🚀 빠른 시작

```bash
npm install
npm run dev
```

[http://localhost:3000](http://localhost:3000)에서 애플리케이션을 확인하세요.

## 📋 주요 기능

- **문서 업로드**: 드래그 앤 드롭 또는 클릭으로 이미지 업로드 (JPEG, PNG, WebP)
- **OCR 처리**: 실시간 텍스트 추출 및 진행률 추적
- **표 인식**: 자동 표 구조 감지 및 파싱
- **데이터 내보내기**: Excel, CSV, JSON 형식으로 결과 다운로드
- **한국 금융 서류 최적화**: 한국 은행 및 금융 양식에 특화
- **반응형 디자인**: 모바일 우선 반응형 인터페이스
- **실시간 편집**: 추출된 데이터의 실시간 수정 및 검증

## 🛠 기술 스택

- **프론트엔드**: React 18 + TypeScript + Vite
- **UI 라이브러리**: shadcn/ui + Radix UI + Tailwind CSS
- **상태 관리**: React Hooks
- **빌드 도구**: Vite with SWC
- **스타일링**: Tailwind CSS with CSS Variables
- **아이콘**: Lucide React

## 📁 프로젝트 구조

```
/
├── .env.example          # 환경 변수 예제 파일
├── .gitignore           # Git 무시 파일 목록
├── .vscode/             # VS Code 설정
├── CLAUDE.md            # Claude Code 개발 가이드
├── README.md            # 프로젝트 문서
├── dist/                # 빌드 출력 디렉토리
├── index.html           # HTML 진입점
├── package.json         # NPM 패키지 설정
├── postcss.config.js    # PostCSS 설정
├── sample/              # 샘플 OCR 이미지 파일들
├── tailwind.config.js   # Tailwind CSS 설정
├── tsconfig.json        # TypeScript 설정
├── vite.config.ts       # Vite 빌드 설정
└── src/                 # 소스 코드
    ├── main.tsx              # 애플리케이션 진입점
    ├── App.tsx               # 메인 애플리케이션 컴포넌트
    ├── index.css             # 글로벌 스타일
    ├── vite-env.d.ts         # Vite 환경 타입 정의
    ├── types/                # TypeScript 타입 정의
    │   └── index.ts          # 공통 인터페이스 (TableData, OCRMetrics 등)
    ├── constants/            # 애플리케이션 상수
    │   ├── documentTypes.ts  # 문서 카테고리 및 타입 정의
    │   └── dummyOCRData.ts   # 모의 OCR 데이터
    ├── components/           # React 컴포넌트
    │   ├── ui/              # shadcn/ui 컴포넌트 라이브러리
    │   │   ├── badge.tsx
    │   │   ├── button.tsx
    │   │   ├── card.tsx
    │   │   ├── collapsible.tsx
    │   │   ├── ImageWithFallback.tsx
    │   │   ├── input.tsx
    │   │   ├── progress.tsx
    │   │   ├── table.tsx
    │   │   ├── tabs.tsx
    │   │   ├── use-mobile.ts
    │   │   └── utils.ts
    │   ├── DocumentTypeSelector.tsx  # 문서 타입 선택
    │   ├── DropdownInfo.tsx # 드롭다운 정보 컴포넌트
    │   ├── GenericTable.tsx # 편집 가능한 데이터 테이블
    │   ├── ImageUpload.tsx  # 드래그 앤 드롭 파일 업로드
    │   ├── OCRMetricsDisplay.tsx  # OCR 정확도 메트릭 표시
    │   ├── OCRProcessor.tsx # OCR 처리 컴포넌트
    │   └── ResultExporter.tsx  # 결과 내보내기 (Excel, CSV, JSON)
    ├── services/             # API 서비스 (FastAPI 연동용)
    │   ├── api.ts           # 범용 API 클라이언트
    │   └── ocrService.ts    # OCR 전용 서비스
    ├── hooks/                # 커스텀 React 훅
    │   ├── useErrorHandler.ts  # 에러 핸들링 훅
    │   └── useOCR.ts        # OCR 처리 훅
    └── config/               # 설정 파일
        └── api.ts           # API 설정 및 유틸리티
```

## 🔧 개발 환경

### 빌드 명령어

```bash
npm run dev      # 개발 서버 실행
npm run build    # 프로덕션 빌드
npm run preview  # 빌드 미리보기
```

### 환경 설정

`.env` 파일 생성:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MOCK_API=false  # 실제 API 사용
VITE_DEBUG_MODE=true
```

## 🎯 지원 문서 유형 (13가지)

### 은행 서류 (Banking Documents)
- 일반 양식 (General Form)
- 본인인증위임장 (Identity Delegation)
- 자동이체 신청서 (Auto Transfer)
- 계좌 개설 신청서 (Account Opening)

### 대출 서류 (Loan Documents)
- 대출 계약서 (Loan Contract)

### 보험 서류 (Insurance Documents)
- 보험 계약서 (Insurance Contract)
- 보험금 청구서 (Insurance Claim)

### 투자 서류 (Investment Documents)
- 증권 취득 신청서 (Securities Acquisition)

### 법적 서류 (Legal Documents)
- 고객 확인서 (Customer Verification)
- 동의서 (Agreement)
- 기타 금융 관련 서류

## 🏗 아키텍처

### 클린 아키텍처
- **타입**: `src/types/`에 중앙화된 TypeScript 인터페이스
- **상수**: `src/constants/`에 문서 정의
- **유틸리티**: `src/utils/`에 재사용 가능한 함수
- **컴포넌트**: 기능별로 정리된 UI 컴포넌트

### 상태 관리
App.tsx 레벨에서 React useState로 모든 상태 관리:
- `uploadedImage`: 미리보기용 파일 데이터 URL
- `imageFile`: 처리용 원본 File 객체
- `processedData`: 헤더와 행이 있는 TableData 구조
- `selectedDocumentType`: 선택된 문서 카테고리 ID
- `ocrMetrics`: 처리 정확도 메트릭 (precision, recall, F1-score)

### 데이터 플로우
1. 사용자 이미지 업로드 → `handleImageUpload`가 파일 상태 업데이트
2. 사용자 문서 타입 선택 → `handleDocumentTypeSelect`가 문서 카테고리 설정
3. OCR 처리 → `handleOCRComplete`가 추출된 데이터와 메트릭 업데이트
4. 사용자 데이터 편집 → `handleDataChange`가 테이블 데이터 실시간 업데이트
5. 내보내기 기능이 현재 테이블 상태를 읽어 다운로드

## 📊 성능 지표

- **번들 크기**: 초기 <500KB, 총 <2MB
- **로딩 시간**: 3G에서 <3초, WiFi에서 <1초
- **접근성**: WCAG 2.1 AA 준수
- **브라우저 지원**: 모던 브라우저 (ES2020+)
- **OCR 정확도**: 87-98% (문서 유형별 상이)

## 🔒 파일 검증

- **지원 형식**: JPEG, PNG, WebP
- **최대 파일 크기**: 10MB
- **보안**: 파일 타입 검증, 크기 제한
- **처리**: 클라이언트 측 검증 + 서버 측 확인

## 🚀 FastAPI 연동

### API 연동 아키텍처
- **환경 설정**: `.env` 파일로 API 엔드포인트 관리
- **서비스 계층**: API 클라이언트 및 OCR 서비스
- **타입 안정성**: FastAPI 응답에 맞춘 TypeScript 인터페이스
- **에러 핸들링**: 전역 에러 상태 관리 및 재시도 로직
- **실시간 진행률**: 파일 업로드 및 OCR 처리 상태 추적

### 주요 파일
- `src/services/api.ts` - 범용 API 클라이언트 (재시도, 타임아웃)
- `src/services/ocrService.ts` - OCR 전용 서비스 (폴링, 상태 관리)
- `src/hooks/useOCR.ts` - OCR 처리를 위한 React 훅
- `src/hooks/useErrorHandler.ts` - 에러 핸들링을 위한 React 훅
- `src/config/api.ts` - API 설정 및 유틸리티

## 🎨 디자인 시스템

- **테마**: 백드롭 블러 효과가 있는 커스텀 Tailwind 설정
- **컴포넌트**: shadcn/ui 컴포넌트 라이브러리 완전 통합
- **반응형**: 글래스 모피즘 스타일링의 모바일 우선 디자인
- **접근성**: 키보드 내비게이션을 지원하는 WCAG 준수 컴포넌트

## 📝 개발 가이드라인

### 새 문서 타입 추가
1. `DocumentTypeSelector.tsx`의 `documentCategories`에 문서 정의 추가
2. `OCRProcessor.tsx` 모의 데이터에 해당 샘플 텍스트 포함
3. 적절한 아이콘 및 카테고리 할당 확인

### 테이블 구조 수정
- 테이블 데이터는 `TableData` 인터페이스를 따름: `{ headers: string[], rows: string[][] }`
- 모든 테이블 작업은 검증을 통해 데이터 일관성 유지
- 실시간 편집은 적절한 상태 동기화 필요

### UI 컴포넌트 사용
- 모든 UI 컴포넌트는 `src/components/ui/`에 위치한 shadcn/ui 라이브러리 사용
- 일관성을 위해 기존 디자인 토큰 및 스타일링 패턴 사용
- 백드롭 블러 및 글래스 모피즘 효과를 일관되게 적용

## 🔧 빌드 설정

### Vite 설정
- **출력**: `dist/` 디렉토리
- **포트**: 3000 (자동 브라우저 열기)
- **별칭**: `@/`는 `src/`에 매핑
- **SWC**: 빠른 빌드를 위한 React 컴파일

### Tailwind CSS
- **shadcn/ui**: CSS 변수를 사용한 완전한 컴포넌트 라이브러리
- **커스텀 변수**: 라이트/다크 테마 지원
- **PostCSS**: Autoprefixer 포함

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.