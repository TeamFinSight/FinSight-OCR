import { APIConfig } from '../types';


// API 설정
export const apiConfig: APIConfig = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: parseInt(import.meta.env.VITE_API_TIMEOUT) || 30000,
  retries: parseInt(import.meta.env.VITE_API_RETRIES) || 3,
};

// API 엔드포인트
export const endpoints = {
  ocr: {
    // 백엔드 FastAPI의 실제 엔드포인트에 맞춤
    process: import.meta.env.VITE_OCR_ENDPOINT || '/insert',
    // 현재 백엔드는 상태 폴링 엔드포인트를 제공하지 않음 (호환을 위해 키만 유지)
    status: import.meta.env.VITE_STATUS_ENDPOINT || '/api/v1/status',
  },
  upload: import.meta.env.VITE_UPLOAD_ENDPOINT || '/api/v1/upload',
  health: '/api/v1/health',
};

// 파일 업로드 설정
export const uploadConfig = {
  maxFileSize: parseInt(import.meta.env.VITE_MAX_FILE_SIZE) || 10 * 1024 * 1024, // 10MB
  allowedTypes: (import.meta.env.VITE_ALLOWED_FILE_TYPES || 'image/jpeg,image/png,image/webp,application/pdf').split(','),
  chunkSize: 1024 * 1024, // 1MB chunks for large files
};

// 개발 모드 설정
export const devConfig = {
  debugMode: import.meta.env.VITE_DEBUG_MODE === 'true',
};

// API URL 생성 헬퍼
export const createApiUrl = (endpoint: string): string => {
  return `${apiConfig.baseURL}${endpoint}`;
};

// 파일 타입 검증
export const isValidFileType = (file: File): boolean => {
  return uploadConfig.allowedTypes.includes(file.type);
};

// 파일 크기 검증
export const isValidFileSize = (file: File): boolean => {
  return file.size <= uploadConfig.maxFileSize;
};