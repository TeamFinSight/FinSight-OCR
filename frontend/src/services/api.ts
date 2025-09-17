import { APIResponse, UploadProgress } from '../types';
import { apiConfig, createApiUrl, devConfig } from '../config/api';

class APIClient {
  private timeout: number;
  private retries: number;

  constructor() {
    this.timeout = apiConfig.timeout;
    this.retries = apiConfig.retries;
  }

  // 기본 fetch 래퍼 with 재시도 로직
  private async fetchWithRetry<T>(
    url: string,
    options: RequestInit,
    retryCount = 0
  ): Promise<APIResponse<T>> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      return {
        success: true,
        data,
      };
    } catch (error) {
      if (retryCount < this.retries) {
        // 지수 백오프로 재시도
        const delay = Math.pow(2, retryCount) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.fetchWithRetry(url, options, retryCount + 1);
      }

      return {
        success: false,
        error: error instanceof Error ? error.message : '알 수 없는 오류',
      };
    }
  }

  // GET 요청
  async get<T>(endpoint: string): Promise<APIResponse<T>> {
    const url = createApiUrl(endpoint);
    
    if (devConfig.debugMode) {
      console.log(`API GET: ${url}`);
    }

    return this.fetchWithRetry<T>(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // POST 요청
  async post<T>(endpoint: string, data?: any): Promise<APIResponse<T>> {
    const url = createApiUrl(endpoint);
    
    if (devConfig.debugMode) {
      console.log(`API POST: ${url}`, data);
    }

    return this.fetchWithRetry<T>(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // 파일 업로드 with 진행률 추적
  async uploadFile<T>(
    endpoint: string,
    file: File,
    additionalData?: Record<string, any>,
    onProgress?: (progress: UploadProgress) => void
  ): Promise<APIResponse<T>> {
    const url = createApiUrl(endpoint);
    
    if (devConfig.debugMode) {
      console.log(`API UPLOAD: ${url}`, { fileName: file.name, fileSize: file.size });
    }

    return new Promise((resolve) => {
      const formData = new FormData();
      formData.append('file', file);
      
      // 추가 데이터 추가
      if (additionalData) {
        Object.entries(additionalData).forEach(([key, value]) => {
          formData.append(key, typeof value === 'string' ? value : JSON.stringify(value));
        });
      }

      const xhr = new XMLHttpRequest();

      // 업로드 진행률 추적
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress: UploadProgress = {
            loaded: event.loaded,
            total: event.total,
            percentage: Math.round((event.loaded / event.total) * 100),
          };
          onProgress(progress);
        }
      });

      // 요청 완료 처리
      xhr.addEventListener('load', () => {
        try {
          if (xhr.status >= 200 && xhr.status < 300) {
            const response = JSON.parse(xhr.responseText);
            resolve({
              success: true,
              data: response,
            });
          } else {
            resolve({
              success: false,
              error: `HTTP ${xhr.status}: ${xhr.statusText}`,
            });
          }
        } catch (error) {
          resolve({
            success: false,
            error: '응답 파싱 오류',
          });
        }
      });

      // 에러 처리
      xhr.addEventListener('error', () => {
        resolve({
          success: false,
          error: '네트워크 오류',
        });
      });

      // 타임아웃 처리
      xhr.addEventListener('timeout', () => {
        resolve({
          success: false,
          error: '요청 시간 초과',
        });
      });

      xhr.timeout = this.timeout;
      xhr.open('POST', url);
      xhr.send(formData);
    });
  }

  // 건강 상태 확인
  async healthCheck(): Promise<APIResponse<{ status: string; timestamp: string }>> {
    return this.get('/api/v1/health');
  }
}

// 싱글톤 인스턴스
export const apiClient = new APIClient();