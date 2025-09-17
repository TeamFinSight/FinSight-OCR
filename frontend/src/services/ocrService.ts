import { 
  OCRRequest, 
  OCRResponse, 
  APIResponse, 
  ProcessingStatus, 
  UploadProgress 
} from '../types';
import { apiClient } from './api';
import { endpoints } from '../config/api';

export class OCRService {
  // FastAPI 백엔드와 연동하여 OCR 처리
  async processOCR(
    request: OCRRequest,
    onProgress?: (progress: UploadProgress) => void,
    onStatusChange?: (status: ProcessingStatus) => void
  ): Promise<APIResponse<OCRResponse>> {

    try {
      // 상태 업데이트: 업로드 시작
      onStatusChange?.({
        status: 'uploading',
        message: '이미지를 업로드하는 중...',
      });

      // 파일 업로드 및 OCR 처리 요청
      const uploadResponse = await apiClient.uploadFile<{
        task_id: string;
        status: string;
      }>(
        endpoints.ocr.process,
        request.image,
        {
          document_type: request.document_type,
          options: request.options || {},
        },
        onProgress
      );

      if (!uploadResponse.success || !uploadResponse.data) {
        throw new Error(uploadResponse.error || '업로드 실패');
      }

      const { task_id } = uploadResponse.data;

      // 상태 업데이트: 처리 중
      onStatusChange?.({
        status: 'processing',
        message: 'OCR 처리 중...',
      });

      // 처리 상태 폴링
      const result = await this.pollProcessingStatus(task_id, onStatusChange);

      if (!result.success) {
        throw new Error(result.error || 'OCR 처리 실패');
      }

      // 상태 업데이트: 완료
      onStatusChange?.({
        status: 'completed',
        message: 'OCR 처리가 완료되었습니다.',
      });

      return result;

    } catch (error) {
      // 상태 업데이트: 오류
      onStatusChange?.({
        status: 'error',
        message: error instanceof Error ? error.message : '알 수 없는 오류',
      });

      return {
        success: false,
        error: error instanceof Error ? error.message : '알 수 없는 오류',
      };
    }
  }

  // 처리 상태 폴링
  private async pollProcessingStatus(
    taskId: string,
    onStatusChange?: (status: ProcessingStatus) => void
  ): Promise<APIResponse<OCRResponse>> {
    const maxAttempts = 60; // 최대 5분 (5초 간격)
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const statusResponse = await apiClient.get<{
          status: string;
          progress?: number;
          result?: OCRResponse;
          error?: string;
        }>(`${endpoints.ocr.status}/${taskId}`);

        if (!statusResponse.success || !statusResponse.data) {
          throw new Error('상태 확인 실패');
        }

        const { status, progress, result, error } = statusResponse.data;

        if (status === 'completed' && result) {
          return {
            success: true,
            data: result,
          };
        }

        if (status === 'failed') {
          throw new Error(error || 'OCR 처리 실패');
        }

        // 진행률 업데이트
        if (progress !== undefined) {
          onStatusChange?.({
            status: 'processing',
            message: `OCR 처리 중... (${Math.round(progress)}%)`,
            progress: {
              loaded: progress,
              total: 100,
              percentage: progress,
            },
          });
        }

        // 5초 대기 후 재시도
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;

      } catch (error) {
        throw new Error(error instanceof Error ? error.message : '상태 확인 오류');
      }
    }

    throw new Error('처리 시간 초과');
  }


  // 지원되는 문서 타입 조회
  async getSupportedDocumentTypes(): Promise<APIResponse<string[]>> {
    return apiClient.get<string[]>('/api/v1/ocr/document-types');
  }

  // OCR 처리 상태 확인
  async getProcessingStatus(taskId: string): Promise<APIResponse<{
    status: string;
    progress?: number;
    result?: OCRResponse;
    error?: string;
  }>> {
    return apiClient.get(`${endpoints.ocr.status}/${taskId}`);
  }
}

// 싱글톤 인스턴스
export const ocrService = new OCRService();