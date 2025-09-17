import {
  OCRRequest,
  OCRResponse,
  APIResponse,
  ProcessingStatus,
  UploadProgress,
  TableData,
} from '../types';
import { apiClient } from './api';
import { endpoints } from '../config/api';
import { mapFrontendIdToBackendTypeName } from '../constants/documentTypeMapping';

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

      // 프론트엔드 문서 타입 ID를 백엔드 type_name으로 변환
      const backendDocumentType = mapFrontendIdToBackendTypeName(request.document_type);
      console.log(`문서 타입 매핑: ${request.document_type} -> ${backendDocumentType}`);

      // 파일 업로드 및 OCR 처리 요청 (백엔드 요구: filename, doc_type)
      const uploadResponse = await apiClient.uploadFile<any>(
        endpoints.ocr.process,
        request.image,
        {
          // FastAPI main.py의 /insert는 filename: UploadFile, doc_type: Form을 기대
          // uploadFile은 FormData에 문자열을 넣으므로 filename은 파일 자체로 대체되고,
          // 추가 데이터에는 doc_type만 실질적으로 사용됨
          // filename 필드는 FastAPI에서 UploadFile로 수신하므로 파일 파트 이름을 'filename'으로 맞춰야 함
          // 아래에서 FormData 키를 오버라이드하도록 uploadFile을 확장 호출
          __customFileFieldName: 'filename',
          doc_type: backendDocumentType, // 매핑된 백엔드 문서 타입 사용
          // 유지: 옵션은 현재 백엔드에서 사용하지 않지만 호환 목적으로 전송
          options: request.options || {},
        },
        onProgress
      );

      if (!uploadResponse.success || !uploadResponse.data) {
        throw new Error(uploadResponse.error || '업로드 실패');
      }

      // 백엔드는 즉시 JSON 결과를 반환함. 이를 프론트의 OCRResponse 스키마로 매핑
      const adapted = this.adaptBackendResultToFrontend(uploadResponse.data);

      // 상태 업데이트: 완료
      onStatusChange?.({
        status: 'completed',
        message: 'OCR 처리가 완료되었습니다.',
      });

      return { success: true, data: adapted };

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

  // 백엔드 JSON -> 프론트 OCRResponse 어댑터
  private adaptBackendResultToFrontend(backendJson: any): OCRResponse {
    // 백엔드에서 오류 응답이 온 경우 처리
    if (backendJson?.status === 'error') {
      throw new Error(backendJson.message || 'OCR 처리 중 오류가 발생했습니다.');
    }

    // backend 구조 예시 (run_ocr.py):
    // {
    //   metadata: {...},
    //   document_info: { width, height, document_type },
    //   fields: [{ id, labels, rotation, value_text, confidence, value_box: { x:[], y:[], type:'polygon' } }]
    // }

    const extracted_text = Array.isArray(backendJson?.fields)
      ? backendJson.fields.map((f: any) => f?.value_text).filter(Boolean).join('\n')
      : '';

    const headers = ['id', 'label', 'text', 'confidence', 'rotation', 'box_x', 'box_y'];
    const rows: string[][] = Array.isArray(backendJson?.fields)
      ? backendJson.fields.map((f: any) => [
          String(f?.id ?? ''),
          String(f?.labels ?? ''),
          String(f?.value_text ?? ''),
          String(typeof f?.confidence === 'number' ? f.confidence.toFixed(4) : ''),
          String(typeof f?.rotation === 'number' ? f.rotation.toFixed(2) : ''),
          Array.isArray(f?.value_box?.x) ? JSON.stringify(f.value_box.x) : '',
          Array.isArray(f?.value_box?.y) ? JSON.stringify(f.value_box.y) : '',
        ])
      : [];

    const table_data: TableData = { headers, rows };

    const processing_time = 0;
    const document_type = backendJson?.document_info?.document_type ?? 'unknown';
    const confidence_score = 0;

    return {
      extracted_text,
      table_data,
      processing_time,
      document_type,
      confidence_score,
    };
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