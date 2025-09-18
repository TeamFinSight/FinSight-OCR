import { useState, useCallback } from 'react';
import {
  OCRRequest,
  OCRResponse,
  ProcessingStatus,
  UploadProgress,
  TableData
} from '../types';
import { ocrService } from '../services/ocrService';

interface UseOCRResult {
  // 상태
  isProcessing: boolean;
  status: ProcessingStatus;
  progress: UploadProgress | null;
  result: OCRResponse | null;
  error: string | null;

  // 액션
  processOCR: (request: OCRRequest) => Promise<void>;
  reset: () => void;
  
  // 편의 getter
  extractedText: string;
  refinedTableData: TableData | null;
  rawTableData: TableData | null;

  // 하위 호환성 (기존 코드용)
  tableData: TableData | null;
}

export const useOCR = (): UseOCRResult => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<ProcessingStatus>({ status: 'idle' });
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [result, setResult] = useState<OCRResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const processOCR = useCallback(async (request: OCRRequest) => {
    try {
      setIsProcessing(true);
      setError(null);
      setResult(null);
      setProgress(null);

      const response = await ocrService.processOCR(
        request,
        // 진행률 콜백
        (uploadProgress) => {
          setProgress(uploadProgress);
        },
        // 상태 변경 콜백
        (processingStatus) => {
          setStatus(processingStatus);
          if (processingStatus.progress) {
            setProgress(processingStatus.progress);
          }
        }
      );

      if (response.success && response.data) {
        setResult(response.data);
        setStatus({ 
          status: 'completed',
          message: 'OCR 처리가 완료되었습니다.'
        });
      } else {
        throw new Error(response.error || 'OCR 처리 실패');
      }

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '알 수 없는 오류';
      setError(errorMessage);
      setStatus({ 
        status: 'error',
        message: errorMessage
      });
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const reset = useCallback(() => {
    setIsProcessing(false);
    setStatus({ status: 'idle' });
    setProgress(null);
    setResult(null);
    setError(null);
  }, []);

  // 편의 getter
  const extractedText = result?.extracted_text || '';
  const refinedTableData = result?.refined_table_data || null;
  const rawTableData = result?.raw_table_data || null;
  const tableData = refinedTableData; // 하위 호환성을 위해 정제된 데이터를 기본으로 사용

  return {
    isProcessing,
    status,
    progress,
    result,
    error,
    processOCR,
    reset,
    extractedText,
    refinedTableData,
    rawTableData,
    tableData,
  };
};