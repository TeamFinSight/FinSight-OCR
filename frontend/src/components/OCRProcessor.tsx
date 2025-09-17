'use client';

import { useEffect } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Loader2, FileText, CheckCircle, AlertCircle, Upload } from 'lucide-react';
import { TableData } from '../types';
import { useOCR } from '../hooks/useOCR';
import { isValidFileType, isValidFileSize } from '../config/api';

interface OCRProcessorProps {
  imageFile: File | null;
  documentType: string;
  onProcessComplete: (extractedData: TableData, originalText: string) => void;
  onError?: (error: string) => void;
}

export function OCRProcessor({ 
  imageFile, 
  documentType, 
  onProcessComplete,
  onError 
}: OCRProcessorProps) {
  const {
    isProcessing,
    status,
    progress,
    error,
    extractedText,
    tableData,
    metrics,
    processOCR,
    reset
  } = useOCR();

  // OCR 처리 완료 시 결과를 부모 컴포넌트에 전달
  useEffect(() => {
    if (status.status === 'completed' && tableData) {
      onProcessComplete(tableData, extractedText);
    }
  }, [status.status, tableData, extractedText, onProcessComplete]);

  const handleProcessOCR = async () => {
    if (!imageFile) return;

    // 파일 유효성 검사
    if (!isValidFileType(imageFile)) {
      const errorMsg = '지원되지 않는 파일 형식입니다.';
      onError?.(errorMsg);
      return;
    }

    if (!isValidFileSize(imageFile)) {
      const errorMsg = '파일 크기가 너무 큽니다. (최대 10MB)';
      onError?.(errorMsg);
      return;
    }

    try {
      await processOCR({
        image: imageFile,
        document_type: documentType,
        options: {
          language: 'ko',
          extract_tables: true,
          extract_text: true,
        },
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '알 수 없는 오류';
      onError?.(errorMessage);
    }
  };

  const handleReset = () => {
    reset();
  };

  // 처리 상태에 따른 UI 렌더링
  const renderStatus = () => {
    if (error) {
      return (
        <div className="space-y-4">
          <div className="bg-destructive/10 backdrop-blur-sm rounded-xl p-4 border border-destructive/20">
            <div className="flex items-center gap-3 mb-3">
              <AlertCircle className="w-5 h-5 text-destructive" />
              <span className="text-destructive font-medium">오류 발생</span>
            </div>
            <p className="text-destructive/80 text-sm mb-3">{error}</p>
            <Button 
              onClick={handleReset}
              variant="outline"
              size="sm"
              className="border-destructive/30 hover:bg-destructive/5"
            >
              다시 시도
            </Button>
          </div>
        </div>
      );
    }

    if (status.status === 'completed') {
      return (
        <div className="space-y-4">
          <div className="bg-green-50 backdrop-blur-sm rounded-xl p-4 border border-green-200">
            <div className="flex items-center gap-3 mb-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-green-700 font-medium">처리 완료</span>
            </div>
            <p className="text-green-600 text-sm">
              OCR 처리가 성공적으로 완료되었습니다.
            </p>
          </div>
          
          <Button 
            onClick={handleReset}
            variant="outline"
            size="sm"
            className="w-full"
          >
            새로 처리하기
          </Button>
        </div>
      );
    }

    if (isProcessing) {
      return (
        <div className="space-y-4">
          <div className="bg-blue-50 backdrop-blur-sm rounded-xl p-4 border border-blue-200">
            <div className="space-y-3">
              {/* 진행률 표시 */}
              {progress && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-blue-700">{status.message}</span>
                    <span className="text-blue-600">{progress.percentage}%</span>
                  </div>
                  <Progress 
                    value={progress.percentage} 
                    className="w-full h-2 bg-blue-100" 
                  />
                </div>
              )}
              
              {/* 상태 메시지 */}
              <div className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                  <Loader2 className="w-3 h-3 animate-spin text-white" />
                </div>
                <span className="text-blue-600 text-sm">
                  {status.message || '처리 중...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return null;
  };

  if (!imageFile) {
    return (
      <div className="text-center space-y-4">
        <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-sm w-16 h-16 rounded-2xl flex items-center justify-center mx-auto border border-white/20">
          <FileText className="w-8 h-8 text-purple-600" />
        </div>
        <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20 inline-block">
          <p className="text-slate-600">이미지를 업로드하면 OCR 처리를 시작할 수 있습니다.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 업로드된 파일 정보 */}
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
        <div className="flex items-center gap-3 mb-3">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <span className="text-slate-700">이미지가 업로드되었습니다</span>
        </div>
        <div className="text-sm text-slate-600 space-y-1">
          <p>파일명: {imageFile.name}</p>
          <p>크기: {(imageFile.size / 1024 / 1024).toFixed(2)} MB</p>
          <p>문서 유형: {documentType}</p>
        </div>
      </div>

      {/* 상태 표시 */}
      {renderStatus()}

      {/* 처리 시작 버튼 */}
      {!isProcessing && status.status !== 'completed' && !error && (
        <Button 
          onClick={handleProcessOCR} 
          className="w-full h-12 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white border-none shadow-lg transition-all duration-300 transform hover:scale-[1.02]"
          disabled={!imageFile || !documentType}
        >
          <Upload className="w-4 h-4 mr-2" />
          OCR 처리 시작
        </Button>
      )}
    </div>
  );
}