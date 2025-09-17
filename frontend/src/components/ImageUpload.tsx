'use client';

import React, { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Upload, X, AlertCircle, CheckCircle, FileImage } from 'lucide-react';
import { ImageWithFallback } from './ui/ImageWithFallback';
import { isValidFileType, isValidFileSize, uploadConfig } from '../config/api';

interface ImageUploadProps {
  onImageUpload: (file: File, dataUrl: string) => void;
  uploadedImage: string | null;
  onClearImage: () => void;
  onError?: (error: string) => void;
}

export function ImageUpload({
  onImageUpload,
  uploadedImage,
  onClearImage,
  onError
}: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validateFile = (file: File): string | null => {
    // 파일 타입 검증
    if (!isValidFileType(file)) {
      const allowedTypes = uploadConfig.allowedTypes.join(', ');
      return `지원되지 않는 파일 형식입니다. 지원 형식: ${allowedTypes}`;
    }

    // 파일 크기 검증
    if (!isValidFileSize(file)) {
      const maxSizeMB = uploadConfig.maxFileSize / (1024 * 1024);
      return `파일 크기가 너무 큽니다. 최대 크기: ${maxSizeMB}MB`;
    }

    return null;
  };

  const handleFileSelect = (file: File) => {
    setValidationError(null);

    // 파일 유효성 검사
    const error = validateFile(file);
    if (error) {
      setValidationError(error);
      onError?.(error);
      return;
    }

    // 파일 읽기
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      onImageUpload(file, result);
    };
    reader.onerror = () => {
      const errorMsg = '파일을 읽을 수 없습니다.';
      setValidationError(errorMsg);
      onError?.(errorMsg);
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleClearImage = () => {
    setValidationError(null);
    onClearImage();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };


  return (
    <div className="w-full bg-card backdrop-blur-md border border-border rounded-3xl shadow-2xl overflow-hidden">
      <div className="p-8">
        {uploadedImage ? (
          <div className="space-y-6">
            {/* 업로드 성공 헤더 */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="bg-green-100 p-2 rounded-xl">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-foreground">업로드 완료</h3>
                  <p className="text-sm text-muted-foreground">이미지가 성공적으로 업로드되었습니다</p>
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearImage}
                className="flex items-center gap-2"
              >
                <X className="w-4 h-4" />
                제거
              </Button>
            </div>

            {/* 업로드된 이미지 */}
            <div className="bg-muted/50 border border-border rounded-2xl overflow-hidden shadow-xl">
              <ImageWithFallback
                src={uploadedImage}
                alt="업로드된 문서"
                className="w-full h-auto max-h-96 object-contain"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* 드래그 앤 드롭 영역 */}
            <div
              className={`border-2 border-dashed rounded-3xl p-12 text-center transition-all duration-300 cursor-pointer relative overflow-hidden ${
                isDragging
                  ? 'border-primary bg-primary/5 scale-105'
                  : validationError
                  ? 'border-destructive bg-destructive/5'
                  : 'border-muted-foreground/30 hover:border-primary/50 hover:bg-accent/50'
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="relative z-10">
                {/* 아이콘 */}
                <div className={`w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-6 border ${
                  validationError 
                    ? 'bg-destructive/10 border-destructive/20' 
                    : 'bg-primary/10 border-primary/20'
                }`}>
                  {validationError ? (
                    <AlertCircle className="w-10 h-10 text-destructive" />
                  ) : (
                    <Upload className="w-10 h-10 text-primary" />
                  )}
                </div>

                {/* 제목 */}
                <div className="bg-card backdrop-blur-sm px-6 py-2 rounded-2xl border border-border inline-block mb-3">
                  <h3 className="text-xl text-foreground">
                    {validationError ? '업로드 오류' : '문서 이미지 업로드'}
                  </h3>
                </div>

                {/* 설명 */}
                <div className="space-y-3">
                  {validationError ? (
                    <p className="text-destructive text-sm font-medium">
                      {validationError}
                    </p>
                  ) : (
                    <p className="text-muted-foreground mb-6 text-lg">
                      클릭하거나 파일을 여기로 드래그하세요
                    </p>
                  )}

                  {/* 파일 제한 정보 */}
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>지원 형식: JPEG, PNG, WebP</p>
                    <p>최대 크기: {formatFileSize(uploadConfig.maxFileSize)}</p>
                  </div>
                </div>

                {/* 업로드 버튼 */}
                <div className="mt-6">
                  <Button
                    variant={validationError ? "destructive" : "default"}
                    className="px-8 py-3 text-lg w-full"
                  >
                    <FileImage className="w-4 h-4 mr-2" />
                    {validationError ? '다시 선택' : '파일 선택'}
                  </Button>
                </div>
              </div>

              {/* 숨겨진 파일 입력 */}
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileInputChange}
                accept={uploadConfig.allowedTypes.join(',')}
                className="hidden"
              />
            </div>
            
            {/* 오류 메시지 */}
            <div className="text-center text-sm text-muted-foreground">
              {validationError ? (
                <p className="text-destructive flex items-center justify-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  {validationError}
                </p>
              ) : (
                <p>FinSight는 모든 정보의 기록 및 저장을 하지 않습니다.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}