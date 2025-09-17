import { useState } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { OCRProcessor } from './components/OCRProcessor';
import { GenericTable } from './components/GenericTable';
import { TableData, OCRMetrics } from './types';
import { ResultExporter } from './components/ResultExporter';
import { OCRMetricsDisplay } from './components/OCRMetricsDisplay';
import DocumentTypeSelector from './components/DocumentTypeSelector';
import DropdownInfo from './components/DropdownInfo';
import { Card, CardContent } from './components/ui/card';
import { FileText, Zap, AlertCircle, TestTube } from 'lucide-react';
import { useErrorHandler } from './hooks/useErrorHandler';
import { dummyDataMap } from './constants/dummyOCRData';

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [processedData, setProcessedData] = useState<TableData | null>(null);
  const [originalText, setOriginalText] = useState<string>('');
  const [ocrMetrics, setOcrMetrics] = useState<OCRMetrics | null>(null);
  const [selectedDocumentType, setSelectedDocumentType] = useState<string | null>(null);
  const [isDummyData, setIsDummyData] = useState<boolean>(false);
  const [dummyDataInfo, setDummyDataInfo] = useState<{name: string, category: string} | null>(null);
  const [showExampleOCR, setShowExampleOCR] = useState<boolean>(false);
  const [currentDummyId, setCurrentDummyId] = useState<string | null>(null);

  const { errorState, setError, clearError } = useErrorHandler();
  
  const handleImageUpload = (file: File, dataUrl: string) => {
    setImageFile(file);
    setUploadedImage(dataUrl);
    // 새 이미지가 업로드되면 이전 처리 결과 초기화
    setProcessedData(null);
    setOriginalText('');
    setOcrMetrics(null);
    setSelectedDocumentType(null);
    setIsDummyData(false);
    setDummyDataInfo(null);
    setShowExampleOCR(false);
    setCurrentDummyId(null);
    clearError();
  };

  const handleClearImage = () => {
    setUploadedImage(null);
    setImageFile(null);
    setProcessedData(null);
    setOriginalText('');
    setOcrMetrics(null);
    setSelectedDocumentType(null);
    setIsDummyData(false);
    setDummyDataInfo(null);
    setShowExampleOCR(false);
    setCurrentDummyId(null);
    clearError();
  };

  const handleDocumentTypeSelect = (documentType: string) => {
    setSelectedDocumentType(documentType);
    // 문서 종류가 변경되면 이전 OCR 결과 초기화
    setProcessedData(null);
    setOriginalText('');
    setOcrMetrics(null);
    clearError();
  };

  const handleExampleOCRShow = (dummyId: string) => {
    const dummyData = dummyDataMap.get(dummyId);
    if (!dummyData) {
      setError('예시 데이터를 찾을 수 없습니다.');
      return;
    }

    // 모든 상태를 완전히 초기화
    setUploadedImage(null);
    setImageFile(null);
    setProcessedData(null);
    setOriginalText('');
    setOcrMetrics(null);
    setSelectedDocumentType(null);
    setIsDummyData(false);
    setDummyDataInfo(null);
    setShowExampleOCR(false);
    setCurrentDummyId(null);
    clearError();

    // 잠시 후 새로운 더미 데이터로 설정 (애니메이션 효과를 위해)
    setTimeout(() => {
      setSelectedDocumentType(dummyData.documentTypeId);
      setProcessedData(dummyData.tableData);
      setOriginalText(dummyData.originalText);
      setOcrMetrics(dummyData.metrics);
      setIsDummyData(true);
      setDummyDataInfo({
        name: dummyData.name,
        category: dummyData.category
      });
      setShowExampleOCR(true);
      setCurrentDummyId(dummyId);
    }, 100);
  };

  const handleOCRComplete = (tableData: TableData, text: string, metrics: OCRMetrics) => {
    setProcessedData(tableData);
    setOriginalText(text);
    setOcrMetrics(metrics);
    setIsDummyData(false);
    setDummyDataInfo(null);
    setShowExampleOCR(false);
    setCurrentDummyId(null);
    clearError();
  };

  const handleDataChange = (newData: TableData) => {
    setProcessedData(newData);
  };

  const handleError = (error: string) => {
    setError(error);
  };

  return (
    <div className="min-h-screen bg-secondary relative">
      {/* Background decoration */}
      <div className="absolute inset-0 bg-muted/30" />
      <div className="absolute top-0 left-1/4 w-72 h-72 lg:w-96 lg:h-96 bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-72 h-72 lg:w-96 lg:h-96 bg-primary/5 rounded-full blur-3xl" />
      
      <div className="container mx-auto px-4 py-6 lg:py-8 relative z-10">
        {/* Header */}
        <div className="text-center mb-8 lg:mb-12">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 lg:gap-4 mb-6">
            <div className="p-2 lg:p-3 bg-card backdrop-blur-lg rounded-2xl border border-border">
              <FileText className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />
            </div>
            <div className="bg-card backdrop-blur-md px-6 lg:px-8 py-3 lg:py-4 rounded-3xl border border-border">
              <h1 className="text-2xl sm:text-3xl lg:text-4xl text-primary">
                FinSight
              </h1>
            </div>
            <div className="p-2 lg:p-3 bg-card backdrop-blur-lg rounded-2xl border border-border">
              <Zap className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />
            </div>
          </div>
          
          {/* API 상태 표시 및 테스트 버튼 */}
          <div className="flex flex-col items-center gap-4 mb-4">
            
            {/* OCR 결과 예시 버튼 */}
            <div className="relative group">
              <div className="px-4 py-2 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 border border-emerald-300 rounded-xl cursor-pointer hover:border-emerald-400 hover:bg-emerald-50 transition-all duration-200">
                <div className="flex items-center gap-2">
                  <TestTube className="w-4 h-4 text-emerald-600" />
                  <span className="text-sm font-medium text-emerald-700">OCR 결과 예시 보기</span>
                </div>
              </div>

              {/* 드롭다운 메뉴 */}
              <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-card border border-border rounded-xl shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 min-w-max">
                <div className="p-2 space-y-1">
                  <div className="px-4 py-2 text-xs text-muted-foreground border-b border-border text-center">
                    클릭하면 OCR 처리 결과 예시를 확인할 수 있습니다
                  </div>
                  {Array.from(dummyDataMap.values()).slice(0, 6).map((dummy) => {
                    const isSelected = currentDummyId === dummy.id;
                    return (
                      <button
                        key={dummy.id}
                        onClick={() => handleExampleOCRShow(dummy.id)}
                        className={`w-full flex items-center gap-3 px-4 py-3 text-left rounded-lg transition-colors ${
                          isSelected 
                            ? 'bg-emerald-100 border border-emerald-300' 
                            : 'hover:bg-accent'
                        }`}
                      >
                        <div className={`p-2 rounded-lg ${
                          isSelected 
                            ? 'bg-emerald-200' 
                            : 'bg-emerald-500/10'
                        }`}>
                          <TestTube className="w-4 h-4 text-emerald-600" />
                        </div>
                        <div className="flex-1">
                          <div className={`font-medium text-sm ${
                            isSelected ? 'text-emerald-800' : ''
                          }`}>
                            {dummy.name}
                          </div>
                          <div className={`text-xs ${
                            isSelected 
                              ? 'text-emerald-600' 
                              : 'text-muted-foreground'
                          }`}>
                            {dummy.category} • 정확도 {Math.round(dummy.metrics.accuracy * 100)}%
                          </div>
                        </div>
                        {isSelected && (
                          <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <DropdownInfo />
        </div>

        {/* 전역 에러 표시 */}
        {errorState.hasError && (
          <div className="max-w-4xl mx-auto mb-6">
            <Card className="bg-destructive/10 border-destructive/20">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-destructive" />
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-destructive">오류</h4>
                    <p className="text-sm text-destructive/80">{errorState.error}</p>
                  </div>
                  <button
                    onClick={clearError}
                    className="text-destructive hover:text-destructive/80 text-xl leading-none"
                  >
                    ×
                  </button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Content - Responsive Layout */}
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 lg:gap-8">
            
            {/* Left Panel - Image Upload & OCR Processing */}
            <div className="space-y-6">
              <div className="bg-card backdrop-blur-md border border-border rounded-3xl overflow-hidden">
                <div className="bg-muted/50 px-4 lg:px-6 py-4 border-b border-border">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-accent backdrop-blur-sm rounded-xl border border-border">
                      <FileText className="w-5 h-5 lg:w-6 lg:h-6 text-primary" />
                    </div>
                    <h2 className="text-lg lg:text-xl text-foreground">이미지 업로드 & 처리</h2>
                  </div>
                </div>
                
                <div className="p-4 lg:p-6 space-y-6">
                  {/* Image Upload Section */}
                  <div className="space-y-4">
                    <div className="bg-accent backdrop-blur-sm px-4 py-2 rounded-xl border border-border inline-block">
                      <h3 className="text-base lg:text-lg text-accent-foreground">문서 업로드</h3>
                    </div>
                    <ImageUpload
                      onImageUpload={handleImageUpload}
                      uploadedImage={uploadedImage}
                      onClearImage={handleClearImage}
                      onError={handleError}
                    />
                  </div>

                  {/* Document Type Selection */}
                  {(uploadedImage || isDummyData) && (
                    <div className="animate-in slide-in-from-bottom duration-500 space-y-4">
                      <div className="h-px bg-border" />
                      <DocumentTypeSelector
                        selectedDocumentType={selectedDocumentType}
                        onDocumentTypeSelect={handleDocumentTypeSelect}
                      />
                    </div>
                  )}

                  {/* OCR Processing Section */}
                  {uploadedImage && selectedDocumentType && !isDummyData && (
                    <div className="animate-in slide-in-from-bottom duration-500 space-y-4">
                      <div className="h-px bg-border" />
                      <div className="bg-accent backdrop-blur-sm px-4 py-2 rounded-xl border border-border inline-block">
                        <h3 className="text-base lg:text-lg text-accent-foreground">OCR 처리</h3>
                      </div>
                      <OCRProcessor
                        imageFile={imageFile}
                        documentType={selectedDocumentType}
                        onProcessComplete={handleOCRComplete}
                        onError={handleError}
                      />
                    </div>
                  )}

                  {/* Example OCR Info Section */}
                  {showExampleOCR && isDummyData && selectedDocumentType && dummyDataInfo && currentDummyId && (
                    <div className="animate-in slide-in-from-bottom duration-500 space-y-4">
                      <div className="h-px bg-border" />
                      <div className="bg-gradient-to-r from-emerald-50 to-blue-50 backdrop-blur-sm border border-emerald-200 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-emerald-100 rounded-lg">
                              <TestTube className="w-4 h-4 text-emerald-600" />
                            </div>
                            <div>
                              <h3 className="text-sm font-medium text-emerald-800">OCR 결과 예시</h3>
                              <p className="text-xs text-emerald-600">{dummyDataInfo.name} • {dummyDataInfo.category}</p>
                            </div>
                          </div>
                          <div className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full">
                            샘플 ID: {currentDummyId}
                          </div>
                        </div>
                        <div className="space-y-2">
                          <p className="text-xs text-emerald-700">
                            실제 한국 금융문서에서 추출된 OCR 결과 예시를 표시합니다.
                          </p>
                          <p className="text-xs text-emerald-600">
                            💡 다른 예시를 보려면 위의 "OCR 결과 예시 보기" 메뉴에서 다른 문서를 선택하세요.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Panel - OCR Results */}
            <div className="space-y-6">
              <div className="bg-card backdrop-blur-md border border-border rounded-3xl overflow-hidden">
                <div className="bg-muted/50 px-4 lg:px-6 py-4 border-b border-border">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-accent backdrop-blur-sm rounded-xl border border-border">
                      <Zap className="w-5 h-5 lg:w-6 lg:h-6 text-primary" />
                    </div>
                    <h2 className="text-lg lg:text-xl text-foreground">OCR 결과</h2>
                  </div>
                </div>
                
                <div className="p-4 lg:p-6">
                  {!processedData ? (
                    <div className="flex items-center justify-center min-h-[300px] lg:min-h-[400px]">
                      <div className="text-center space-y-4">
                        <div className="bg-muted backdrop-blur-sm w-16 h-16 lg:w-20 lg:h-20 rounded-2xl flex items-center justify-center mx-auto border border-border">
                          <FileText className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />
                        </div>
                        <div className="bg-card backdrop-blur-sm px-4 lg:px-6 py-3 rounded-2xl border border-border inline-block">
                          <p className="text-muted-foreground text-base lg:text-lg">
                            FinSight가 문서를 분석한 후 결과가 여기에 표시됩니다
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4 animate-in slide-in-from-right duration-500">
                      {/* Table Data Section */}
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-primary rounded-full"></div>
                          <h3 className="text-sm text-foreground">구조화된 데이터</h3>
                        </div>
                        <div className="overflow-x-auto">
                          <GenericTable
                            data={processedData}
                            title="한국 금융 문서 데이터"
                            subtitle="자동으로 추출된 개인정보 및 계좌정보"
                            onDataChange={handleDataChange}
                          />
                        </div>
                      </div>

                      {/* OCR Metrics Section */}
                      {ocrMetrics && (
                        <div className="space-y-3">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-primary rounded-full"></div>
                            <h3 className="text-sm text-foreground">OCR 정확도</h3>
                          </div>
                          <OCRMetricsDisplay metrics={ocrMetrics} />
                        </div>
                      )}

                      {/* Original Text Section */}
                      {originalText && (
                        <div className="space-y-3">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-primary rounded-full"></div>
                            <h3 className="text-sm text-foreground">추출된 텍스트</h3>
                          </div>
                          <div className="bg-muted/50 backdrop-blur-sm rounded-xl border border-border p-3 max-h-40 overflow-y-auto">
                            <pre className="text-muted-foreground text-xs leading-relaxed whitespace-pre-wrap font-mono">{originalText}</pre>
                          </div>
                        </div>
                      )}

                      {/* Export Section */}
                      <div className="space-y-3">
                        <div className="h-px bg-border" />
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-primary rounded-full"></div>
                          <h3 className="text-sm text-foreground">다운로드</h3>
                        </div>
                        <ResultExporter
                          data={processedData}
                          tableType="korean_financial"
                          title="FinSight 한국 금융문서 추출 데이터"
                          originalText={originalText}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}