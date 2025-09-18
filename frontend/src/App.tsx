import { useState } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { OCRProcessor } from './components/OCRProcessor';
import { GenericTable } from './components/GenericTable';
import { TableData } from './types';
import { ResultExporter } from './components/ResultExporter';
import DocumentTypeSelector from './components/DocumentTypeSelector';
import DropdownInfo from './components/DropdownInfo';
import { Card, CardContent } from './components/ui/card';
import { FileText, Zap, AlertCircle, TestTube } from 'lucide-react';
import { useErrorHandler } from './hooks/useErrorHandler';
import { dummyDataMap } from './constants/dummyOCRData';

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [refinedData, setRefinedData] = useState<TableData | null>(null);
  const [rawData, setRawData] = useState<TableData | null>(null);
  const [originalText, setOriginalText] = useState<string>('');
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
    setRefinedData(null);
    setRawData(null);
    setOriginalText('');
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
    setRefinedData(null);
    setRawData(null);
    setOriginalText('');
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
    setRefinedData(null);
    setRawData(null);
    setOriginalText('');
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
    setRefinedData(null);
    setRawData(null);
    setOriginalText('');
    setSelectedDocumentType(null);
    setIsDummyData(false);
    setDummyDataInfo(null);
    setShowExampleOCR(false);
    setCurrentDummyId(null);
    clearError();

    // 잠시 후 새로운 더미 데이터로 설정 (애니메이션 효과를 위해)
    setTimeout(() => {
      setSelectedDocumentType(dummyData.documentTypeId);
      setRefinedData(dummyData.tableData);
      setRawData(dummyData.tableData); // 더미 데이터에서는 같은 데이터를 사용
      setOriginalText(dummyData.originalText);
      setIsDummyData(true);
      setDummyDataInfo({
        name: dummyData.name,
        category: dummyData.category
      });
      setShowExampleOCR(true);
      setCurrentDummyId(dummyId);
    }, 100);
  };

  const handleOCRComplete = (refinedTableData: TableData, rawTableData: TableData, text: string) => {
    setRefinedData(refinedTableData);
    setRawData(rawTableData);
    setOriginalText(text);
    setIsDummyData(false);
    setDummyDataInfo(null);
    setShowExampleOCR(false);
    setCurrentDummyId(null);
    clearError();
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
                        className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${
                          isSelected
                            ? 'bg-primary/10 text-primary border border-primary/20'
                            : 'hover:bg-muted/50 text-foreground'
                        }`}
                      >
                        <div className="flex flex-col gap-1">
                          <span className="font-medium">{dummy.name}</span>
                          <span className="text-xs text-muted-foreground">{dummy.category}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="max-w-2xl mx-auto">
            <p className="text-base lg:text-lg text-muted-foreground mb-4">
              한국 금융문서에 특화된 AI OCR 시스템으로 손글씨까지 정확하게 인식합니다
            </p>

            {/* Step-by-step guide */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-xl p-4 mb-6">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                <h3 className="text-sm font-medium text-blue-800">사용 방법</h3>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="text-center">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-medium mx-auto mb-1">1</div>
                  <p className="text-blue-700">이미지 업로드</p>
                </div>
                <div className="text-center">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-medium mx-auto mb-1">2</div>
                  <p className="text-blue-700">문서 종류 선택</p>
                </div>
                <div className="text-center">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-medium mx-auto mb-1">3</div>
                  <p className="text-blue-700">OCR 처리 시작</p>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
              <div className="flex items-center gap-2 bg-muted/50 px-3 py-1.5 rounded-full">
                <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                <span className="text-muted-foreground">한글 특화</span>
              </div>
              <div className="flex items-center gap-2 bg-muted/50 px-3 py-1.5 rounded-full">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span className="text-muted-foreground">손글씨 인식</span>
              </div>
              <div className="flex items-center gap-2 bg-muted/50 px-3 py-1.5 rounded-full">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-muted-foreground">표 자동 추출</span>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {errorState.hasError && (
          <div className="mb-6 max-w-2xl mx-auto">
            <div className="bg-destructive/10 backdrop-blur-sm rounded-xl p-4 border border-destructive/20 animate-in slide-in-from-top duration-300">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-destructive font-medium">오류 발생</p>
                  <p className="text-destructive/80 text-sm mt-1">{errorState.error}</p>
                </div>
                <button
                  onClick={clearError}
                  className="text-destructive/60 hover:text-destructive transition-colors text-sm"
                >
                  닫기
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Left Panel - Upload and Process */}
          <div className="space-y-6">
            <div className="bg-card backdrop-blur-md border border-border rounded-3xl overflow-hidden">
              <div className="bg-muted/50 px-4 lg:px-6 py-4 border-b border-border">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-accent backdrop-blur-sm rounded-xl border border-border">
                    <FileText className="w-5 h-5 lg:w-6 lg:h-6 text-primary" />
                  </div>
                  <h2 className="text-lg lg:text-xl text-foreground">문서 업로드</h2>
                </div>
              </div>

              <div className="p-4 lg:p-6 space-y-6">
                {/* Step 1: Image Upload */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-medium">1</div>
                    <h3 className="text-sm text-foreground font-medium">이미지 업로드</h3>
                  </div>
                  <ImageUpload
                    onImageUpload={handleImageUpload}
                    onClearImage={handleClearImage}
                    uploadedImage={uploadedImage}
                  />
                  {!uploadedImage && (
                    <div className="bg-muted/30 border border-dashed border-muted-foreground/30 rounded-lg p-3">
                      <p className="text-xs text-muted-foreground text-center">
                        📁 먼저 분석할 문서 이미지를 업로드해주세요
                      </p>
                    </div>
                  )}
                </div>

                {/* Step 2: Document Type Selection - Only show after image upload */}
                {uploadedImage && (
                  <div className="space-y-3 animate-in slide-in-from-bottom duration-300">
                    <div className="h-px bg-border" />
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-medium">2</div>
                      <h3 className="text-sm text-foreground font-medium">문서 종류 선택</h3>
                    </div>
                    <DocumentTypeSelector
                      selectedDocumentType={selectedDocumentType}
                      onDocumentTypeSelect={handleDocumentTypeSelect}
                    />
                    <DropdownInfo />
                    {!selectedDocumentType && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <p className="text-xs text-blue-700 text-center">
                          📋 업로드하신 문서의 종류를 선택해주세요
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Step 3: OCR Processing - Only show after document type selection */}
                {uploadedImage && selectedDocumentType && (
                  <div className="space-y-3 animate-in slide-in-from-bottom duration-300">
                    <div className="h-px bg-border" />
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-medium">3</div>
                      <h3 className="text-sm text-foreground font-medium">OCR 처리</h3>
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
                {!refinedData && !rawData ? (
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
                  <div className="space-y-6 animate-in slide-in-from-right duration-500">
                    {/* 정제된 데이터 표 */}
                    {refinedData && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                          <h3 className="text-sm text-foreground">정제된 데이터</h3>
                        </div>
                        <div className="overflow-x-auto">
                          <GenericTable
                            data={refinedData}
                            title="정제된 데이터"
                            subtitle="라벨별로 그룹핑하고 정렬된 OCR 결과"
                            onDataChange={(newData) => setRefinedData(newData)}
                          />
                        </div>
                      </div>
                    )}

                    {/* 원본 데이터 표 */}
                    {rawData && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <h3 className="text-sm text-foreground">원본 데이터</h3>
                        </div>
                        <div className="overflow-x-auto">
                          <GenericTable
                            data={rawData}
                            title="원본 데이터"
                            subtitle="OCR에서 직접 추출된 원본 필드 데이터"
                            onDataChange={(newData) => setRawData(newData)}
                          />
                        </div>
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
                          <pre className="text-sm text-foreground whitespace-pre-wrap font-mono">
                            {originalText}
                          </pre>
                        </div>
                      </div>
                    )}

                    {/* Export Section */}
                    {(refinedData || rawData) && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-primary rounded-full"></div>
                          <h3 className="text-sm text-foreground">내보내기</h3>
                        </div>

                        {/* 정제된 데이터 내보내기 */}
                        {refinedData && (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                              <h4 className="text-xs text-muted-foreground">정제된 데이터</h4>
                            </div>
                            <ResultExporter
                              data={refinedData}
                              tableType={selectedDocumentType || 'unknown'}
                              title={`${selectedDocumentType || 'OCR 결과'} (정제됨)`}
                              originalText={originalText}
                              documentType={selectedDocumentType}
                              sourceImageName={imageFile?.name || 'uploaded-image'}
                            />
                          </div>
                        )}

                        {/* 원본 데이터 내보내기 */}
                        {rawData && (
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                              <h4 className="text-xs text-muted-foreground">원본 데이터</h4>
                            </div>
                            <ResultExporter
                              data={rawData}
                              tableType={selectedDocumentType || 'unknown'}
                              title={`${selectedDocumentType || 'OCR 결과'} (원본)`}
                              originalText={originalText}
                              documentType={selectedDocumentType}
                              sourceImageName={imageFile?.name || 'uploaded-image'}
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}