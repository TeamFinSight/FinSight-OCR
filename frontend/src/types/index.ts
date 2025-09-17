export interface TableData {
  headers: string[];
  rows: string[][];
}


export interface DocumentType {
  id: string;
  name: string;
  category: string;
  icon: React.ComponentType<any>;
  description: string;
}

export interface DocumentCategory {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  color: string;
  documents: DocumentType[];
}

// FastAPI 연동을 위한 API 타입들
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface OCRRequest {
  image: File;
  document_type: string;
  options?: {
    language?: string;
    extract_tables?: boolean;
    extract_text?: boolean;
  };
}

export interface OCRResponse {
  extracted_text: string;
  table_data: TableData;
  processing_time: number;
  document_type: string;
  confidence_score: number;
}

export interface APIConfig {
  baseURL: string;
  timeout: number;
  retries: number;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

export interface ProcessingStatus {
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  message?: string;
  progress?: UploadProgress;
}