'use client';

import { Button } from './ui/button';
import { Download, FileText, Table } from 'lucide-react';
import { TableData } from '../types';

interface ResultExporterProps {
  data: TableData | any;
  tableType: string | null;
  title: string;
  originalText?: string;
  // 백엔드 JSON 스키마 생성을 위한 추가 정보
  documentType?: string | null;
  sourceImageName?: string | null;
}

export function ResultExporter({ data, tableType, title, originalText, documentType, sourceImageName }: ResultExporterProps) {
  const exportToJSON = () => {
    if (!data) return;

    // 이미 백엔드 스키마(metadata, document_info, fields)를 따르는 경우 그대로 저장
    const isBackendSchema = data && typeof data === 'object' && ('metadata' in data) && ('document_info' in data) && ('fields' in data);

    let exportData: any;
    if (isBackendSchema) {
      exportData = data;
    } else {
      // TableData 등 프론트 구조를 백엔드 스키마로 변환
      const headers: string[] = Array.isArray(data?.headers) ? data.headers : [];
      const rows: string[][] = Array.isArray(data?.rows) ? data.rows : [];

      const fields = rows.map((row: string[], idx: number) => {
        // 가장 흔한 2열(항목, 내용) 테이블을 가정하되, 유연하게 처리
        const label = headers.length >= 1 ? headers[0] : 'label';
        const rowLabel = row[0] ?? '';
        const rowText = row[1] ?? row.join(' | ');
        return {
          id: idx + 1,
          labels: rowLabel || label,
          rotation: 0,
          value_text: rowText || '',
          confidence: 0,
          value_box: {
            x: [],
            y: [],
            type: 'table'
          }
        };
      });

      exportData = {
        metadata: {
          source_image: sourceImageName || '',
          processed_at: new Date().toISOString(),
          total_detections: fields.length,
          model_info: {
            detection_model: '',
            recognition_model: ''
          }
        },
        document_info: {
          width: 0,
          height: 0,
          document_type: documentType || tableType || 'unknown'
        },
        fields
      };
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr-result-${tableType}-${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportToCSV = () => {
    if (!data) return;

    let csvContent = '';
    
    // 새로운 TableData 구조 처리
    if (data.headers && data.rows) {
      // 헤더 추가
      csvContent = data.headers.join(',') + '\n';
      
      // 데이터 행 추가
      data.rows.forEach((row: string[]) => {
        // CSV에서 쉼표나 개행문자가 포함된 필드는 따옴표로 감싸기
        const escapedRow = row.map(cell => {
          if (cell.includes(',') || cell.includes('\n') || cell.includes('"')) {
            return `"${cell.replace(/"/g, '""')}"`;
          }
          return cell;
        });
        csvContent += escapedRow.join(',') + '\n';
      });
    } else {
      // 기존 구조 호환성 유지 (레거시 데이터)
      if (Array.isArray(data) && data.length > 0) {
        const keys = Object.keys(data[0]);
        csvContent = keys.join(',') + '\n';
        
        data.forEach((row: any) => {
          const values = keys.map(key => {
            const value = String(row[key] || '');
            if (value.includes(',') || value.includes('\n') || value.includes('"')) {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value;
          });
          csvContent += values.join(',') + '\n';
        });
      }
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr-result-${tableType}-${new Date().getTime()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportToText = () => {
    if (!originalText) return;

    const blob = new Blob([originalText], { type: 'text/plain;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr-text-${tableType}-${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!data) {
    return null;
  }

  // 데이터 항목 수 계산
  const getDataCount = () => {
    if (data.headers && data.rows) {
      return data.rows.length;
    }
    if (Array.isArray(data)) {
      return data.length;
    }
    return Object.keys(data).length;
  };

  // 컬럼 수 계산
  const getColumnCount = () => {
    if (data.headers) {
      return data.headers.length;
    }
    if (Array.isArray(data) && data.length > 0) {
      return Object.keys(data[0]).length;
    }
    return 0;
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3">
        {originalText && (
          <Button
            onClick={exportToText}
            variant="outline"
            className="flex items-center gap-2 bg-muted/50 border-border hover:bg-muted/70 transition-all duration-200 h-10 text-sm"
          >
            <FileText className="w-4 h-4" />
            텍스트 파일 (.txt)
          </Button>
        )}
        <Button
          onClick={exportToJSON}
          variant="outline"
          className="flex items-center gap-2 bg-muted/50 border-border hover:bg-muted/70 transition-all duration-200 h-10 text-sm"
        >
          <Download className="w-4 h-4" />
          JSON 형식 (.json)
        </Button>
        <Button
          onClick={exportToCSV}
          variant="outline"
          className="flex items-center gap-2 bg-muted/50 border-border hover:bg-muted/70 transition-all duration-200 h-10 text-sm"
        >
          <Table className="w-4 h-4" />
          엑셀 형식 (.csv)
        </Button>
      </div>

      <div className="bg-muted/30 backdrop-blur-sm rounded-lg p-3 border border-border">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 bg-primary rounded-full"></div>
          <h4 className="text-xs font-medium text-foreground">데이터 정보</h4>
        </div>
        <div className="space-y-1 text-muted-foreground text-xs">
          <div className="flex items-center justify-between">
            <span>데이터 행:</span>
            <span className="font-medium">{getDataCount()}개</span>
          </div>
          {getColumnCount() > 0 && (
            <div className="flex items-center justify-between">
              <span>컬럼 수:</span>
              <span className="font-medium">{getColumnCount()}개</span>
            </div>
          )}
          <div className="flex items-center justify-between">
            <span>문서 종류:</span>
            <span className="font-medium">{title}</span>
          </div>
        </div>
      </div>
    </div>
  );
}