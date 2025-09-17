'use client';

import { Button } from './ui/button';
import { Download, FileText, Table } from 'lucide-react';
import { TableData } from '../types';

interface ResultExporterProps {
  data: TableData | any;
  tableType: string | null;
  title: string;
  originalText?: string;
}

export function ResultExporter({ data, tableType, title, originalText }: ResultExporterProps) {
  const exportToJSON = () => {
    if (!data) return;

    const exportData = {
      title,
      tableType,
      timestamp: new Date().toISOString(),
      data
    };

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
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4">
        {originalText && (
          <Button 
            onClick={exportToText} 
            variant="outline" 
            className="flex items-center gap-2 bg-white/10 backdrop-blur-sm border-white/20 hover:bg-white/20 transition-all duration-300 h-12"
          >
            <FileText className="w-4 h-4" />
            텍스트 파일 다운로드
          </Button>
        )}
        <Button 
          onClick={exportToJSON} 
          variant="outline" 
          className="flex items-center gap-2 bg-white/10 backdrop-blur-sm border-white/20 hover:bg-white/20 transition-all duration-300 h-12"
        >
          <Download className="w-4 h-4" />
          JSON 형식 다운로드
        </Button>
        <Button 
          onClick={exportToCSV} 
          variant="outline" 
          className="flex items-center gap-2 bg-white/10 backdrop-blur-sm border-white/20 hover:bg-white/20 transition-all duration-300 h-12"
        >
          <Table className="w-4 h-4" />
          CSV 형식 다운로드
        </Button>
      </div>
      
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/20 shadow-inner">
        <div className="bg-white/10 backdrop-blur-sm px-3 py-1 rounded-lg border border-white/20 inline-block mb-3">
          <h4 className="bg-gradient-to-r from-slate-700 to-slate-500 bg-clip-text text-transparent">데이터 정보</h4>
        </div>
        <div className="space-y-2 text-slate-600 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full"></div>
            <p>표 유형: {title}</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-full"></div>
            <p>데이터 행: {getDataCount()}개</p>
          </div>
          {getColumnCount() > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"></div>
              <p>컬럼 수: {getColumnCount()}개</p>
            </div>
          )}
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"></div>
            <p>처리 시간: {new Date().toLocaleString()}</p>
          </div>
        </div>
      </div>
    </div>
  );
}