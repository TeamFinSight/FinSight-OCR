'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Plus, Trash2, Edit2, Save, X } from 'lucide-react';

import { TableData } from '../types';

interface GenericTableProps {
  data: TableData;
  title?: string;
  subtitle?: string;
  onDataChange?: (newData: TableData) => void;
}

export function GenericTable({ data, title = "추출된 표", subtitle, onDataChange }: GenericTableProps) {
  const [editingCell, setEditingCell] = useState<{ row: number; col: number } | null>(null);
  const [editValue, setEditValue] = useState('');
  const [tableData, setTableData] = useState<TableData>(data);

  const handleCellEdit = (rowIndex: number, colIndex: number) => {
    setEditingCell({ row: rowIndex, col: colIndex });
    setEditValue(tableData.rows[rowIndex][colIndex]);
  };

  const handleSaveEdit = () => {
    if (!editingCell) return;
    
    const newRows = [...tableData.rows];
    newRows[editingCell.row][editingCell.col] = editValue;
    
    const newData = { ...tableData, rows: newRows };
    setTableData(newData);
    onDataChange?.(newData);
    setEditingCell(null);
    setEditValue('');
  };

  const handleCancelEdit = () => {
    setEditingCell(null);
    setEditValue('');
  };

  const handleAddRow = () => {
    const newRow = new Array(tableData.headers.length).fill('');
    const newData = { ...tableData, rows: [...tableData.rows, newRow] };
    setTableData(newData);
    onDataChange?.(newData);
  };

  const handleDeleteRow = (rowIndex: number) => {
    const newRows = tableData.rows.filter((_, index) => index !== rowIndex);
    const newData = { ...tableData, rows: newRows };
    setTableData(newData);
    onDataChange?.(newData);
  };

  const handleAddColumn = () => {
    const newHeaders = [...tableData.headers, `열 ${tableData.headers.length + 1}`];
    const newRows = tableData.rows.map(row => [...row, '']);
    const newData = { headers: newHeaders, rows: newRows };
    setTableData(newData);
    onDataChange?.(newData);
  };

  const handleDeleteColumn = (colIndex: number) => {
    if (tableData.headers.length <= 1) return; // 최소 1개 열은 유지
    
    const newHeaders = tableData.headers.filter((_, index) => index !== colIndex);
    const newRows = tableData.rows.map(row => row.filter((_, index) => index !== colIndex));
    const newData = { headers: newHeaders, rows: newRows };
    setTableData(newData);
    onDataChange?.(newData);
  };

  const handleHeaderEdit = (colIndex: number, newValue: string) => {
    const newHeaders = [...tableData.headers];
    newHeaders[colIndex] = newValue;
    const newData = { ...tableData, headers: newHeaders };
    setTableData(newData);
    onDataChange?.(newData);
  };

  return (
    <Card className="bg-card backdrop-blur-md border border-border overflow-hidden">
      <CardHeader className="bg-muted/50 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-foreground text-base">
              {title}
            </CardTitle>
            {subtitle && (
              <p className="text-muted-foreground text-sm mt-1">{subtitle}</p>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              onClick={handleAddColumn}
              size="sm"
              variant="outline"
              className="bg-background hover:bg-accent"
            >
              <Plus className="w-4 h-4 mr-1" />
              열 추가
            </Button>
            <Button
              onClick={handleAddRow}
              size="sm"
              variant="outline"
              className="bg-background hover:bg-accent"
            >
              <Plus className="w-4 h-4 mr-1" />
              행 추가
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-muted/50">
                {tableData.headers.map((header, colIndex) => (
                  <TableHead key={colIndex} className="text-foreground bg-muted/30 relative group">
                    <div className="flex items-center justify-between">
                      <Input
                        value={header}
                        onChange={(e) => handleHeaderEdit(colIndex, e.target.value)}
                        className="bg-transparent border-none p-0 h-auto text-sm text-foreground"
                      />
                      {tableData.headers.length > 1 && (
                        <Button
                          onClick={() => handleDeleteColumn(colIndex)}
                          size="sm"
                          variant="ghost"
                          className="opacity-0 group-hover:opacity-100 transition-opacity ml-2 h-6 w-6 p-0 text-destructive hover:text-destructive hover:bg-destructive/10"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      )}
                    </div>
                  </TableHead>
                ))}
                <TableHead className="w-16 bg-muted/30"></TableHead>
              </TableRow>
            </TableHeader>
            
            <TableBody>
              {tableData.rows.map((row, rowIndex) => (
                <TableRow key={rowIndex} className="border-border hover:bg-muted/30 group">
                  {row.map((cell, colIndex) => (
                    <TableCell key={colIndex} className="relative">
                      {editingCell?.row === rowIndex && editingCell?.col === colIndex ? (
                        <div className="flex items-center gap-2">
                          <Input
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleSaveEdit();
                              if (e.key === 'Escape') handleCancelEdit();
                            }}
                            className="text-sm bg-input border-border"
                            autoFocus
                          />
                          <div className="flex gap-1">
                            <Button
                              onClick={handleSaveEdit}
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0 text-green-600 hover:text-green-800 hover:bg-green-600/10"
                            >
                              <Save className="w-3 h-3" />
                            </Button>
                            <Button
                              onClick={handleCancelEdit}
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground hover:bg-muted"
                            >
                              <X className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div 
                          className="cursor-pointer text-foreground text-sm p-1 rounded hover:bg-accent transition-colors flex items-center justify-between group"
                          onClick={() => handleCellEdit(rowIndex, colIndex)}
                        >
                          <span className="truncate max-w-[200px]" title={cell || '(빈 값)'}>
                            {cell?.trim() ? 
                              (cell.trim().length > 30 ? 
                                `${cell.trim().substring(0, 30)}...` : 
                                cell.trim()) 
                              : '(빈 값)'}
                          </span>
                          <Edit2 className="w-3 h-3 opacity-0 group-hover:opacity-50 transition-opacity flex-shrink-0 ml-1" />
                        </div>
                      )}
                    </TableCell>
                  ))}
                  <TableCell className="w-16">
                    <Button
                      onClick={() => handleDeleteRow(rowIndex)}
                      size="sm"
                      variant="ghost"
                      className="opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0 text-destructive hover:text-destructive hover:bg-destructive/10"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              
              {tableData.rows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={tableData.headers.length + 1} className="text-center text-muted-foreground py-8">
                    데이터가 없습니다. 행을 추가해보세요.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}