'use client';

import { useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { FileCheck } from 'lucide-react';
import { documentCategories } from '../constants/documentTypes';

interface DocumentTypeSelectorProps {
  selectedDocumentType: string | null;
  onDocumentTypeSelect: (documentType: string) => void;
}

export default function DocumentTypeSelector({ selectedDocumentType, onDocumentTypeSelect }: DocumentTypeSelectorProps) {
  const [activeTab, setActiveTab] = useState(documentCategories[0].id);

  return (
    <div className="space-y-4">
      <div className="text-center space-y-2">
        <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20 inline-block">
          <h3 className="bg-gradient-to-r from-slate-700 to-slate-500 bg-clip-text text-transparent">문서 종류 선택</h3>
        </div>
        <p className="text-slate-600 text-xs">분석할 금융 문서의 종류를 선택해주세요</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        {/* Category Tabs */}
        <TabsList className="grid w-full grid-cols-4 bg-white/10 backdrop-blur-sm border border-white/20 p-1">
          {documentCategories.map((category) => {
            const CategoryIcon = category.icon;
            return (
              <TabsTrigger
                key={category.id}
                value={category.id}
                className="flex items-center gap-2 data-[state=active]:bg-white/20 data-[state=active]:text-slate-700 text-slate-600 text-xs"
              >
                <CategoryIcon className="w-3 h-3" />
                <span className="hidden sm:inline">{category.name}</span>
                <Badge variant="outline" className="text-[10px] px-1 py-0 h-4">
                  {category.documents.length}
                </Badge>
              </TabsTrigger>
            );
          })}
        </TabsList>

        {/* Document Grid for each category */}
        {documentCategories.map((category) => (
          <TabsContent key={category.id} value={category.id} className="mt-4">
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
              {category.documents.map((document) => {
                const DocumentIcon = document.icon;
                const isSelected = selectedDocumentType === document.id;
                
                return (
                  <Card
                    key={document.id}
                    className={`cursor-pointer transition-all duration-200 hover:shadow-md ${
                      isSelected
                        ? 'bg-white/20 border-indigo-400/50 shadow-md ring-1 ring-indigo-400/30'
                        : 'bg-white/5 border-white/20 hover:bg-white/10'
                    }`}
                    onClick={() => onDocumentTypeSelect(document.id)}
                  >
                    <CardContent className="p-3">
                      <div className="flex flex-col items-center text-center space-y-2">
                        <div className={`p-2 rounded-lg ${
                          isSelected 
                            ? `bg-gradient-to-r ${category.color} shadow-lg` 
                            : 'bg-white/10'
                        }`}>
                          <DocumentIcon className={`w-4 h-4 ${
                            isSelected ? 'text-white' : 'text-slate-600'
                          }`} />
                        </div>
                        
                        <div className="space-y-1">
                          <div className="flex items-center justify-center gap-1">
                            <h5 className={`text-xs leading-tight ${
                              isSelected 
                                ? 'text-indigo-700' 
                                : 'text-slate-700'
                            }`}>
                              {document.name}
                            </h5>
                            {isSelected && (
                              <div className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse flex-shrink-0" />
                            )}
                          </div>
                          <p className="text-[10px] text-slate-500 leading-snug line-clamp-2">
                            {document.description}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
        ))}
      </Tabs>

      {selectedDocumentType && (
        <div className="bg-indigo-50/50 backdrop-blur-sm border border-indigo-200/50 rounded-xl p-3 animate-in slide-in-from-bottom duration-300">
          <div className="flex items-center gap-2 text-indigo-700">
            <FileCheck className="w-3 h-3" />
            <span className="text-xs">
              선택완료: {documentCategories
                .flatMap(cat => cat.documents)
                .find(doc => doc.id === selectedDocumentType)?.name}
            </span>
          </div>
          <p className="text-[10px] text-indigo-600 mt-1">
            이제 OCR 처리를 진행할 수 있습니다.
          </p>
        </div>
      )}
    </div>
  );
}

