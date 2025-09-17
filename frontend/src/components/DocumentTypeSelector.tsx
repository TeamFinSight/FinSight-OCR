'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { FileCheck, Loader2, Search, X } from 'lucide-react';
import { apiClient } from '../services/api';
import { DocumentCategory } from '../types';
import {
  Building,
  Shield,
  TrendingUp,
  Scale,
  FileText,
  Users,
  CreditCard,
  Banknote,
  UserCheck,
  HandHeart,
  AlertTriangle,
  PiggyBank,
  Receipt,
  Coins,
  Calculator
} from 'lucide-react';

interface DocumentTypeSelectorProps {
  selectedDocumentType: string | null;
  onDocumentTypeSelect: (documentType: string) => void;
}

// 아이콘 매핑
const iconMap: Record<string, any> = {
  Building,
  Shield,
  TrendingUp,
  Scale,
  FileText,
  Users,
  CreditCard,
  Banknote,
  UserCheck,
  HandHeart,
  AlertTriangle,
  FileCheck,
  PiggyBank,
  Receipt,
  Coins,
  Calculator
};

export default function DocumentTypeSelector({ selectedDocumentType, onDocumentTypeSelect }: DocumentTypeSelectorProps) {
  const [documentCategories, setDocumentCategories] = useState<DocumentCategory[]>([]);
  const [filteredCategories, setFilteredCategories] = useState<DocumentCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    const loadDocumentTypes = async () => {
      try {
        setLoading(true);
        const response = await apiClient.getDocumentTypes();

        if (!response.success || !response.data) {
          throw new Error(response.error || '문서 종류를 가져올 수 없습니다.');
        }

        // 백엔드 데이터를 프론트엔드 형식으로 변환 (아이콘 매핑)
        const categories: DocumentCategory[] = response.data.categories.map((category: any) => ({
          ...category,
          icon: iconMap[category.icon] || FileText,
          documents: category.documents.map((doc: any) => ({
            ...doc,
            icon: iconMap[doc.icon] || FileText
          }))
        }));

        setDocumentCategories(categories);
        setFilteredCategories(categories);
        if (categories.length > 0) {
          setActiveTab(categories[0].id);
        }
        setError(null);
      } catch (err) {
        // 백엔드 연결 실패 시 fallback 데이터 사용
        console.warn('백엔드 연결 실패, fallback 데이터 사용:', err);

        // document_types.json과 동일한 fallback 데이터
        const fallbackCategories: DocumentCategory[] = [
          {
            id: 'banking',
            name: '은행/금융',
            icon: Building,
            color: 'from-blue-500 to-indigo-500',
            documents: [
              {
                id: 'basic_form',
                name: '기재사항 양식',
                category: '은행/금융',
                icon: FileText,
                description: '기본 신상정보 및 연락처 기재 양식'
              },
              {
                id: 'identity_delegation',
                name: '실명확인위임장',
                category: '은행/금융',
                icon: UserCheck,
                description: '실명확인 업무 위임 관련 서류'
              },
              {
                id: 'auto_transfer',
                name: '자동이체신청서',
                category: '은행/금융',
                icon: CreditCard,
                description: '계좌 자동이체 신청 양식'
              },
              {
                id: 'account_opening',
                name: '계좌개설신청서',
                category: '은행/금융',
                icon: Banknote,
                description: '은행 계좌 개설 신청 양식'
              },
              {
                id: 'loan_contract',
                name: '대출계약서',
                category: '은행/금융',
                icon: HandHeart,
                description: '대출 계약 관련 서류'
              }
            ]
          },
          {
            id: 'insurance',
            name: '보험',
            icon: Shield,
            color: 'from-emerald-500 to-teal-500',
            documents: [
              {
                id: 'insurance_contract',
                name: '보험계약확인서',
                category: '보험',
                icon: Shield,
                description: '보험계약 내용 확인 서류'
              },
              {
                id: 'insurance_claim',
                name: '보험금청구서',
                category: '보험',
                icon: FileCheck,
                description: '보험금 청구 관련 서류'
              },
              {
                id: 'insurance_succession',
                name: '보험계약 승계동의서',
                category: '보험',
                icon: Users,
                description: '보험계약 승계 동의 관련 서류'
              },
              {
                id: 'insurance_delegation',
                name: '보험 위임장',
                category: '보험',
                icon: UserCheck,
                description: '보험업무 위임 관련 서류'
              },
              {
                id: 'theft_damage_report',
                name: '도난(파손)사실 확인서',
                category: '보험',
                icon: AlertTriangle,
                description: '도난 및 파손사실 확인 서류'
              }
            ]
          },
          {
            id: 'securities',
            name: '증권/투자',
            icon: TrendingUp,
            color: 'from-purple-500 to-pink-500',
            documents: [
              {
                id: 'securities_acquisition',
                name: '증권취득신고서',
                category: '증권/투자',
                icon: TrendingUp,
                description: '증권 취득 신고 관련 서류'
              },
              {
                id: 'customer_verification',
                name: '고객거래확인서',
                category: '증권/투자',
                icon: UserCheck,
                description: '고객 신원확인 및 거래목적 확인서'
              }
            ]
          },
          {
            id: 'legal',
            name: '법무/합의',
            icon: Scale,
            color: 'from-orange-500 to-red-500',
            documents: [
              {
                id: 'agreement',
                name: '합의서',
                category: '법무/합의',
                icon: Scale,
                description: '법적 합의사항 관련 서류'
              }
            ]
          }
        ];

        setDocumentCategories(fallbackCategories);
        setFilteredCategories(fallbackCategories);
        if (fallbackCategories.length > 0) {
          setActiveTab(fallbackCategories[0].id);
        }
        setError(null); // 에러 상태 해제
      } finally {
        setLoading(false);
      }
    };

    loadDocumentTypes();
  }, []);

  // 검색 필터링
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredCategories(documentCategories);
      return;
    }

    const filtered = documentCategories.map(category => ({
      ...category,
      documents: category.documents.filter(doc =>
        doc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        doc.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        doc.category.toLowerCase().includes(searchTerm.toLowerCase())
      )
    })).filter(category => category.documents.length > 0);

    setFilteredCategories(filtered);

    // 검색 결과가 있으면 첫 번째 카테고리로 이동
    if (filtered.length > 0) {
      setActiveTab(filtered[0].id);
    }
  }, [searchTerm, documentCategories]);

  const handleClearSearch = () => {
    setSearchTerm('');
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="text-center space-y-2">
          <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20 inline-block">
            <h3 className="bg-gradient-to-r from-slate-700 to-slate-500 bg-clip-text text-transparent">문서 종류 선택</h3>
          </div>
          <p className="text-slate-600 text-xs">분석할 금융 문서의 종류를 선택해주세요</p>
        </div>
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center gap-2 text-slate-600">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">문서 종류를 불러오는 중...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div className="text-center space-y-2">
          <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20 inline-block">
            <h3 className="bg-gradient-to-r from-slate-700 to-slate-500 bg-clip-text text-transparent">문서 종류 선택</h3>
          </div>
          <p className="text-slate-600 text-xs">분석할 금융 문서의 종류를 선택해주세요</p>
        </div>
        <div className="bg-red-50/50 border border-red-200/50 rounded-lg p-3 text-center">
          <p className="text-red-700 text-xs">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="text-red-700 text-xs underline mt-1"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center space-y-3">
        <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20 inline-block">
          <h3 className="bg-gradient-to-r from-slate-700 to-slate-500 bg-clip-text text-transparent">문서 종류 선택</h3>
        </div>
        <p className="text-slate-600 text-xs">분석할 금융 문서의 종류를 선택해주세요</p>

        {/* 검색 바 */}
        <div className="max-w-md mx-auto">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="문서명 또는 설명으로 검색..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-10 py-2 text-sm bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-400/50 focus:border-indigo-400/50"
            />
            {searchTerm && (
              <button
                onClick={handleClearSearch}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        {/* Category Tabs */}
        <TabsList className="grid w-full bg-white/10 backdrop-blur-sm border border-white/20 p-1" style={{gridTemplateColumns: `repeat(${filteredCategories.length}, 1fr)`}}>
          {filteredCategories.map((category) => {
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
        {filteredCategories.map((category) => (
          <TabsContent key={category.id} value={category.id} className="mt-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {category.documents.map((document) => {
                const DocumentIcon = document.icon;
                const isSelected = selectedDocumentType === document.id;

                return (
                  <Card
                    key={document.id}
                    className={`cursor-pointer transition-all duration-200 hover:shadow-md hover:scale-105 ${
                      isSelected
                        ? 'bg-white/20 border-indigo-400/50 shadow-md ring-1 ring-indigo-400/30 scale-105'
                        : 'bg-white/5 border-white/20 hover:bg-white/10'
                    }`}
                    onClick={() => onDocumentTypeSelect(document.id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex flex-col items-center text-center space-y-3">
                        <div className={`p-3 rounded-lg transition-all duration-200 ${
                          isSelected
                            ? `bg-gradient-to-r ${category.color} shadow-lg`
                            : 'bg-white/10 hover:bg-white/20'
                        }`}>
                          <DocumentIcon className={`w-5 h-5 ${
                            isSelected ? 'text-white' : 'text-slate-600'
                          }`} />
                        </div>

                        <div className="space-y-2 min-h-[60px] flex flex-col justify-between">
                          <div className="flex items-center justify-center gap-1">
                            <h5 className={`text-sm font-medium leading-tight text-center ${
                              isSelected
                                ? 'text-indigo-700'
                                : 'text-slate-700'
                            }`}>
                              {document.name}
                            </h5>
                            {isSelected && (
                              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse flex-shrink-0 ml-1" />
                            )}
                          </div>
                          <p className="text-[11px] text-slate-500 leading-relaxed line-clamp-2 px-1">
                            {document.description}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            {/* 카테고리별 요약 정보 */}
            <div className="mt-4 p-3 bg-white/5 rounded-lg border border-white/10">
              <div className="flex items-center gap-2 text-slate-600 text-xs">
                <div className={`p-1 rounded bg-gradient-to-r ${category.color}`}>
                  <div className="w-2 h-2"></div>
                </div>
                <span className="font-medium">{category.name}</span>
                <span>•</span>
                <span>{category.documents.length}개 문서 유형</span>
              </div>
            </div>
          </TabsContent>
        ))}
      </Tabs>

      {selectedDocumentType && (
        <div className="bg-indigo-50/50 backdrop-blur-sm border border-indigo-200/50 rounded-xl p-4 animate-in slide-in-from-bottom duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <FileCheck className="w-4 h-4 text-indigo-600" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-indigo-800">선택 완료</span>
                  <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
                </div>
                <p className="text-xs text-indigo-600 mt-1">
                  {documentCategories
                    .flatMap(cat => cat.documents)
                    .find(doc => doc.id === selectedDocumentType)?.name}
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-indigo-600 font-medium">✓ 준비완료</div>
              <div className="text-[10px] text-indigo-500">OCR 처리 가능</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

