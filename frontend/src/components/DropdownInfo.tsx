'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Zap, Info } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';

export default function DropdownInfo() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="bg-card backdrop-blur-md px-6 lg:px-8 py-3 lg:py-4 rounded-2xl border border-border max-w-3xl mx-auto">
        <div className="flex items-center justify-between">
          <p className="text-muted-foreground text-base lg:text-lg">
            AI 기반 금융 문서 OCR 솔루션 - 기재사항, 위임장, 신청서 등 한국 금융 양식을 정확하게 인식하고 구조화합니다.
          </p>
          <CollapsibleTrigger asChild>
            <Button 
              variant="ghost" 
              size="sm" 
              className="ml-4 p-2 hover:bg-accent"
            >
              {isOpen ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </Button>
          </CollapsibleTrigger>
        </div>
        
        <CollapsibleContent className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 지원하는 금융 문서 */}
            <div className="bg-muted/50 backdrop-blur-sm rounded-2xl p-6 border border-border">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-primary rounded-xl">
                  <FileText className="w-4 h-4 text-primary-foreground" />
                </div>
                <h3 className="text-lg text-foreground">
                  지원하는 금융 문서
                </h3>
                <Badge variant="outline" className="bg-background">
                  13종
                </Badge>
              </div>
              
              <div className="grid grid-cols-1 gap-4">
                {/* 은행/금융 */}
                <div className="space-y-3">
                  <h4 className="text-sm text-primary border-b border-border pb-1">
                    은행/금융 (5종)
                  </h4>
                  <div className="grid grid-cols-1 gap-2 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>기재사항 양식</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>실명확인위임장</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>자동이체신청서</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>계좌개설신청서</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>대출계약서</span>
                    </div>
                  </div>
                </div>

                {/* 보험 */}
                <div className="space-y-3">
                  <h4 className="text-sm text-primary border-b border-border pb-1">
                    보험 (5종)
                  </h4>
                  <div className="grid grid-cols-1 gap-2 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>보험계약확인서</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>보험금청구서</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>보험계약 승계동의서</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>보험 위임장</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>도난(파손)사실 확인서</span>
                    </div>
                  </div>
                </div>

                {/* 증권/투자 & 법무/합의 */}
                <div className="grid grid-cols-1 gap-3">
                  <div className="space-y-3">
                    <h4 className="text-sm text-primary border-b border-border pb-1">
                      증권/투자 (2종)
                    </h4>
                    <div className="grid grid-cols-1 gap-2 text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                        <span>증권취득신고서</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                        <span>고객거래확인서</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h4 className="text-sm text-primary border-b border-border pb-1">
                      법무/합의 (1종)
                    </h4>
                    <div className="text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                        <span>합의서</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* 분석 기능 */}
            <div className="bg-muted/50 backdrop-blur-sm rounded-2xl p-6 border border-border">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-primary rounded-xl">
                  <Zap className="w-4 h-4 text-primary-foreground" />
                </div>
                <h3 className="text-lg text-foreground">
                  AI 분석 기능
                </h3>
                <Badge variant="outline" className="bg-background">
                  4가지
                </Badge>
              </div>
              
              <div className="space-y-4">
                {/* 핵심 기능 */}
                <div className="space-y-3">
                  <h4 className="text-sm text-primary border-b border-border pb-1">
                    핵심 OCR 기능
                  </h4>
                  <div className="grid grid-cols-1 gap-2 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>개인정보 자동 인식</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>F1 Score 정확도 측정</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>실시간 데이터 편집</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
                      <span>JSON/CSV 내보내기</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 하단 정보 */}
          <div className="bg-muted/50 backdrop-blur-sm rounded-2xl p-4 border border-border">
            <div className="flex items-center gap-2 justify-center text-muted-foreground">
              <Info className="w-4 h-4" />
              <span className="text-sm">
                FinSight는 한국 금융기관의 다양한 문서 양식을 정확하게 인식하고 구조화된 데이터로 변환합니다
              </span>
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}