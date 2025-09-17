'use client';

import { useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Timer, Target, TrendingUp, CheckCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { OCRMetrics } from '../types';

interface OCRMetricsDisplayProps {
  metrics: OCRMetrics;
}

export function OCRMetricsDisplay({ metrics }: OCRMetricsDisplayProps) {
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);

  const getScoreColor = (score: number) => {
    const percentage = score * 100; // 0.98 -> 98%로 변환
    if (percentage >= 95) return 'text-emerald-600';
    if (percentage >= 90) return 'text-green-600';
    if (percentage >= 85) return 'text-yellow-600';
    if (percentage >= 80) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number) => {
    const percentage = score * 100; // 0.98 -> 98%로 변환
    if (percentage >= 95) return 'from-emerald-500/20 to-emerald-600/10';
    if (percentage >= 90) return 'from-green-500/20 to-green-600/10';
    if (percentage >= 85) return 'from-yellow-500/20 to-yellow-600/10';
    if (percentage >= 80) return 'from-orange-500/20 to-orange-600/10';
    return 'from-red-500/20 to-red-600/10';
  };

  const getGradeText = (score: number) => {
    const percentage = score * 100; // 0.98 -> 98%로 변환
    if (percentage >= 95) return 'Excellent';
    if (percentage >= 90) return 'Very Good';
    if (percentage >= 85) return 'Good';
    if (percentage >= 80) return 'Fair';
    return 'Poor';
  };

  return (
    <Card className="bg-white/5 backdrop-blur-md border-white/20 shadow-lg overflow-hidden">
      <CardContent className="p-3 space-y-3">
        {/* Main F1 Score Display */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-white/20 backdrop-blur-sm rounded-lg border border-white/20">
              <CheckCircle className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <div className="text-xs text-slate-600">F1 Score</div>
              <div className={`text-xl ${getScoreColor(metrics.f1Score)}`}>
                {(metrics.f1Score * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <Badge 
            variant="outline" 
            className={`bg-gradient-to-r ${getScoreBgColor(metrics.f1Score)} border-white/30 text-slate-700 text-xs px-2 py-0.5`}
          >
            {getGradeText(metrics.f1Score)}
          </Badge>
        </div>

        <Progress 
          value={metrics.f1Score * 100} 
          className="w-full h-1.5 bg-white/20" 
        />

        {/* Collapsible Detailed Metrics */}
        <Collapsible open={isDetailsOpen} onOpenChange={setIsDetailsOpen}>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-center gap-2 bg-white/5 backdrop-blur-sm px-2 py-1.5 rounded-lg border border-white/20 hover:bg-white/10 transition-colors text-xs text-slate-600">
              <span>세부 정보</span>
              {isDetailsOpen ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </div>
          </CollapsibleTrigger>
          
          <CollapsibleContent className="space-y-3 animate-in slide-in-from-top-2 duration-200">
            <div className="pt-1 space-y-3">
              {/* Detailed Metrics Grid */}
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 border border-white/20">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Target className="w-3 h-3 text-blue-600" />
                    <h4 className="text-xs text-slate-600">Precision</h4>
                  </div>
                  <div className={`text-sm ${getScoreColor(metrics.precision)}`}>
                    {(metrics.precision * 100).toFixed(1)}%
                  </div>
                  <Progress 
                    value={metrics.precision * 100} 
                    className="w-full h-1 bg-white/20 mt-1" 
                  />
                  <div className="text-xs text-slate-500 mt-0.5">추출 정확성</div>
                </div>

                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 border border-white/20">
                  <div className="flex items-center gap-1.5 mb-1">
                    <TrendingUp className="w-3 h-3 text-emerald-600" />
                    <h4 className="text-xs text-slate-600">Recall</h4>
                  </div>
                  <div className={`text-sm ${getScoreColor(metrics.recall)}`}>
                    {(metrics.recall * 100).toFixed(1)}%
                  </div>
                  <Progress 
                    value={metrics.recall * 100} 
                    className="w-full h-1 bg-white/20 mt-1" 
                  />
                  <div className="text-xs text-slate-500 mt-0.5">데이터 추출률</div>
                </div>

                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 border border-white/20">
                  <div className="flex items-center gap-1.5 mb-1">
                    <CheckCircle className="w-3 h-3 text-purple-600" />
                    <h4 className="text-xs text-slate-600">Confidence</h4>
                  </div>
                  <div className={`text-sm ${getScoreColor(metrics.confidence)}`}>
                    {(metrics.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-0.5">모델 신뢰도</div>
                </div>

                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 border border-white/20">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Timer className="w-3 h-3 text-indigo-600" />
                    <h4 className="text-xs text-slate-600">Time</h4>
                  </div>
                  <div className="text-sm text-slate-700">
                    {metrics.processingTime}s
                  </div>
                  <div className="text-xs text-slate-500 mt-0.5">처리 시간</div>
                </div>
              </div>

              {/* Score Interpretation */}
              <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 border border-white/20">
                <h4 className="text-xs text-slate-600 mb-1.5 flex items-center gap-1.5">
                  <TrendingUp className="w-3 h-3" />
                  분석 결과
                </h4>
                <div className="space-y-1 text-xs text-slate-600">
                  {(metrics.f1Score * 100) >= 95 && (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div>
                      <span>매우 높은 정확도로 금융 데이터가 추출되었습니다.</span>
                    </div>
                  )}
                  {(metrics.f1Score * 100) >= 90 && (metrics.f1Score * 100) < 95 && (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 bg-green-500 rounded-full"></div>
                      <span>높은 정확도로 대부분의 금융 데이터가 정확히 추출되었습니다.</span>
                    </div>
                  )}
                  {(metrics.f1Score * 100) >= 85 && (metrics.f1Score * 100) < 90 && (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 bg-yellow-500 rounded-full"></div>
                      <span>양호한 정확도입니다. 일부 데이터 검토가 필요할 수 있습니다.</span>
                    </div>
                  )}
                  {(metrics.f1Score * 100) < 85 && (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 bg-orange-500 rounded-full"></div>
                      <span>추출된 데이터를 신중히 검토하시기 바랍니다.</span>
                    </div>
                  )}
                  <div className="text-xs text-slate-500 leading-relaxed">
                    정확도가 높을수록 추출된 금융 데이터의 신뢰성이 높습니다.
                  </div>
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}