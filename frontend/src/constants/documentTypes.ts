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
  FileCheck
} from 'lucide-react';
import { DocumentCategory } from '../types';

export const documentCategories: DocumentCategory[] = [
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