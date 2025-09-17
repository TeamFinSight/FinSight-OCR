/**
 * 프론트엔드 문서 타입 ID와 백엔드 labelings.json의 type_name 매핑
 */
export const DOCUMENT_TYPE_MAPPING: Record<string, string> = {
  // 은행/금융 문서
  'basic_form': '제신고서',
  'identity_delegation': '실명확인_위임장',
  'auto_transfer': '자동이체신청서',
  'account_opening': '신청서',
  'loan_contract': '자동이체_승인서',
  'name_change_application': '명의변경 신청서',
  'virtual_account_purpose_confirmation': '기업인터넷뱅킹_가상계좌발급_목적확인서',

  // 보험 문서
  'insurance_contract': '위임장',
  'insurance_claim': '간병인 지원 서비스 신청서',
  'insurance_succession': '보험계약대출 승계 동의서',
  'insurance_delegation': '위임장',
  'theft_damage_report': '도난_파손_사실 확인서',
  'caregiver_service_application': '간병인 지원 서비스 신청서',

  // 직접 매칭 (ID와 type_name이 동일한 경우)
  '제신고서': '제신고서',
  '신청서': '신청서',
  '자동이체_승인서': '자동이체_승인서',
  '명의변경 신청서': '명의변경 신청서',
  '기업인터넷뱅킹_가상계좌발급_목적확인서': '기업인터넷뱅킹_가상계좌발급_목적확인서',
  '실명확인_위임장': '실명확인_위임장',
  '위임장': '위임장',
  '자동이체신청서': '자동이체신청서',
  '간병인 지원 서비스 신청서': '간병인 지원 서비스 신청서',
  '도난_파손_사실 확인서': '도난_파손_사실 확인서',
  '보험계약대출 승계 동의서': '보험계약대출 승계 동의서'
};

/**
 * 프론트엔드 문서 타입 ID를 백엔드 type_name으로 변환
 */
export function mapFrontendIdToBackendTypeName(frontendId: string): string {
  const backendTypeName = DOCUMENT_TYPE_MAPPING[frontendId];

  if (!backendTypeName) {
    console.warn(`문서 타입 매핑을 찾을 수 없습니다: ${frontendId}`);
    // 매핑을 찾을 수 없으면 원본 ID를 반환
    return frontendId;
  }

  return backendTypeName;
}

/**
 * 사용 가능한 백엔드 문서 타입 목록
 */
export const AVAILABLE_BACKEND_TYPES = [
  '제신고서',
  '신청서',
  '자동이체_승인서',
  '명의변경 신청서',
  '기업인터넷뱅킹_가상계좌발급_목적확인서',
  '실명확인_위임장',
  '위임장',
  '자동이체신청서',
  '간병인 지원 서비스 신청서',
  '도난_파손_사실 확인서',
  '보험계약대출 승계 동의서'
];