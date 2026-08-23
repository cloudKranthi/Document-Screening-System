/**
 * Type definitions matching Document Screening & Biometric Verification API Guide
 * Endpoint: POST /api/v1/screenings/submit-and-validate
 * Endpoint: GET /api/v1/screenings/admin/flagged-queue
 */

export type ScreeningVerdict = 'ALLOW' | 'MANUAL_REVIEW_REQUIRED' | 'REJECT';

export type RiskCategory = 'LOW' | 'MEDIUM' | 'HIGH';

export type ScreeningStatus = 
  | 'PASSED'
  | 'SUSPICIOUS'
  | 'VALIDATION_FAILED'
  | 'FLAGGED_FOR_ADMIN'
  | 'IN_PROGRESS';

export interface ExtractedDocumentData {
  passportNumber: string | null;
  passportName: string | null;
  passportDob: string | null; // ISO Date String (YYYY-MM-DD)
  passportExpiry: string | null; // ISO Date String (YYYY-MM-DD)
  passportNationality: string | null;
  mrzChecksumValid: boolean | null;
  visaNumber: string | null;
  visaValidUntil: string | null;
  visaType: string | null;
  nationalIdNumber: string | null;
  nationalIdName: string | null;
}

export interface ValidationSummary {
  isValid: boolean;
  isNameConsistent: boolean;
  isDobConsistent: boolean;
  isNationalityConsistent: boolean;
  isDocumentExpired: boolean;
  validationErrors: string[];
}

export interface BiometricSummary {
  faceSimilarityScore: number; // 0.00 to 1.00
  isMatch: boolean;
}

export interface TamperingSummary {
  photoTamperingScore: number; // 0.00 to 1.00 (ELA Score)
  textManipulationScore: number; // 0.00 to 1.00
  metadataAnomalyFound: boolean;
}

/**
 * Full Unified Screening Response Payload (200 OK)
 */
export interface UnifiedScreeningResponse {
  screeningId: string; // UUID
  status: ScreeningStatus | string;
  verdict: ScreeningVerdict | string;
  riskCategory: RiskCategory | string;
  riskScore: number; // 0 to 100
  extractedData: ExtractedDocumentData;
  validationSummary: ValidationSummary;
  biometricSummary: BiometricSummary;
  tamperingSummary: TamperingSummary;
  flaggedReasons: string[];
}

/**
 * Form field inputs matching exact multipart keys
 */
export interface UnifiedScreeningRequest {
  passportFile: File;
  visaFile: File;
  nationalIdFile: File;
  livePhotoFile: File;
  nationality: string;
}

/**
 * Admin Flagged Queue Case DTO
 */
export interface AdminFlaggedQueueItem {
  screeningId: string;
  claimedNationality: string;
  status: string;
  finalRiskScore: number;
  riskCategory: string;
  flaggedReasons: string[];
  createdAt: string;
}
