import { ScreeningInitResponse, VisaType } from '../types/screening';

export const MOCK_SUCCESS_RESPONSE: ScreeningInitResponse = {
  screeningId: 'c8f43a91-4e78-43d2-9721-a1b7e6d0812f',
  status: 'VERIFIED',
  extractedData: {
    // Passport Details
    passportNumber: 'Z8942104',
    passportName: 'ADITYA RAJ SHARMA',
    passportDob: '1995-06-14',
    passportExpiry: '2032-11-20',
    passportNationality: 'IND',
    mrzChecksumValid: true,

    // Visa Details
    visaNumber: 'VS-2024-98421',
    visaValidUntil: '2027-08-15',
    visaType: VisaType.TOURIST,

    // National ID Details
    nationalIdNumber: 'XXXX-XXXX-8921',
    nationalIdName: 'ADITYA RAJ SHARMA'
  },
  validationSummary: {
    overallRiskScore: 4,
    isFlagged: false,
    nameMatchConfidence: 99.2,
    expiryValid: true,
    mrzValid: true,
    flags: [],
    checks: [
      {
        id: 'chk-1',
        name: 'MRZ Checksum Integrity',
        category: 'PASSPORT',
        status: 'PASSED',
        description: 'Machine Readable Zone 2-line checksum verified against ICAO 9303 standard.',
        details: 'Passport line 1 & 2 hashes match data fields accurately.'
      },
      {
        id: 'chk-2',
        name: 'Passport Validity Period',
        category: 'PASSPORT',
        status: 'PASSED',
        description: 'Passport has more than 6 months of validity remaining (Expires 2032-11-20).',
        details: 'Valid for 6+ years.'
      },
      {
        id: 'chk-3',
        name: 'Visa Validity Status',
        category: 'VISA',
        status: 'PASSED',
        description: 'Visa is active and valid until 2027-08-15.',
        details: 'Type: TOURIST (Multiple Entry eligible).'
      },
      {
        id: 'chk-4',
        name: 'Cross-Document Identity Match',
        category: 'CROSS_MATCH',
        status: 'PASSED',
        description: 'Exact string match between Passport Name and National ID holder name.',
        details: 'Confidence Score: 99.2%'
      },
      {
        id: 'chk-5',
        name: 'Security & Watchlist Check',
        category: 'SECURITY',
        status: 'PASSED',
        description: 'No active flags in Interpol or national immigration database.',
        details: 'Clearance Code: SEC-CLR-IND-8812'
      }
    ],
    recommendedAction: 'AUTO_APPROVE',
    notes: 'All documents verified successfully with high OCR confidence. Ready for automated clearance.'
  }
};

export const MOCK_NAME_MISMATCH_RESPONSE: ScreeningInitResponse = {
  screeningId: 'e2b779a4-51c3-4d0f-b258-3d5f8a0021c9',
  status: 'FLAGGED',
  extractedData: {
    // Passport Details
    passportNumber: 'M4412093',
    passportName: 'MICHAEL JONATHAN REED',
    passportDob: '1988-03-22',
    passportExpiry: '2029-05-10',
    passportNationality: 'GBR',
    mrzChecksumValid: true,

    // Visa Details
    visaNumber: 'GB-VS-88910',
    visaValidUntil: '2026-12-31',
    visaType: VisaType.BUSINESS,

    // National ID Details
    nationalIdNumber: 'NI-881920-A',
    nationalIdName: 'MIKE J. REED'
  },
  validationSummary: {
    overallRiskScore: 48,
    isFlagged: true,
    nameMatchConfidence: 68.4,
    expiryValid: true,
    mrzValid: true,
    flags: [
      'NAME_PARTIAL_MISMATCH: Passport ("MICHAEL JONATHAN REED") vs National ID ("MIKE J. REED")',
      'MANUAL_VERIFICATION_REQUIRED'
    ],
    checks: [
      {
        id: 'chk-1',
        name: 'MRZ Checksum Integrity',
        category: 'PASSPORT',
        status: 'PASSED',
        description: 'ICAO Doc 9303 checksum passed.',
        details: 'MRZ checksum digits match data fields.'
      },
      {
        id: 'chk-2',
        name: 'Cross-Document Name Verification',
        category: 'CROSS_MATCH',
        status: 'WARNING',
        description: 'Significant variation between official passport name and national ID card name.',
        details: 'Similarity score 68.4%. Potential nickname or abbreviated middle name.'
      },
      {
        id: 'chk-3',
        name: 'Visa Validity Check',
        category: 'VISA',
        status: 'PASSED',
        description: 'Visa valid until 2026-12-31 (Business category).',
        details: 'Active visa status.'
      }
    ],
    recommendedAction: 'MANUAL_REVIEW_REQUIRED',
    notes: 'Please review supporting name-change affidavit or secondary identity document.'
  }
};

export const MOCK_EXPIRED_VISA_RESPONSE: ScreeningInitResponse = {
  screeningId: 'f9104b2c-671e-450a-8e2b-7c19a4e32d18',
  status: 'REJECTED',
  extractedData: {
    // Passport Details
    passportNumber: 'K7721904',
    passportName: 'ELENA VASILIEVA',
    passportDob: '1992-09-08',
    passportExpiry: '2028-04-19',
    passportNationality: 'RUS',
    mrzChecksumValid: true,

    // Visa Details
    visaNumber: 'EV-884102',
    visaValidUntil: '2024-01-15', // Expired!
    visaType: VisaType.STUDENT,

    // National ID Details
    nationalIdNumber: 'RU-7721-0982',
    nationalIdName: 'ELENA VASILIEVA'
  },
  validationSummary: {
    overallRiskScore: 85,
    isFlagged: true,
    nameMatchConfidence: 98.5,
    expiryValid: false,
    mrzValid: true,
    flags: [
      'VISA_EXPIRED: Document expired on 2024-01-15',
      'UNAUTHORIZED_ENTRY_RISK'
    ],
    checks: [
      {
        id: 'chk-1',
        name: 'Visa Expiration Check',
        category: 'VISA',
        status: 'FAILED',
        description: 'Visa expired on 2024-01-15.',
        details: 'Cannot authorize entry or screening approval with lapsed visa.'
      },
      {
        id: 'chk-2',
        name: 'Passport Validity Period',
        category: 'PASSPORT',
        status: 'PASSED',
        description: 'Passport is valid until 2028-04-19.',
        details: 'Passport validity intact.'
      }
    ],
    recommendedAction: 'REJECT_APPLICATION',
    notes: 'Immediate rejection due to invalid/expired travel visa authorization.'
  }
};

export const MOCK_INVALID_MRZ_RESPONSE: ScreeningInitResponse = {
  screeningId: 'a38d9271-912c-4f81-9b12-58e1c7429110',
  status: 'FLAGGED',
  extractedData: {
    // Passport Details
    passportNumber: 'P8921443',
    passportName: 'CARLOS MENDEZ SILVA',
    passportDob: '1984-12-05',
    passportExpiry: '2030-07-25',
    passportNationality: 'ESP',
    mrzChecksumValid: false, // Fraud / Tampering flag!

    // Visa Details
    visaNumber: 'ESP-V-9921',
    visaValidUntil: '2026-10-30',
    visaType: VisaType.WORK,

    // National ID Details
    nationalIdNumber: 'DNI-44810294-K',
    nationalIdName: 'CARLOS MENDEZ SILVA'
  },
  validationSummary: {
    overallRiskScore: 92,
    isFlagged: true,
    nameMatchConfidence: 96.0,
    expiryValid: true,
    mrzValid: false,
    flags: [
      'MRZ_CHECKSUM_FAILED: ICAO standard checksum mismatch on line 2 (digit 14 & 21)',
      'SUSPECTED_DOCUMENT_FORGERY_OR_DAMAGED_SCAN'
    ],
    checks: [
      {
        id: 'chk-1',
        name: 'MRZ Checksum Integrity',
        category: 'PASSPORT',
        status: 'FAILED',
        description: 'Calculated checksum does not equal scanned MRZ check digits.',
        details: 'Potential security tamper or degraded OCR quality on physical passport.'
      },
      {
        id: 'chk-2',
        name: 'Security Watchlist Evaluation',
        category: 'SECURITY',
        status: 'WARNING',
        description: 'High fraud risk score triggered by cryptographic MRZ discrepancy.',
        details: 'Escalated to Tier-2 forensic document investigator.'
      }
    ],
    recommendedAction: 'MANUAL_REVIEW_REQUIRED',
    notes: 'Document must be subjected to physical UV/hologram inspection before any clearance.'
  }
};

export const SCENARIO_LIST = [
  { id: 'valid', label: '1. All Documents Valid (Auto Approve)', data: MOCK_SUCCESS_RESPONSE },
  { id: 'mismatch', label: '2. Name Mismatch (Review Required)', data: MOCK_NAME_MISMATCH_RESPONSE },
  { id: 'expired', label: '3. Expired Visa (Reject)', data: MOCK_EXPIRED_VISA_RESPONSE },
  { id: 'mrz_fail', label: '4. Invalid MRZ / Tamper Alert', data: MOCK_INVALID_MRZ_RESPONSE }
];
