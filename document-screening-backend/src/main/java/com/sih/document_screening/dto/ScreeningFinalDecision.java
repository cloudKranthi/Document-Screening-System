package com.sih.document_screening.dto;

import java.util.List;
import java.util.UUID;

public record ScreeningFinalDecision(
    UUID screeningId,
    String finalVerdict,       // ALLOW, MANUAL_REVIEW_REQUIRED, REJECT
    String riskCategory,       // LOW, MEDIUM, HIGH
    int riskScore,             // 0 to 100
    BiometricSummary biometrics,
    TamperingSummary tampering,
    List<String> flaggedReasons
) {
    
}
