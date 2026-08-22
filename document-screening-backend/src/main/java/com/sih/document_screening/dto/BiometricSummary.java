package com.sih.document_screening.dto;

public record BiometricSummary(
    double faceSimilarityScore,
        boolean isMatch
) {
    
}
