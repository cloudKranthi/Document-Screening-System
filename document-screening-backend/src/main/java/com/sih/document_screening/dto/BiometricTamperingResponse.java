package com.sih.document_screening.dto;

import java.util.List;

public record BiometricTamperingResponse(
    double faceSimilarityScore,
    double photoTamperingScore,
    double textManipulationScore,
    boolean metadataAnomalyFound,
     List<String> detectedAnomalies
) {
    
}
