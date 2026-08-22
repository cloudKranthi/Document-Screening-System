package com.sih.document_screening.dto;

public record TamperingSummary(double photoTamperingScore,
        double textManipulationScore,
        boolean metadataAnomalyFound) {
    
}
