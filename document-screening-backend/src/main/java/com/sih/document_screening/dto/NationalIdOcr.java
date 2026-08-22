package com.sih.document_screening.dto;

public record NationalIdOcr(
    String idNumber,
        String fullName,
        double confidence
) {
    
}
