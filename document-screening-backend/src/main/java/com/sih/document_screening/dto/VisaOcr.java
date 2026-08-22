package com.sih.document_screening.dto;

import java.time.LocalDate;

public record VisaOcr(
    String visaNumber,
        LocalDate validUntil,
        String visaType,
        double confidence
) {
    
}
