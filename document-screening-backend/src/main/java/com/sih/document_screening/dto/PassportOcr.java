package com.sih.document_screening.dto;

import java.time.LocalDate;

public record PassportOcr(String documentNumber,
        String fullName,
        LocalDate dateOfBirth,
        LocalDate expiryDate,
        String nationality,
        String rawMrz,
        boolean mrzValid,
        double confidence) {
    
}
