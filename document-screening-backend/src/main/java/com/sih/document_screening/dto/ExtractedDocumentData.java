package com.sih.document_screening.dto;

import java.time.LocalDate;

public record ExtractedDocumentData(String passportNumber,
        String passportName,
        LocalDate passportDob,
        LocalDate passportExpiry,
        String passportNationality,
        Boolean mrzChecksumValid,
        String visaNumber,
        LocalDate visaValidUntil,
        String visaType,
        String nationalIdNumber,
        String nationalIdName) {
    
}
