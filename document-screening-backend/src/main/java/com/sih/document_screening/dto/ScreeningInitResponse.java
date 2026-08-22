package com.sih.document_screening.dto;

import com.sih.document_screening.model.VisaType;

import java.time.LocalDate;
import java.util.UUID;

public record ScreeningInitResponse(
    UUID screeningId,
    String status,
    ExtractedDocumentData extractedData,
    ValidationSummary validationSummary
) {
    public record ExtractedDocumentData(
        // Passport Details
        String passportNumber,
        String passportName,
        LocalDate passportDob,
        LocalDate passportExpiry,
        String passportNationality,
        Boolean mrzChecksumValid,

        // Visa Details
        String visaNumber,
        LocalDate visaValidUntil,
        VisaType visaType,

        // National ID Details
        String nationalIdNumber,
        String nationalIdName
    ) {}
}