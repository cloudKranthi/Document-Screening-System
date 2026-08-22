package com.sih.document_screening.dto;

import com.sih.document_screening.model.VisaType;

import java.time.LocalDate;
import java.util.List;
import java.util.UUID;

public record ScreeningInitResponse(
    UUID screeningId,
    String status,
    String verdict,
    String riskCategory,
    Integer riskScore,
    ExtractedDocumentData extractedData,
    ValidationSummary validationSummary,
    BiometricSummary biometricSummary,
    TamperingSummary tamperingSummary,
    List<String> flaggedReasons
) {
    public record ExtractedDocumentData(
        String passportNumber,
        String passportName,
        LocalDate passportDob,
        LocalDate passportExpiry,
        String passportNationality,
        Boolean mrzChecksumValid,
        String visaNumber,
        LocalDate visaValidUntil,
        VisaType visaType,
        String nationalIdNumber,
        String nationalIdName
    ) {}
}