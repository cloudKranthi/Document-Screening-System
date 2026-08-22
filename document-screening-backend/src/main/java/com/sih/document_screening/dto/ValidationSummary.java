package com.sih.document_screening.dto;

import java.time.LocalDate;

public record ValidationSummary(boolean isValid,
        boolean isNameConsistent,
        boolean isDobConsistent,
        boolean isNationalityConsistent,
        boolean isDocumentExpired) {
    
}
