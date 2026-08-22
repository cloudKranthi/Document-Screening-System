package com.sih.document_screening.dto;

import java.time.LocalDate;
import java.util.List;

public record ValidationSummary(boolean isValid,
        boolean isNameConsistent,
        boolean isDobConsistent,
        boolean isNationalityConsistent,
        boolean isDocumentExpired,
        List<String> validationErrors
) {
    
}
