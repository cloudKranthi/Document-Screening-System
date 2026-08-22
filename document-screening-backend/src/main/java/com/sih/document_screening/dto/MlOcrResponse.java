package com.sih.document_screening.dto;

public record MlOcrResponse(PassportOcr passport,
    VisaOcr visa,
    NationalIdOcr nationalId) {
    
}
