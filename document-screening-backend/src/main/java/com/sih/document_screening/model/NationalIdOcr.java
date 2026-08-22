package com.sih.document_screening.model;

public record NationalIdOcr(
    String idNumber,
        String fullName,
        double confidence
) {

}
