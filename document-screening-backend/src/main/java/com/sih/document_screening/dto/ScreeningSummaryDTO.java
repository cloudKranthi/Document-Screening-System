package com.sih.document_screening.dto;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
public record ScreeningSummaryDTO(
    UUID screeningId,
    String claimedNationality,
    String status,
    int riskScore,
    String riskCategory,
    List<String> keyViolations,
    LocalDateTime createdAt
) {
    
}
