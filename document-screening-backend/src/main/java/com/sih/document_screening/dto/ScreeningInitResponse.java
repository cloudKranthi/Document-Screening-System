package com.sih.document_screening.dto;

import java.util.UUID;

public record ScreeningInitResponse(UUID screeningId,
    String status
   ) {
} 
