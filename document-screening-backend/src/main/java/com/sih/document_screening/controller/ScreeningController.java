package com.sih.document_screening.controller;

import java.util.List;
import java.util.UUID;

import org.apache.tomcat.util.http.parser.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

public class ScreeningController {
    private final ScreeningService screeningService;

    // Step 1: Submit All 3 Documents + Nationality
    @PostMapping(value = "/submit-documents", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ScreeningInitResponse> submitDocuments(
            @RequestParam("passport") MultipartFile passportFile,
            @RequestParam("visa") MultipartFile visaFile,
            @RequestParam("nationalId") MultipartFile nationalIdFile,
            @RequestParam("nationality") String nationality) {
        
        ScreeningInitResponse response = screeningService.processDocuments(
                passportFile, visaFile, nationalIdFile, nationality);
        return ResponseEntity.ok(response);
    }

    // Step 2: Submit Live Photo for Deep Learning Matching & Tampering Checks
    @PostMapping(value = "/{screeningId}/verify-biometrics", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ScreeningFinalDecision> verifyBiometrics(
            @PathVariable UUID screeningId,
            @RequestParam("livePhoto") MultipartFile livePhoto) {
        
        ScreeningFinalDecision decision = screeningService.runBiometricAndTamperingVerification(
                screeningId, livePhoto);
        return ResponseEntity.ok(decision);
    }

    // Step 3: Admin Review Endpoint for High-Risk / Suspicious Cases
    @GetMapping("/admin/flagged-queue")
    public ResponseEntity<List<ScreeningSummaryDTO>> getFlaggedScreenings(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        
        List<ScreeningSummaryDTO> flaggedCases = screeningService.getFlaggedCasesForReview(page, size);
        return ResponseEntity.ok(flaggedCases);
    }
}
