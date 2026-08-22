package com.sih.document_screening.controller;

import java.util.List;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.sih.document_screening.dto.ScreeningInitResponse;
import com.sih.document_screening.dto.ScreeningSummaryDTO;
import com.sih.document_screening.service.ScreeningService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/v1/screenings")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ScreeningController {

    private final ScreeningService screeningService;

    @PostMapping(value = "/submit-and-validate", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ScreeningInitResponse> submitAndValidate(
            @RequestParam("passport") MultipartFile passportFile,
            @RequestParam("visa") MultipartFile visaFile,
            @RequestParam("nationalId") MultipartFile nationalIdFile,
            @RequestParam("livePhoto") MultipartFile livePhoto,
            @RequestParam("nationality") String nationality) {

        ScreeningInitResponse response = screeningService.processCompleteScreening(
                passportFile, visaFile, nationalIdFile, livePhoto, nationality);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/admin/flagged-queue")
    public ResponseEntity<List<ScreeningSummaryDTO>> getFlaggedScreenings(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        List<ScreeningSummaryDTO> flaggedCases = screeningService.getFlaggedCasesForReview(page, size);
        return ResponseEntity.ok(flaggedCases);
    }
}