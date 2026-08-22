package com.sih.document_screening.service;

import com.sih.document_screening.dto.BiometricTamperingResponse;
import com.sih.document_screening.dto.MlOcrResponse;
import com.sih.document_screening.dto.NationalIdOcr;
import com.sih.document_screening.dto.PassportOcr;
import com.sih.document_screening.dto.ScreeningFinalDecision;
import com.sih.document_screening.dto.ScreeningInitResponse;
import com.sih.document_screening.dto.ScreeningSummaryDTO;
import com.sih.document_screening.dto.VisaOcr;
import com.sih.document_screening.model.NationalIdRecord;
import com.sih.document_screening.model.PassportRecord;
import com.sih.document_screening.model.Screening;
import com.sih.document_screening.model.VerificationScore;
import com.sih.document_screening.model.VisaRecord;
import com.sih.document_screening.repository.ScreeningRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class ScreeningService {

    private final ScreeningRepository screeningRepository;
    private final DocumentValidationService validationService;
    private final MlServiceClient mlServiceClient;

    /**
     * Step 1: Ingest documents, extract OCR via ML, validate rules, save real data.
     */
    @Transactional
    public ScreeningInitResponse processAndValidateDocuments(
            MultipartFile passportFile,
            MultipartFile visaFile,
            MultipartFile nationalIdFile,
            String claimedNationality) {

        // 1. Call ML Service for live OCR extraction
        MlOcrResponse ocrResponse = mlServiceClient.extractDocumentData(passportFile, visaFile, nationalIdFile);

        // 2. Initialize parent Screening entity
        Screening screening = new Screening();
        screening.setClaimedNationality(claimedNationality.trim());
        screening.setStatus("IN_PROGRESS");

        // 3. Map extracted OCR data into respective JPA entities
        PassportRecord passport = mapToPassportRecord(ocrResponse.passport(), screening);
        VisaRecord visa = mapToVisaRecord(ocrResponse.visa(), screening);
        NationalIdRecord nationalId = mapToNationalIdRecord(ocrResponse.nationalId(), screening);

        screening.setPassportRecord(passport);
        screening.setVisaRecord(visa);
        screening.setNationalIdRecord(nationalId);

        // 4. Run validation rules against extracted data
        ScreeningInitResponse.ValidationSummary validationSummary = validationService.validate(
                claimedNationality, passport, visa, nationalId);

        if (!validationSummary.isValid()) {
            screening.setStatus("VALIDATION_FAILED");
        }

        // 5. Persist entity to PostgreSQL
        Screening savedScreening = screeningRepository.save(screening);
        log.info("Saved screening session: {} for document: {}", savedScreening.getId(), passport.getDocumentNumber());

        // 6. Build response DTO from persisted data
        ScreeningInitResponse.ExtractedDocumentData extractedData = new ScreeningInitResponse.ExtractedDocumentData(
                passport.getDocumentNumber(),
                passport.getFullName(),
                passport.getDateOfBirth(),
                passport.getExpiryDate(),
                passport.getNationality(),
                passport.getMrzChecksumValid(),
                visa.getVisaNumber(),
                visa.getValidUntil(),
                visa.getVisaType(),
                nationalId.getNationalIdNumber(),
                nationalId.getFullName()
        );

        return new ScreeningInitResponse(
                savedScreening.getId(),
                savedScreening.getStatus(),
                extractedData,
                validationSummary
        );
    }

    /**
     * Step 2: Send live selfie & doc photo to ML model, compute dynamic risk score.
     */
    @Transactional
    public ScreeningFinalDecision runBiometricAndTamperingVerification(
            UUID screeningId,
            MultipartFile livePhoto) {

        Screening screening = screeningRepository.findById(screeningId)
                .orElseThrow(() -> new IllegalArgumentException("Screening record not found for ID: " + screeningId));

        // 1. Call ML service for deep learning verification
        BiometricTamperingResponse mlResult = mlServiceClient.verifyBiometricsAndTampering(null, livePhoto);

        // 2. Map ML response to entity
        VerificationScore score = new VerificationScore();
        score.setScreening(screening);
        score.setFaceMatchScore(mlResult.faceSimilarityScore());
        score.setPhotoTamperingScore(mlResult.photoTamperingScore());
        score.setTextManipulationScore(mlResult.textManipulationScore());
        score.setMetadataAnomalyFound(mlResult.metadataAnomalyFound());

        List<String> flaggedReasons = new ArrayList<>();
        if (mlResult.detectedAnomalies() != null) {
            flaggedReasons.addAll(mlResult.detectedAnomalies());
        }

        // 3. Dynamic Risk Scoring Engine
        int riskScore = 0;

        if (score.getFaceMatchScore() != null && score.getFaceMatchScore() < 0.75) {
            riskScore += 40;
            flaggedReasons.add("Biometric facial similarity below threshold (" + score.getFaceMatchScore() + ")");
        }
        if (score.getPhotoTamperingScore() != null && score.getPhotoTamperingScore() > 0.50) {
            riskScore += 30;
            flaggedReasons.add("Photo tampering detected by ELA (Score: " + score.getPhotoTamperingScore() + ")");
        }
        if ("VALIDATION_FAILED".equals(screening.getStatus())) {
            riskScore += 25;
            flaggedReasons.add("Document failed field integrity / ICAO checksum checks");
        }
        if (Boolean.TRUE.equals(score.getMetadataAnomalyFound())) {
            riskScore += 15;
            flaggedReasons.add("Metadata anomaly or editing artifacts present");
        }

        riskScore = Math.min(100, Math.max(0, riskScore));
        score.setFlaggedReasons(flaggedReasons);
        screening.setVerificationScore(score);
        screening.setFinalRiskScore(riskScore);

        // 4. Decision Thresholds
        String verdict;
        String riskCategory;

        if (riskScore >= 60) {
            riskCategory = "HIGH";
            verdict = "REJECT";
            screening.setStatus("FLAGGED_FOR_ADMIN");
        } else if (riskScore >= 30) {
            riskCategory = "MEDIUM";
            verdict = "MANUAL_REVIEW_REQUIRED";
            screening.setStatus("FLAGGED_FOR_ADMIN");
        } else {
            riskCategory = "LOW";
            verdict = "ALLOW";
            screening.setStatus("PASSED");
        }

        screening.setRiskCategory(riskCategory);
        screeningRepository.save(screening);

        return new ScreeningFinalDecision(
                screening.getId(),
                verdict,
                riskCategory,
                riskScore,
                new ScreeningFinalDecision.BiometricSummary(
                        score.getFaceMatchScore() != null ? score.getFaceMatchScore() : 0.0,
                        score.getFaceMatchScore() != null && score.getFaceMatchScore() >= 0.75
                ),
                new ScreeningFinalDecision.TamperingSummary(
                        score.getPhotoTamperingScore() != null ? score.getPhotoTamperingScore() : 0.0,
                        score.getTextManipulationScore() != null ? score.getTextManipulationScore() : 0.0,
                        Boolean.TRUE.equals(score.getMetadataAnomalyFound())
                ),
                flaggedReasons
        );
    }

    /**
     * Step 3: Admin Review Queue
     */
    @Transactional(readOnly = true)
    public List<ScreeningSummaryDTO> getFlaggedCasesForReview(int page, int size) {
        return screeningRepository.findAll(PageRequest.of(page, size))
                .stream()
                .filter(s -> "FLAGGED_FOR_ADMIN".equals(s.getStatus()) || "VALIDATION_FAILED".equals(s.getStatus()))
                .map(s -> new ScreeningSummaryDTO(
                        s.getId(),
                        s.getClaimedNationality(),
                        s.getStatus(),
                        s.getFinalRiskScore() != null ? s.getFinalRiskScore() : 0,
                        s.getRiskCategory() != null ? s.getRiskCategory() : "UNKNOWN",
                        s.getVerificationScore() != null ? s.getVerificationScore().getFlaggedReasons() : List.of(),
                        s.getCreatedAt()
                ))
                .toList();
    }

    // Mapping Helpers
    private PassportRecord mapToPassportRecord(PassportOcr ocr, Screening screening) {
        PassportRecord p = new PassportRecord();
        p.setScreening(screening);
        if (ocr != null) {
            p.setDocumentNumber(ocr.documentNumber());
            p.setFullName(ocr.fullName());
            p.setDateOfBirth(ocr.dateOfBirth());
            p.setExpiryDate(ocr.expiryDate());
            p.setNationality(ocr.nationality());
            p.setRawMrz(ocr.rawMrz());
            p.setMrzChecksumValid(ocr.mrzValid());
            p.setOcrConfidence(ocr.confidence());
        }
        return p;
    }

    private VisaRecord mapToVisaRecord(VisaOcr ocr, Screening screening) {
        VisaRecord v = new VisaRecord();
        v.setScreening(screening);
        if (ocr != null) {
            v.setVisaNumber(ocr.visaNumber());
            v.setVisaType(ocr.visaType());
            v.setValidUntil(ocr.validUntil());
            v.setOcrConfidence(ocr.confidence());
        }
        return v;
    }

    private NationalIdRecord mapToNationalIdRecord(NationalIdOcr ocr, Screening screening) {
        NationalIdRecord n = new NationalIdRecord();
        n.setScreening(screening);
        if (ocr != null) {
            n.setNationalIdNumber(ocr.idNumber());
            n.setFullName(ocr.fullName());
            n.setOcrConfidence(ocr.confidence());
        }
        return n;
    }
}