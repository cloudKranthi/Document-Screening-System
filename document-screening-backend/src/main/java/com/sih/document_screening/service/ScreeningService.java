package com.sih.document_screening.service;

import com.sih.document_screening.dto.BiometricSummary;
import com.sih.document_screening.dto.BiometricTamperingResponse;
import com.sih.document_screening.dto.MlOcrResponse;
import com.sih.document_screening.dto.NationalIdOcr;
import com.sih.document_screening.dto.PassportOcr;
import com.sih.document_screening.dto.ScreeningInitResponse;
import com.sih.document_screening.dto.ScreeningSummaryDTO;
import com.sih.document_screening.dto.TamperingSummary;
import com.sih.document_screening.dto.ValidationSummary;
import com.sih.document_screening.dto.VisaOcr;
import com.sih.document_screening.model.NationalIdRecord;
import com.sih.document_screening.model.PassportRecord;
import com.sih.document_screening.model.Screening;
import com.sih.document_screening.model.ScreeningStatus;
import com.sih.document_screening.model.VerificationScore;
import com.sih.document_screening.model.VisaRecord;
import com.sih.document_screening.model.VisaType;
import com.sih.document_screening.repository.ScreeningRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class ScreeningService {

    private final ScreeningRepository screeningRepository;
    private final DocumentValidationService validationService;
    private final MlServiceClient mlServiceClient;

    /**
     * Complete pipeline: Ingest 4 files, perform OCR, validate cross-fields, match selfie with passport, compute risk.
     */
    @Transactional
    public ScreeningInitResponse processCompleteScreening(
            MultipartFile passportFile,
            MultipartFile visaFile,
            MultipartFile nationalIdFile,
            MultipartFile livePhoto,
            String claimedNationality) {

        // 1. ML OCR Extraction
        MlOcrResponse ocrResponse = mlServiceClient.extractDocumentData(passportFile, visaFile, nationalIdFile);

        // 2. ML Biometrics & Tampering
        BiometricTamperingResponse mlBioResult = mlServiceClient.verifyBiometricsAndTampering(passportFile, livePhoto);

        // 3. Initialize parent Screening entity
        Screening screening = new Screening();
        screening.setClaimedNationality(claimedNationality.trim());
        screening.setStatus(ScreeningStatus.IN_PROGRESS);

        // 4. Map OCR details into respective JPA entities
        PassportRecord passport = mapToPassportRecord(ocrResponse.passport(), screening);
        VisaRecord visa = mapToVisaRecord(ocrResponse.visa(), screening);
        NationalIdRecord nationalId = mapToNationalIdRecord(ocrResponse.nationalId(), screening);

        screening.setPassportRecord(passport);
        screening.setVisaRecord(visa);
        screening.setNationalIdRecord(nationalId);

        // 5. Run Document Validation Rules (cross-field matching, ICAO checksum, expiry, stay duration)
        ValidationSummary validationSummary = validationService.validate(
                claimedNationality, passport, visa, nationalId);

        // 6. Map Biometric & Tampering scores to VerificationScore entity
        VerificationScore score = new VerificationScore();
        score.setScreening(screening);
        score.setFaceMatchScore(mlBioResult.faceSimilarityScore());
        score.setPhotoTamperingScore(mlBioResult.photoTamperingScore());
        score.setTextManipulationScore(mlBioResult.textManipulationScore());
        score.setMetadataAnomalyFound(mlBioResult.metadataAnomalyFound());

        List<String> flaggedReasons = new ArrayList<>();
        if (mlBioResult.detectedAnomalies() != null) {
            flaggedReasons.addAll(mlBioResult.detectedAnomalies());
        }
        if (!validationSummary.isValid() && validationSummary.validationErrors() != null) {
            flaggedReasons.addAll(validationSummary.validationErrors());
        }

        // 7. Composite Risk Scoring Engine (0 - 100)
        int riskScore = 0;

        if (score.getFaceMatchScore() != null && score.getFaceMatchScore() < 0.75) {
            riskScore += 40;
            flaggedReasons.add("Biometric facial similarity below threshold (" + score.getFaceMatchScore() + ")");
        }
        if (score.getPhotoTamperingScore() != null && score.getPhotoTamperingScore() > 0.50) {
            riskScore += 30;
            flaggedReasons.add("Photo tampering detected by ELA (Score: " + score.getPhotoTamperingScore() + ")");
        }
        if (!validationSummary.isValid()) {
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

        // 8. Determine Final Border Verdict
        String verdict;
        String riskCategory;

        if (riskScore >= 60) {
            riskCategory = "HIGH";
            verdict = "REJECT";
            screening.setStatus(ScreeningStatus.FLAGGED_FOR_ADMIN);
        } else if (riskScore >= 30) {
            riskCategory = "MEDIUM";
            verdict = "MANUAL_REVIEW_REQUIRED";
            screening.setStatus(ScreeningStatus.FLAGGED_FOR_ADMIN);
        } else {
            riskCategory = "LOW";
            verdict = "ALLOW";
            screening.setStatus(ScreeningStatus.PASSED);
        }

        screening.setRiskCategory(riskCategory);

        // Persist all records and scores to PostgreSQL
        Screening savedScreening = screeningRepository.save(screening);
        log.info("Completed screening for passport: {}, verdict: {}, riskScore: {}", 
                passport.getDocumentNumber(), verdict, riskScore);

        // 9. Build and return ScreeningInitResponse
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

        BiometricSummary biometricSummary = new BiometricSummary(
                score.getFaceMatchScore() != null ? score.getFaceMatchScore() : 0.0,
                score.getFaceMatchScore() != null && score.getFaceMatchScore() >= 0.75
        );

        TamperingSummary tamperingSummary = new TamperingSummary(
                score.getPhotoTamperingScore() != null ? score.getPhotoTamperingScore() : 0.0,
                score.getTextManipulationScore() != null ? score.getTextManipulationScore() : 0.0,
                Boolean.TRUE.equals(score.getMetadataAnomalyFound())
        );

        return new ScreeningInitResponse(
                savedScreening.getId(),
                savedScreening.getStatus().name(),
                verdict,
                riskCategory,
                riskScore,
                extractedData,
                validationSummary,
                biometricSummary,
                tamperingSummary,
                flaggedReasons
        );
    }

    /**
     * Admin Review Queue: Fetch suspicious / flagged screenings
     */
    @Transactional(readOnly = true)
    public List<ScreeningSummaryDTO> getFlaggedCasesForReview(int page, int size) {
        return screeningRepository.findAll(PageRequest.of(page, size))
                .stream()
                .filter(s -> s.getStatus() == ScreeningStatus.FLAGGED_FOR_ADMIN
                          || s.getStatus() == ScreeningStatus.SUSPICIOUS
                          )
                .map(s -> new ScreeningSummaryDTO(
                        s.getId(),
                        s.getClaimedNationality(),
                        s.getStatus() != null ? s.getStatus().name() : "UNKNOWN",
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
            v.setVisaType(VisaType.fromString(ocr.visaType()));
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