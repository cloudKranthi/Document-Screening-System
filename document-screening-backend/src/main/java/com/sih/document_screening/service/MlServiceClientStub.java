package com.sih.document_screening.service;

import java.util.Collections;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.sih.document_screening.dto.BiometricTamperingResponse;
import com.sih.document_screening.dto.MlOcrResponse;
import com.sih.document_screening.dto.NationalIdOcr;
import com.sih.document_screening.dto.PassportOcr;
import com.sih.document_screening.dto.VisaOcr;

@Service
public class MlServiceClientStub implements MlServiceClient {

    @Override
    public MlOcrResponse extractDocumentData(MultipartFile passport, MultipartFile visa, MultipartFile nationalId) {
        // Returns blank/empty OCR records so nothing fails during local runs or deployment
        return new MlOcrResponse(
            new PassportOcr(null, null, null, null, null, null, false, 0.0),
            new VisaOcr(null, null, null, 0.0),
            new NationalIdOcr(null, null, 0.0)
        );
    }

    @Override
    public BiometricTamperingResponse verifyBiometricsAndTampering(MultipartFile documentPhoto, MultipartFile livePhoto) {
        // Returns neutral biometric & tampering scores
        return new BiometricTamperingResponse(
            0.0,
            0.0,
            0.0,
            false,
            Collections.emptyList()
        );
    }
}