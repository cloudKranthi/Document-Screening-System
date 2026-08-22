package com.sih.document_screening.service;

import com.sih.document_screening.dto.BiometricTamperingResponse;
import com.sih.document_screening.dto.MlOcrResponse;
import org.springframework.web.multipart.MultipartFile;

public interface MlServiceClient {
    MlOcrResponse extractDocumentData(MultipartFile passport, MultipartFile visa, MultipartFile nationalId);
    BiometricTamperingResponse verifyBiometricsAndTampering(MultipartFile documentPhoto, MultipartFile livePhoto);
}
