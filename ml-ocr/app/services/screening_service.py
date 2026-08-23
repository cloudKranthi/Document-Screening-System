"""Unified Screening Service coordinating OCR extraction, MRZ validation, tampering forensics, and biometric face verification."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.models.schemas import (
    DocumentTypeEnum,
    OCRExtractResponse,
    UnifiedScreeningResult,
    TamperingResult,
    FaceVerificationResult,
    FaceDetail,
)
from app.services.confidence_service import ConfidenceService
from app.services.document_service import DocumentService
from app.services.face_verification_service import FaceVerificationService
from app.services.image_service import ImageService
from app.services.mrz_service import MRZService
from app.services.ocr_service import OCRService
from app.services.tampering_service import TamperingService

logger = logging.getLogger(__name__)


class ScreeningService:
    """Orchestrates end-to-end document verification without duplicating individual module algorithms."""

    def __init__(
        self,
        ocr_service: Optional[OCRService] = None,
        mrz_service: Optional[MRZService] = None,
        document_service: Optional[DocumentService] = None,
        tampering_service: Optional[TamperingService] = None,
        face_verification_service: Optional[FaceVerificationService] = None,
    ):
        self.ocr_service = ocr_service or OCRService()
        self.mrz_service = mrz_service or MRZService()
        self.document_service = document_service or DocumentService()
        self.tampering_service = tampering_service or TamperingService()
        self.face_verification_service = face_verification_service or FaceVerificationService()

    def screen(
        self,
        document_bytes: bytes,
        selfie_bytes: Optional[bytes] = None,
        document_type: str = "auto"
    ) -> UnifiedScreeningResult:
        """Executes the full screening pipeline across OCR, Tampering, and Face Verification."""
        all_warnings: List[str] = []
        doc_type_clean = (document_type or "auto").strip().lower()

        # -------------------------------------------------------------
        # 1. Decode & Preprocess Document Image
        # -------------------------------------------------------------
        try:
            original_doc_img, ocr_optimized_img, processing_meta = ImageService.process_document_image(document_bytes)
        except Exception as img_err:
            logger.error(f"Document image preprocessing failed: {img_err}")
            raise ValueError(f"Could not decode or process document image: {str(img_err)}")

        if original_doc_img is None or original_doc_img.size == 0:
            raise ValueError("Decoded document image is empty or invalid.")

        # -------------------------------------------------------------
        # 2. OCR & MRZ Document Extraction Pipeline
        # -------------------------------------------------------------
        ocr_data: Dict[str, Any] = {}
        extracted_fields: Dict[str, Any] = {}
        field_confs: Dict[str, float] = {}
        mrz_fields: Dict[str, Any] = {}
        all_regions = []
        effective_doc_type = doc_type_clean

        try:
            ocr_result = self.ocr_service.extract(ocr_optimized_img)

            # Multi-crop MRZ candidate generation
            mrz_candidate_sources: List[Tuple[str, str]] = []
            try:
                crop_variants = ImageService.get_all_mrz_crop_variants(
                    ocr_optimized_img,
                    ratios=settings.MRZ_CROP_RATIOS,
                    scale_factor=settings.MRZ_UPSCALE_FACTOR
                )
                for var_name, ratio, var_img in crop_variants:
                    for psm in [6, 4, 11]:
                        src_label = f"{var_name}_psm{psm}"
                        try:
                            mrz_ocr_res = self.ocr_service.extract_mrz(var_img, psm=psm)
                            if mrz_ocr_res.raw_text.strip():
                                mrz_candidate_sources.append((src_label, mrz_ocr_res.raw_text))
                        except Exception:
                            pass
            except Exception as crop_err:
                logger.warning(f"MRZ crop candidate generation note: {crop_err}")

            region_texts = [r.text for r in ocr_result.regions]
            mrz_result, mrz_fields, _ = self.mrz_service.extract_and_validate_mrz(
                raw_ocr_text=ocr_result.raw_text,
                ocr_lines=region_texts,
                mrz_candidate_texts=mrz_candidate_sources,
                include_debug=False
            )

            all_regions = list(ocr_result.regions)
            if mrz_result.detected and mrz_result.line1 and mrz_result.line2:
                mrz_lines = [mrz_result.line1, mrz_result.line2]
                if mrz_result.line3:
                    mrz_lines.append(mrz_result.line3)
                mrz_block = "\n".join(mrz_lines)
                if mrz_result.line1 not in ocr_result.raw_text:
                    combined_text = ocr_result.raw_text + "\n" + mrz_block
                else:
                    combined_text = ocr_result.raw_text
            else:
                combined_text = ocr_result.raw_text

            avg_conf = ConfidenceService.calculate_average_confidence(all_regions)
            if avg_conf == 0.0 and ocr_result.average_confidence > 0.0:
                avg_conf = ocr_result.average_confidence

            effective_doc_type, extracted_fields, field_confs, extraction_warnings, _, field_sources_info = self.document_service.process_extraction(
                requested_type=doc_type_clean,
                ocr_text=combined_text,
                mrz_result=mrz_result,
                mrz_fields=mrz_fields,
                ocr_regions=all_regions,
                document_image=ocr_optimized_img,
                ocr_service=self.ocr_service,
                include_debug=False
            )
            all_warnings.extend(extraction_warnings)

            ocr_extract_resp = OCRExtractResponse(
                success=True,
                document_type=effective_doc_type,
                average_confidence=avg_conf,
                extracted_text=combined_text,
                fields=extracted_fields,
                field_confidences=field_confs,
                field_sources=field_sources_info,
                mrz=mrz_result,
                field_validation=mrz_result.field_validation,
                ocr_regions=all_regions,
                processing=processing_meta,
                language_mode=settings.DEFAULT_LANGUAGE_MODE,
                warnings=extraction_warnings
            )
            ocr_data = ocr_extract_resp.model_dump()
            
            # Explicitly release intermediate OCR image arrays to reclaim memory before tampering analysis
            del ocr_optimized_img
        except Exception as ocr_err:
            logger.error(f"OCR extraction failed during screening: {ocr_err}")
            all_warnings.append(f"OCR extraction error: {str(ocr_err)}")
            ocr_data = {
                "success": False,
                "error": str(ocr_err),
                "fields": {},
                "warnings": [f"OCR extraction error: {str(ocr_err)}"]
            }


        # -------------------------------------------------------------
        # 3. Multi-Signal Tampering Forensics
        # -------------------------------------------------------------
        tampering_data: Optional[Dict[str, Any]] = None
        try:
            vis_fields_to_pass = getattr(self.document_service, "last_visual_fields", None) or extracted_fields
            vis_confs_to_pass = getattr(self.document_service, "last_visual_confs", None) or field_confs

            tampering_res = self.tampering_service.analyze_document(
                image_bytes=document_bytes,
                document_image=original_doc_img,
                visual_fields=vis_fields_to_pass,
                mrz_fields=mrz_fields,
                field_confidences=vis_confs_to_pass,
                layout_regions=all_regions
            )
            tampering_data = tampering_res.model_dump()
            if tampering_res.warnings:
                all_warnings.extend(tampering_res.warnings)
        except Exception as tamp_err:
            logger.error(f"Tampering analysis encountered an error during screening: {tamp_err}")
            all_warnings.append(f"Tampering analysis encountered an error: {str(tamp_err)}")
            tampering_data = {
                "tampering_risk_score": 0.0,
                "risk_level": "UNKNOWN",
                "evidence_coverage": 0.0,
                "signals": {},
                "indicators": [],
                "warnings": [f"Tampering module error: {str(tamp_err)}"],
                "disclaimer": "Tampering evaluation could not be completed."
            }

        # -------------------------------------------------------------
        # 4. Biometric Face Verification (if selfie provided)
        # -------------------------------------------------------------
        face_data: Optional[Dict[str, Any]] = None
        if selfie_bytes is not None and len(selfie_bytes) > 0:
            try:
                selfie_np = np.frombuffer(selfie_bytes, np.uint8)
                selfie_img = cv2.imdecode(selfie_np, cv2.IMREAD_COLOR)

                if selfie_img is None or selfie_img.size == 0:
                    face_data = {
                        "success": False,
                        "status": "PROCESSING_ERROR",
                        "match": None,
                        "similarity_score": None,
                        "raw_cosine_similarity": None,
                        "threshold": settings.FACE_MATCH_THRESHOLD,
                        "match_band": "NOT_EVALUATED",
                        "ui_color": "GRAY",
                        "document_face": {"detected": False, "warnings": []},
                        "selfie_face": {"detected": False, "warnings": []},
                        "warnings": ["Could not decode uploaded selfie image."],
                        "disclaimer": "Face similarity is a biometric comparison signal and does not by itself prove identity or document authenticity."
                    }
                    all_warnings.append("Could not decode uploaded selfie image.")
                else:
                    face_res = self.face_verification_service.verify_faces(
                        document_image=original_doc_img,
                        selfie_image=selfie_img
                    )
                    face_data = face_res.model_dump()
                    if face_res.warnings:
                        all_warnings.extend(face_res.warnings)
            except Exception as face_err:
                logger.error(f"Face verification failed during screening: {face_err}")
                face_data = {
                    "success": False,
                    "status": "PROCESSING_ERROR",
                    "match": None,
                    "similarity_score": None,
                    "raw_cosine_similarity": None,
                    "threshold": settings.FACE_MATCH_THRESHOLD,
                    "match_band": "NOT_EVALUATED",
                    "ui_color": "GRAY",
                    "document_face": {"detected": False, "warnings": []},
                    "selfie_face": {"detected": False, "warnings": []},
                    "warnings": [f"Face verification module error: {str(face_err)}"],
                    "disclaimer": "Face similarity is a biometric comparison signal and does not by itself prove identity or document authenticity."
                }
                all_warnings.append(f"Face verification module error: {str(face_err)}")
        else:
            face_data = {
                "status": "NOT_EVALUATED",
                "match": None,
                "similarity_score": None,
                "raw_cosine_similarity": None,
                "threshold": settings.FACE_MATCH_THRESHOLD,
                "match_band": "NOT_EVALUATED",
                "ui_color": "GRAY",
                "document_face": {
                    "detected": False,
                    "detection_confidence": None,
                    "quality_score": None,
                    "quality_passed": None,
                    "bbox": None,
                    "warnings": []
                },
                "selfie_face": {
                    "detected": False,
                    "detection_confidence": None,
                    "quality_score": None,
                    "quality_passed": None,
                    "bbox": None,
                    "warnings": []
                },
                "warnings": ["No selfie image provided for biometric face verification."],
                "disclaimer": "Face similarity is a biometric comparison signal and does not by itself prove identity or document authenticity."
            }

        # -------------------------------------------------------------
        # 5. Consolidate Unified Result
        # -------------------------------------------------------------
        unique_warnings = list(dict.fromkeys(all_warnings))

        return UnifiedScreeningResult(
            success=True,
            document_type=effective_doc_type,
            ocr=ocr_data,
            tampering=tampering_data,
            face_verification=face_data,
            warnings=unique_warnings
        )
