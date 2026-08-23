"""Face Verification Service for Identity Document and Live Selfie Biometric Matching.

Coordinates face detection, quality screening, 5-point landmark alignment,
128-d L2-normalized embedding extraction, and cosine similarity comparison.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

from app.config import settings
from app.models.schemas import FaceDetail, FaceVerificationResult
from app.services.face_detection_service import FaceDetectionService
from app.services.face_alignment_service import FaceAlignmentService
from app.services.face_embedding_service import FaceEmbeddingService

logger = logging.getLogger(__name__)


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes deterministic cosine similarity between two float vectors.
    
    For L2-normalized vectors, this is simply the dot product dot(vec_a, vec_b).
    """
    if vec_a is None or vec_b is None:
        return 0.0

    a = np.array(vec_a, dtype=np.float32).flatten()
    b = np.array(vec_b, dtype=np.float32).flatten()

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    # Normalized dot product
    dot_prod = float(np.dot(a, b)) / (norm_a * norm_b)
    return float(np.clip(dot_prod, -1.0, 1.0))


def normalize_similarity_to_public_score(raw_cosine: float) -> float:
    """Maps raw cosine similarity to an interpretable [0.0, 1.0] public metric.
    
    Note: similarity_score is a similarity metric, NOT a percentage probability of same person.
    """
    # Clamp to [0.0, 1.0] range
    public_score = max(0.0, min(1.0, raw_cosine))
    return round(float(public_score), 4)


def classify_similarity_band(score: Optional[float]) -> Tuple[str, str]:
    """Classifies the final similarity score into operational decision bands and UI colors.
    
    Bands:
    - >= 0.80       -> STRONG_MATCH, UI color GREEN
    - 0.75 - 0.7999 -> BORDERLINE_MATCH, UI color YELLOW
    - 0.60 - 0.7499 -> MANUAL_REVIEW, UI color ORANGE
    - < 0.60        -> NO_MATCH, UI color RED
    - None          -> NOT_EVALUATED, UI color GRAY
    """
    if score is None:
        return "NOT_EVALUATED", "GRAY"

    if score >= 0.80:
        return "STRONG_MATCH", "GREEN"
    elif score >= 0.75:
        return "BORDERLINE_MATCH", "YELLOW"
    elif score >= 0.60:
        return "MANUAL_REVIEW", "ORANGE"
    else:
        return "NO_MATCH", "RED"


class FaceVerificationService:
    """End-to-end face verification orchestrator between identity documents and selfies."""


    def __init__(
        self,
        detection_service: Optional[FaceDetectionService] = None,
        alignment_service: Optional[FaceAlignmentService] = None,
        embedding_service: Optional[FaceEmbeddingService] = None,
        threshold: Optional[float] = None
    ):
        self.detector = detection_service or FaceDetectionService()
        self.aligner = alignment_service or FaceAlignmentService()
        self.embedder = embedding_service or FaceEmbeddingService()
        self.threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD

    def verify_faces(
        self,
        document_image: np.ndarray,
        selfie_image: np.ndarray
    ) -> FaceVerificationResult:
        """Compares identity document portrait against selfie image.
        
        Args:
            document_image: Decoded BGR identity document image.
            selfie_image: Decoded BGR selfie image.
            
        Returns:
            Structured FaceVerificationResult.
        """
        all_warnings: List[str] = []

        # -------------------------------------------------------------
        # 1. Validate Input Images
        # -------------------------------------------------------------
        if document_image is None or document_image.size == 0:
            return self._build_error_result(
                status="PROCESSING_ERROR",
                warnings=["Document image could not be decoded or is empty."],
                doc_detail=FaceDetail(detected=False, warnings=["Document image decoding failed."]),
                selfie_detail=FaceDetail(detected=False)
            )

        if selfie_image is None or selfie_image.size == 0:
            return self._build_error_result(
                status="PROCESSING_ERROR",
                warnings=["Selfie image could not be decoded or is empty."],
                doc_detail=FaceDetail(detected=False),
                selfie_detail=FaceDetail(detected=False, warnings=["Selfie image decoding failed."])
            )

        # -------------------------------------------------------------
        # 2. Detect Document Portrait Face
        # -------------------------------------------------------------
        doc_face, doc_quality, doc_warns = self.detector.detect_document_portrait(document_image)
        all_warnings.extend(doc_warns)

        if doc_face is None:
            return self._build_error_result(
                status="NO_FACE",
                warnings=all_warnings or ["No face detected in identity document."],
                doc_detail=FaceDetail(detected=False, warnings=doc_warns),
                selfie_detail=FaceDetail(detected=False)
            )

        doc_detail = FaceDetail(
            detected=True,
            detection_confidence=doc_face.get("confidence"),
            quality_score=doc_quality.get("quality_score") if doc_quality else None,
            quality_passed=doc_quality.get("quality_passed") if doc_quality else None,
            bbox=doc_face.get("bbox"),
            warnings=doc_warns
        )

        # -------------------------------------------------------------
        # 3. Detect Selfie Face & Enforce Single Subject
        # -------------------------------------------------------------
        selfie_face, selfie_quality, selfie_warns, selfie_status = self.detector.detect_selfie_face(selfie_image)
        all_warnings.extend(selfie_warns)

        if selfie_status == "NO_FACE":
            return self._build_error_result(
                status="NO_FACE",
                warnings=all_warnings,
                doc_detail=doc_detail,
                selfie_detail=FaceDetail(detected=False, warnings=selfie_warns)
            )

        if selfie_status == "MULTIPLE_FACES":
            return self._build_error_result(
                status="MULTIPLE_FACES",
                warnings=all_warnings,
                doc_detail=doc_detail,
                selfie_detail=FaceDetail(detected=True, warnings=selfie_warns)
            )

        selfie_detail = FaceDetail(
            detected=True,
            detection_confidence=selfie_face.get("confidence") if selfie_face else None,
            quality_score=selfie_quality.get("quality_score") if selfie_quality else None,
            quality_passed=selfie_quality.get("quality_passed") if selfie_quality else None,
            bbox=selfie_face.get("bbox") if selfie_face else None,
            warnings=selfie_warns
        )

        if selfie_status == "INSUFFICIENT_QUALITY" or (doc_quality and not doc_quality.get("quality_passed")):
            return self._build_error_result(
                status="INSUFFICIENT_QUALITY",
                warnings=all_warnings,
                doc_detail=doc_detail,
                selfie_detail=selfie_detail
            )

        # -------------------------------------------------------------
        # 4. Face Alignment (112x112 canonical crop)
        # -------------------------------------------------------------
        try:
            recognizer = self.embedder.get_recognizer()
            doc_aligned = self.aligner.align_face(document_image, doc_face, recognizer=recognizer)
            selfie_aligned = self.aligner.align_face(selfie_image, selfie_face, recognizer=recognizer)
        except Exception as e:
            logger.error(f"Face alignment exception: {e}")
            return self._build_error_result(
                status="PROCESSING_ERROR",
                warnings=all_warnings + [f"Face alignment encountered an error: {str(e)}"],
                doc_detail=doc_detail,
                selfie_detail=selfie_detail
            )

        if doc_aligned is None or selfie_aligned is None:
            return self._build_error_result(
                status="PROCESSING_ERROR",
                warnings=all_warnings + ["Face alignment failed on one or both inputs."],
                doc_detail=doc_detail,
                selfie_detail=selfie_detail
            )

        # -------------------------------------------------------------
        # 5. Extract 128-d L2-Normalized Embeddings
        # -------------------------------------------------------------
        try:
            doc_emb = self.embedder.get_embedding(doc_aligned)
            selfie_emb = self.embedder.get_embedding(selfie_aligned)
        except Exception as e:
            logger.error(f"Embedding extraction exception: {e}")
            return self._build_error_result(
                status="PROCESSING_ERROR",
                warnings=all_warnings + [f"Embedding extraction encountered an error: {str(e)}"],
                doc_detail=doc_detail,
                selfie_detail=selfie_detail
            )

        if doc_emb is None or selfie_emb is None:
            return self._build_error_result(
                status="PROCESSING_ERROR",
                warnings=all_warnings + ["Face embedding extraction failed."],
                doc_detail=doc_detail,
                selfie_detail=selfie_detail
            )


        # -------------------------------------------------------------
        # 6. Compute Cosine Similarity & Normalization
        # -------------------------------------------------------------
        raw_cosine = compute_cosine_similarity(doc_emb, selfie_emb)
        similarity_score = normalize_similarity_to_public_score(raw_cosine)
        match_band, ui_color = classify_similarity_band(similarity_score)

        # -------------------------------------------------------------
        # 7. Evaluate Match Against Configurable Threshold
        # -------------------------------------------------------------
        is_match = bool(similarity_score >= self.threshold)
        status = "MATCH" if is_match else "NO_MATCH"

        return FaceVerificationResult(
            success=True,
            status=status,
            match=is_match,
            similarity_score=similarity_score,
            raw_cosine_similarity=round(raw_cosine, 4),
            threshold=self.threshold,
            match_band=match_band,
            ui_color=ui_color,
            document_face=doc_detail,
            selfie_face=selfie_detail,
            warnings=all_warnings,
            disclaimer="Face similarity is a biometric comparison signal and does not by itself prove identity or document authenticity."
        )

    def _build_error_result(
        self,
        status: str,
        warnings: List[str],
        doc_detail: FaceDetail,
        selfie_detail: FaceDetail
    ) -> FaceVerificationResult:
        """Helper to construct non-match or error states with explainable context."""
        return FaceVerificationResult(
            success=(status not in ["PROCESSING_ERROR"]),
            status=status,
            match=None,
            similarity_score=None,
            raw_cosine_similarity=None,
            threshold=self.threshold,
            match_band="NOT_EVALUATED",
            ui_color="GRAY",
            document_face=doc_detail,
            selfie_face=selfie_detail,
            warnings=warnings,
            disclaimer="Face similarity is a biometric comparison signal and does not by itself prove identity or document authenticity."
        )

