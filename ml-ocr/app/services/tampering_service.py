"""Document tampering detection service utilizing Error Level Analysis (ELA), EXIF metadata inspection, and cross-zone semantic forensics."""

import io
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.config import settings
from app.models.schemas import (
    FieldComparisonItem,
    OCRRegion,
    TamperingIndicator,
    TamperingResult,
    TamperingSignalDetail,
)
from app.utils.forensic_utils import (
    compute_error_level_analysis,
    compute_noise_residual,
    extract_exif_metadata,
)
from app.utils.logger import get_logger
from app.utils.tampering_utils import (
    compute_local_gradient_and_sharpness,
    detect_orb_copy_move_clusters,
    normalize_date_for_comparison,
    normalize_doc_number_for_comparison,
    normalize_name_for_comparison,
    normalize_nationality_for_comparison,
    normalize_sex_for_comparison,
)

logger = get_logger(__name__)


class TamperingService:
    """Explainable document tampering and digital manipulation detection service.
    
    Evaluates independent forensic signals:
      1. Error Level Analysis (ELA) - JPEG recompression residual variance & localized splicing.
      2. EXIF / Container Metadata - Photo editor footprints (Photoshop, GIMP, etc.).
      3. Local Noise Residual - High-frequency sensor noise distribution.
      4. Cross-Zone Semantic Consistency - Visual OCR fields vs ICAO MRZ fields.
    """

    def __init__(
        self,
        ela_weight: Optional[float] = None,
        metadata_weight: Optional[float] = None,
        noise_weight: Optional[float] = None,
        consistency_weight: Optional[float] = None,
        compression_weight: Optional[float] = None,
        edge_weight: Optional[float] = None,
        copy_move_weight: Optional[float] = None,
        quality: Optional[int] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        block_size: Optional[int] = None,
        editing_keywords: Optional[List[str]] = None,
        ocr_service: Optional[Any] = None,
        mrz_service: Optional[Any] = None,
        document_service: Optional[Any] = None,
    ):
        self.w_ela = ela_weight if ela_weight is not None else (compression_weight if compression_weight is not None else settings.TAMPERING_ELA_WEIGHT)
        self.w_compression = self.w_ela
        self.w_metadata = metadata_weight if metadata_weight is not None else settings.TAMPERING_METADATA_WEIGHT
        self.w_noise = noise_weight if noise_weight is not None else settings.TAMPERING_NOISE_WEIGHT
        self.w_consistency = consistency_weight if consistency_weight is not None else settings.TAMPERING_CONSISTENCY_WEIGHT
        self.w_edge = edge_weight if edge_weight is not None else settings.TAMPERING_EDGE_WEIGHT
        self.w_copy_move = copy_move_weight if copy_move_weight is not None else settings.TAMPERING_COPY_MOVE_WEIGHT

        self.quality = quality if quality is not None else settings.TAMPERING_ELA_QUALITY
        self.low_threshold = low_threshold if low_threshold is not None else settings.TAMPERING_LOW_THRESHOLD
        self.high_threshold = high_threshold if high_threshold is not None else settings.TAMPERING_HIGH_THRESHOLD
        self.block_size = block_size if block_size is not None else settings.TAMPERING_BLOCK_SIZE
        self.editing_keywords = editing_keywords if editing_keywords is not None else settings.TAMPERING_EDITING_SOFTWARE_KEYWORDS

        self._ocr_service = ocr_service
        self._mrz_service = mrz_service
        self._document_service = document_service

    def analyze_document(
        self,
        image_bytes: Optional[bytes] = None,
        document_image: Optional[np.ndarray] = None,
        visual_fields: Optional[Dict[str, Any]] = None,
        mrz_fields: Optional[Dict[str, Any]] = None,
        field_confidences: Optional[Dict[str, float]] = None,
        layout_regions: Optional[List[OCRRegion]] = None,
    ) -> TamperingResult:
        """Executes explainable tampering risk evaluation on document image, fields, and metadata.
        
        Args:
            image_bytes: Raw original uploaded image file bytes (for metadata analysis).
            document_image: Decoded BGR or Grayscale document image NumPy array.
            visual_fields: Optional dictionary of structured visual-zone OCR fields.
            mrz_fields: Optional dictionary of parsed MRZ fields.
            field_confidences: Optional dictionary of OCR confidence scores per field.
            layout_regions: Optional list of detected OCR text bounding box regions.
            
        Returns:
            TamperingResult containing overall risk score, risk level, evidence coverage, individual signals, and explainable indicators.
        """
        warnings: List[str] = []
        indicators: List[TamperingIndicator] = []
        signals: Dict[str, TamperingSignalDetail] = {}

        # 1. Safely decode or validate document image
        if document_image is None or document_image.size == 0:
            if image_bytes:
                try:
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    document_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    document_image = None

        has_valid_image = (document_image is not None and document_image.size > 0)
        if not has_valid_image:
            warnings.append("Document image data is empty or unreadable; physical image forensic signals skipped.")
            if not visual_fields and not mrz_fields:
                return TamperingResult(
                    tampering_risk_score=0.0,
                    risk_level="LOW",
                    evidence_coverage=0.0,
                    signals={},
                    indicators=[],
                    warnings=warnings
                )

        if has_valid_image:
            h, w = document_image.shape[:2]
            if h < 200 or w < 200:
                warnings.append(f"Low-resolution image ({w}x{h} px); forensic signal reliability may be reduced.")

        consistency_debug: Dict[str, Any] = {
            "ocr_pipeline_called": False,
            "visual_fields_available": list(visual_fields.keys()) if visual_fields else [],
            "mrz_detected": bool(mrz_fields),
            "mrz_fields_available": list(mrz_fields.keys()) if mrz_fields else [],
            "comparable_fields": []
        }

        # 2. Optional Auto-Extraction for cross-zone consistency when fields are not explicitly supplied
        if (not visual_fields or not mrz_fields) and has_valid_image:
            try:
                from app.services.image_service import ImageService
                from app.services.ocr_service import OCRService
                from app.services.mrz_service import MRZService
                from app.services.document_service import DocumentService

                ocr_srv = self._ocr_service or OCRService()
                mrz_srv = self._mrz_service or MRZService()
                doc_srv = self._document_service or DocumentService()

                encoded_bytes = image_bytes
                if encoded_bytes is None:
                    _, buf = cv2.imencode(".jpg", document_image)
                    encoded_bytes = buf.tobytes()

                _, ocr_opt_img, _ = ImageService.process_document_image(encoded_bytes)
                ocr_res = ocr_srv.extract(ocr_opt_img)
                if not layout_regions:
                    layout_regions = list(ocr_res.regions)

                mrz_candidate_sources = []
                try:
                    crop_variants = ImageService.get_all_mrz_crop_variants(
                        ocr_opt_img,
                        ratios=settings.MRZ_CROP_RATIOS,
                        scale_factor=settings.MRZ_UPSCALE_FACTOR
                    )
                    for var_name, ratio, var_img in crop_variants:
                        for psm in [6, 4, 11]:
                            src_label = f"{var_name}_psm{psm}"
                            try:
                                mrz_ocr_res = ocr_srv.extract_mrz(var_img, psm=psm)
                                if mrz_ocr_res.raw_text.strip():
                                    mrz_candidate_sources.append((src_label, mrz_ocr_res.raw_text))
                            except Exception:
                                pass
                except Exception:
                    pass

                region_texts = [r.text for r in ocr_res.regions]
                mrz_res, auto_mrz_fields, _ = MRZService.extract_and_validate_mrz(
                    raw_ocr_text=ocr_res.raw_text,
                    ocr_lines=region_texts,
                    mrz_candidate_texts=mrz_candidate_sources,
                    include_debug=False
                )

                all_regs = list(ocr_res.regions)
                if mrz_res.detected and mrz_res.line1 and mrz_res.line2:
                    mrz_lines = [mrz_res.line1, mrz_res.line2]
                    if mrz_res.line3:
                        mrz_lines.append(mrz_res.line3)
                    mrz_block = "\n".join(mrz_lines)
                    comb_text = ocr_res.raw_text + "\n" + mrz_block if mrz_res.line1 not in ocr_res.raw_text else ocr_res.raw_text
                else:
                    comb_text = ocr_res.raw_text

                _, ext_fields, ext_confs, _, _, _ = doc_srv.process_extraction(
                    requested_type="auto",
                    ocr_text=comb_text,
                    mrz_result=mrz_res,
                    mrz_fields=auto_mrz_fields,
                    ocr_regions=all_regs,
                    document_image=ocr_opt_img,
                    ocr_service=ocr_srv,
                    include_debug=False
                )

                if not visual_fields:
                    visual_fields = getattr(doc_srv, "last_visual_fields", None) or ext_fields
                if not field_confidences:
                    field_confidences = getattr(doc_srv, "last_visual_confs", None) or ext_confs
                if not mrz_fields:
                    mrz_fields = auto_mrz_fields

                consistency_debug["ocr_pipeline_called"] = True
                consistency_debug["mrz_detected"] = bool(mrz_res.detected)
                consistency_debug["visual_fields_available"] = list(visual_fields.keys()) if visual_fields else []
                consistency_debug["mrz_fields_available"] = list(mrz_fields.keys()) if mrz_fields else []
            except Exception as e:
                logger.warning(f"Internal OCR/MRZ pipeline call inside tampering service skipped/failed: {str(e)}")

        debug_info: Dict[str, Any] = {
            "ela_blocks_evaluated": 0,
            "ela_threshold": 0.0,
            "jpeg_quality": self.quality,
            "metadata_keys_found": []
        }

        # -------------------------------------------------------------
        # Signal 1: Error Level Analysis (ELA)
        # -------------------------------------------------------------
        try:
            ela_sig, ela_inds, ela_dbg = self._analyze_ela(document_image, layout_regions)
            signals["ela"] = ela_sig
            signals["compression_inconsistency"] = ela_sig  # Backward-compatible alias
            indicators.extend(ela_inds)
            debug_info.update(ela_dbg)
        except Exception as e:
            logger.error(f"ELA analysis error: {str(e)}")
            signals["ela"] = TamperingSignalDetail(
                score=0.0,
                weight=self.w_ela,
                evaluated=False,
                evidence_confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE",
                summary=f"ELA encountered an error: {str(e)}"
            )
            signals["compression_inconsistency"] = signals["ela"]

        # -------------------------------------------------------------
        # Signal 2: Image Metadata / EXIF Inspection
        # -------------------------------------------------------------
        try:
            meta_sig, meta_inds, meta_dbg = self._analyze_metadata(image_bytes)
            signals["metadata"] = meta_sig
            indicators.extend(meta_inds)
            debug_info.update(meta_dbg)
        except Exception as e:
            logger.error(f"Metadata analysis error: {str(e)}")
            signals["metadata"] = TamperingSignalDetail(
                score=0.0,
                weight=self.w_metadata,
                evaluated=False,
                evidence_confidence=0.0,
                reason="NO_METADATA",
                summary=f"Metadata analysis encountered an error: {str(e)}"
            )

        # -------------------------------------------------------------
        # Signal 3: Local Noise Residual (Optional Supporting Signal)
        # -------------------------------------------------------------
        try:
            noise_sig, noise_inds = self._analyze_noise(document_image)
            signals["noise_inconsistency"] = noise_sig
            signals["noise"] = noise_sig
            indicators.extend(noise_inds)
        except Exception as e:
            logger.error(f"Noise analysis error: {str(e)}")
            signals["noise_inconsistency"] = TamperingSignalDetail(
                score=0.0,
                weight=self.w_noise,
                evaluated=False,
                evidence_confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE",
                summary=f"Noise analysis encountered an error: {str(e)}"
            )
            signals["noise"] = signals["noise_inconsistency"]

        # -------------------------------------------------------------
        # Signal 4: Document-Field Cross-Zone Semantic Consistency
        # -------------------------------------------------------------
        try:
            cons_sig, cons_inds = self._analyze_document_consistency(
                visual_fields=visual_fields,
                mrz_fields=mrz_fields,
                field_confidences=field_confidences,
                consistency_debug=consistency_debug
            )
            signals["document_consistency"] = cons_sig
            indicators.extend(cons_inds)
        except Exception as e:
            logger.error(f"Document consistency analysis error: {str(e)}")
            signals["document_consistency"] = TamperingSignalDetail(
                score=0.0,
                weight=self.w_consistency,
                evaluated=False,
                evidence_confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE",
                summary=f"Consistency analysis encountered an error: {str(e)}"
            )

        # -------------------------------------------------------------
        # Signal 5: Edge & Texture Sharpness Discontinuity
        # -------------------------------------------------------------
        try:
            edge_sig, edge_inds = self._analyze_edge_texture(document_image, layout_regions)
            signals["edge_texture"] = edge_sig
            signals["edge_texture_inconsistency"] = edge_sig
            indicators.extend(edge_inds)
        except Exception as e:
            logger.error(f"Edge/Texture analysis error: {str(e)}")
            signals["edge_texture"] = TamperingSignalDetail(
                score=0.0,
                weight=self.w_edge,
                evaluated=False,
                evidence_confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE",
                summary=f"Edge/Texture analysis encountered an error: {str(e)}"
            )
            signals["edge_texture_inconsistency"] = signals["edge_texture"]

        # -------------------------------------------------------------
        # Signal 6: Copy-Move Duplication
        # -------------------------------------------------------------
        try:
            cm_sig, cm_inds = self._analyze_copy_move(document_image)
            signals["copy_move"] = cm_sig
            indicators.extend(cm_inds)
        except Exception as e:
            logger.error(f"Copy-Move analysis error: {str(e)}")
            signals["copy_move"] = TamperingSignalDetail(
                score=0.0,
                weight=self.w_copy_move,
                evaluated=False,
                evidence_confidence=0.0,
                reason="INSUFFICIENT_EVIDENCE",
                summary=f"Copy-Move analysis encountered an error: {str(e)}"
            )

        # -------------------------------------------------------------
        # Signal Fusion
        # -------------------------------------------------------------
        fused_result = self._fuse_signals(
            signals=signals,
            indicators=indicators,
            warnings=warnings,
            consistency_debug=consistency_debug,
            debug_info=debug_info
        )
        return fused_result



    def _analyze_ela(
        self,
        image: Optional[np.ndarray],
        layout_regions: Optional[List[OCRRegion]]
    ) -> Tuple[TamperingSignalDetail, List[TamperingIndicator], Dict[str, Any]]:
        """Computes Error Level Analysis (ELA) using in-memory JPEG recompression and block variance."""
        if image is None or image.size == 0:
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_ela,
                    evaluated=False,
                    evidence_confidence=0.0,
                    reason="INSUFFICIENT_EVIDENCE",
                    summary="Image array unavailable for ELA analysis."
                ),
                [],
                {}
            )

        ela_dict = compute_error_level_analysis(
            image=image,
            quality=self.quality,
            block_size=self.block_size,
            layout_regions=layout_regions
        )

        score = float(ela_dict["score"])
        evaluated = bool(ela_dict["evaluated"])
        reason = str(ela_dict["reason"])
        summary = str(ela_dict["summary"])
        suspicious_regions = ela_dict["suspicious_regions"]
        metrics = ela_dict["metrics"]
        dbg = ela_dict.get("debug", {})

        detail = TamperingSignalDetail(
            score=score,
            weight=self.w_ela,
            evaluated=evaluated,
            evidence_confidence=0.92,
            reason=reason,
            summary=summary,
            regions=suspicious_regions,
            suspicious_regions=suspicious_regions,
            metrics=metrics
        )

        indicators: List[TamperingIndicator] = []
        if score >= self.low_threshold and suspicious_regions:
            severity = "HIGH" if score >= self.high_threshold else "MEDIUM"
            largest_dev = metrics.get("largest_deviation", 0.0)
            indicator = TamperingIndicator(
                type="ela_local_inconsistency",
                score=score,
                severity=severity,
                regions=suspicious_regions,
                explanation=(
                    f"Localized JPEG recompression residual differs significantly from surrounding document regions "
                    f"(peak deviation: +{largest_dev:.1f} intensity units)."
                ),
                details=metrics
            )
            indicators.append(indicator)

        return detail, indicators, dbg

    def _analyze_metadata(
        self,
        image_bytes: Optional[bytes]
    ) -> Tuple[TamperingSignalDetail, List[TamperingIndicator], Dict[str, Any]]:
        """Inspects EXIF and container metadata for photo editing signatures using Pillow."""
        meta_dict = extract_exif_metadata(
            image_bytes=image_bytes,
            software_keywords=self.editing_keywords
        )

        score = float(meta_dict["score"])
        evaluated = bool(meta_dict["evaluated"])
        reason = str(meta_dict["reason"])
        summary = str(meta_dict["summary"])
        editing_software_detected = bool(meta_dict.get("editing_software_detected", False))
        software = meta_dict.get("software")
        dbg = meta_dict.get("debug", {})

        detail = TamperingSignalDetail(
            score=score,
            weight=self.w_metadata,
            evaluated=evaluated,
            evidence_confidence=0.90 if evaluated else 0.0,
            reason=reason,
            summary=summary,
            regions=[],
            metrics=meta_dict.get("metadata_fields"),
            editing_software_detected=editing_software_detected,
            software=software
        )

        indicators: List[TamperingIndicator] = []
        if editing_software_detected:
            indicator = TamperingIndicator(
                type="editing_software_metadata",
                score=score,
                severity="LOW",
                regions=[],
                explanation=(
                    f"Image metadata contains a digital image editing software signature ('{software}'). "
                    f"This is a supporting risk signal and does not alone prove document alteration."
                ),
                details=meta_dict.get("metadata_fields")
            )
            indicators.append(indicator)

        return detail, indicators, dbg

    def _analyze_noise(
        self,
        image: Optional[np.ndarray]
    ) -> Tuple[TamperingSignalDetail, List[TamperingIndicator]]:
        """Computes localized sensor noise variance distribution with neutral explainable wording."""
        if image is None or image.size == 0:
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_noise,
                    evaluated=False,
                    evidence_confidence=0.0,
                    reason="INSUFFICIENT_EVIDENCE",
                    summary="Image unavailable for noise analysis."
                ),
                []
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        noise_dict = compute_noise_residual(gray, block_size=self.block_size)

        score = float(noise_dict["score"])
        evaluated = bool(noise_dict["evaluated"])
        reason = str(noise_dict["reason"])
        summary = str(noise_dict["summary"])
        regions = noise_dict.get("regions", [])
        metrics = noise_dict.get("metrics", {})

        detail = TamperingSignalDetail(
            score=score,
            weight=self.w_noise,
            evaluated=evaluated,
            evidence_confidence=0.80,
            reason=reason,
            summary=summary,
            regions=regions,
            suspicious_regions=regions,
            metrics=metrics
        )

        indicators: List[TamperingIndicator] = []
        if score >= self.low_threshold and regions:
            severity = "HIGH" if score >= self.high_threshold else "MEDIUM"
            max_dev = metrics.get("max_noise_deviation", 0.0)
            indicator = TamperingIndicator(
                type="noise_inconsistency",
                score=score,
                severity=severity,
                regions=regions,
                explanation=(
                    f"Localized regions show noise statistics different from the image baseline (peak deviation: {max_dev:.1f}). "
                    f"Differences may result from document content, scanning, compression, or editing and require corroboration."
                ),
                details=metrics
            )
            indicators.append(indicator)

        return detail, indicators

    def _analyze_document_consistency(
        self,
        visual_fields: Optional[Dict[str, Any]],
        mrz_fields: Optional[Dict[str, Any]],
        field_confidences: Optional[Dict[str, float]],
        consistency_debug: Optional[Dict[str, Any]] = None
    ) -> Tuple[TamperingSignalDetail, List[TamperingIndicator]]:
        """Compares independently extracted visual-zone fields against MRZ fields."""
        if not visual_fields or not mrz_fields:
            if consistency_debug is not None:
                consistency_debug["comparable_fields"] = []
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_consistency,
                    evaluated=False,
                    evidence_confidence=0.0,
                    reason="INSUFFICIENT_EVIDENCE",
                    summary="Cross-zone consistency not evaluated: visual or MRZ fields unavailable.",
                    metrics={"consistency_debug": consistency_debug} if consistency_debug else None
                ),
                []
            )

        field_confs = field_confidences or {}
        comparisons: Dict[str, FieldComparisonItem] = {}
        mismatches: List[Tuple[str, str, str, float]] = []

        field_pairs = [
            (["passport_number", "document_number", "visa_number", "id_number"], "document_number", normalize_doc_number_for_comparison, 0.90, "document_number", "Document / Passport Number"),
            (["date_of_birth", "dob", "birth_date"], "date_of_birth", normalize_date_for_comparison, 0.90, "date_of_birth", "Date of Birth"),
            (["date_of_expiry", "expiry_date", "expiration_date"], "date_of_expiry", normalize_date_for_comparison, 0.85, "date_of_expiry", "Date of Expiry"),
            (["name", "full_name"], "name_composite", normalize_name_for_comparison, 0.75, "name", "Full Name"),
            (["sex", "gender"], "sex", normalize_sex_for_comparison, 0.65, "sex", "Sex / Gender"),
            (["nationality", "country_code"], "nationality", normalize_nationality_for_comparison, 0.75, "nationality", "Nationality"),
        ]

        mrz_lookup = dict(mrz_fields)
        if "name_composite" not in mrz_lookup:
            sur = str(mrz_lookup.get("surname", "")).strip()
            giv = str(mrz_lookup.get("given_names", "")).strip()
            if sur or giv:
                mrz_lookup["name_composite"] = f"{sur} {giv}".strip()

        for vis_keys, mrz_key, norm_fn, mismatch_weight, canon_key, label in field_pairs:
            v_raw = ""
            v_conf = 0.85
            for vk in vis_keys:
                if vk in visual_fields and visual_fields[vk]:
                    v_raw = str(visual_fields[vk]).strip()
                    v_conf = float(field_confs.get(vk, 0.85))
                    break

            m_raw = str(mrz_lookup.get(mrz_key, "")).strip()
            m_conf = 0.98

            if not v_raw or not m_raw:
                continue

            v_norm = norm_fn(v_raw) or ""
            m_norm = norm_fn(m_raw) or ""

            if not v_norm or not m_norm:
                continue

            is_match = (v_norm == m_norm)
            if not is_match and canon_key == "name":
                v_tokens = set(v_norm.split())
                m_tokens = set(m_norm.split())
                if v_tokens and m_tokens and (v_tokens.issubset(m_tokens) or m_tokens.issubset(v_tokens)):
                    is_match = True

            comp_conf = round(min(v_conf, m_conf), 2)
            severity = "NONE"
            if not is_match:
                if v_conf >= 0.35:
                    severity = "HIGH" if mismatch_weight >= 0.85 else "MEDIUM"
                    mismatches.append((label, v_raw, m_raw, mismatch_weight))
                else:
                    severity = "LOW"

            comparisons[canon_key] = FieldComparisonItem(
                field=canon_key,
                visual=v_raw,
                mrz=m_raw,
                visual_value=v_raw,
                mrz_value=m_raw,
                normalized_visual=v_norm,
                normalized_mrz=m_norm,
                match=is_match,
                visual_confidence=round(v_conf, 2),
                mrz_confidence=round(m_conf, 2),
                comparison_confidence=comp_conf,
                confidence_visual=round(v_conf, 2),
                confidence_mrz=round(m_conf, 2),
                mismatch_severity=severity,
                note=None if is_match else f"Visual printed value '{v_raw}' contradicts MRZ value '{m_raw}'."
            )

        if not comparisons:
            if consistency_debug is not None:
                consistency_debug["comparable_fields"] = []
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_consistency,
                    evaluated=False,
                    evidence_confidence=0.0,
                    reason="INSUFFICIENT_EVIDENCE",
                    summary="Cross-zone consistency not evaluated: visual or MRZ fields unavailable.",
                    metrics={"consistency_debug": consistency_debug} if consistency_debug else None
                ),
                []
            )

        if consistency_debug is not None:
            consistency_debug["comparable_fields"] = list(comparisons.keys())

        indicators: List[TamperingIndicator] = []
        if mismatches:
            max_mismatch_weight = max(m[3] for m in mismatches)
            score = round(min(1.0, max_mismatch_weight), 4)
            severity = "HIGH" if score >= self.high_threshold else "MEDIUM"

            reasons_text = "; ".join([f"{label}: Visual '{vr}' != MRZ '{mr}'" for label, vr, mr, _ in mismatches])
            summary = f"Detected {len(mismatches)} cross-zone field mismatch(es) between visual printed text and MRZ: {reasons_text}."

            indicator = TamperingIndicator(
                type="document_consistency_mismatch",
                score=score,
                severity=severity,
                regions=[],
                explanation=(
                    f"Strong semantic inconsistency detected: {summary} "
                    f"This indicates probable document alteration where printed visual information was modified without matching MRZ update."
                ),
                details={"mismatches": [{"field": l, "visual": vr, "mrz": mr} for l, vr, mr, _ in mismatches]}
            )
            indicators.append(indicator)
            reason = "EVALUATED"
        else:
            score = 0.0
            summary = f"All {len(comparisons)} evaluated fields are perfectly consistent between visual text and MRZ."
            reason = "NO_ANOMALY_FOUND"

        detail = TamperingSignalDetail(
            score=score,
            weight=self.w_consistency,
            evaluated=True,
            evidence_confidence=0.95,
            reason=reason,
            summary=summary,
            comparisons=comparisons,
            metrics={"consistency_debug": consistency_debug} if consistency_debug else None
        )
        return detail, indicators

    def _analyze_edge_texture(
        self,
        image: Optional[np.ndarray],
        layout_regions: Optional[List[OCRRegion]] = None
    ) -> Tuple[TamperingSignalDetail, List[TamperingIndicator]]:
        """Evaluates localized edge sharpness and gradient anomalies."""
        if image is None or image.size == 0:
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_edge,
                    evaluated=False,
                    evidence_confidence=0.0,
                    reason="INSUFFICIENT_EVIDENCE",
                    summary="Image unavailable for edge/texture analysis."
                ),
                []
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        grad_mag, lap_abs = compute_local_gradient_and_sharpness(gray)
        h, w = gray.shape[:2]
        bs = self.block_size
        n_by = max(1, h // bs)
        n_bx = max(1, w // bs)

        block_laps = []
        for by in range(n_by):
            for bx in range(n_bx):
                y1 = by * bs
                x1 = bx * bs
                y2 = min(h, y1 + bs)
                x2 = min(w, x1 + bs)
                tile = lap_abs[y1:y2, x1:x2]
                block_laps.append(float(np.mean(tile)))

        arr_laps = np.array(block_laps, dtype=np.float32)
        med_lap = float(np.median(arr_laps))
        mad_lap = float(np.median(np.abs(arr_laps - med_lap)))
        min_lap = float(np.min(arr_laps)) if len(arr_laps) > 0 else 0.0

        # Check for localized smudging/blurring (abnormally low Laplacian sharpness compared to text baseline)
        if med_lap > 8.0 and (med_lap - min_lap) > 6.0 and min_lap < 3.0:
            score = 0.25
            reason = "EVALUATED"
            summary = "Localized blur or smudged region detected with sharpness lower than document baseline."
            indicators = [
                TamperingIndicator(
                    type="edge_texture_inconsistency",
                    score=score,
                    severity="LOW",
                    regions=[],
                    explanation="Localized area exhibits significantly reduced edge sharpness / blur relative to document content.",
                    details={"median_laplacian": med_lap, "min_laplacian": min_lap}
                )
            ]
        else:
            score = 0.0
            reason = "NO_ANOMALY_FOUND"
            summary = "Consistent edge and texture sharpness across document."
            indicators = []

        detail = TamperingSignalDetail(
            score=score,
            weight=self.w_edge,
            evaluated=True,
            evidence_confidence=0.75,
            reason=reason,
            summary=summary,
            metrics={"median_laplacian": round(med_lap, 2), "min_laplacian": round(min_lap, 2)}
        )
        return detail, indicators

    def _analyze_copy_move(
        self,
        image: Optional[np.ndarray]
    ) -> Tuple[TamperingSignalDetail, List[TamperingIndicator]]:
        """Detects duplicated/cloned regions within the document using ORB feature matching."""
        if image is None or image.size == 0:
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_copy_move,
                    evaluated=False,
                    evidence_confidence=0.0,
                    reason="INSUFFICIENT_EVIDENCE",
                    summary="Image unavailable for copy-move analysis."
                ),
                []
            )

        clusters = detect_orb_copy_move_clusters(image)
        if not clusters:
            return (
                TamperingSignalDetail(
                    score=0.0,
                    weight=self.w_copy_move,
                    evaluated=True,
                    evidence_confidence=0.85,
                    reason="NO_ANOMALY_FOUND",
                    summary="No duplicated feature clusters detected."
                ),
                []
            )

        total_clusters = len(clusters)
        max_matches = max(c["match_count"] for c in clusters)

        if max_matches >= 12 or total_clusters >= 2:
            score = 0.80
            severity = "HIGH"
        elif max_matches >= 6:
            score = 0.55
            severity = "MEDIUM"
        else:
            score = 0.30
            severity = "LOW"

        regions = []
        for cl in clusters[:3]:
            regions.append(cl["source_region"])
            regions.append(cl["target_region"])

        indicator = TamperingIndicator(
            type="copy_move",
            score=score,
            severity=severity,
            regions=regions,
            explanation=(
                f"Detected {total_clusters} duplicated region pair(s) with matching feature clusters "
                f"(peak {max_matches} coherent point matches with consistent spatial displacement), "
                f"indicating probable copy-move cloning."
            ),
            details={"cluster_count": total_clusters, "peak_match_count": max_matches}
        )

        detail = TamperingSignalDetail(
            score=score,
            weight=self.w_copy_move,
            evaluated=True,
            evidence_confidence=0.90,
            reason="EVALUATED",
            summary=f"Detected {total_clusters} copy-move cluster(s) with up to {max_matches} matching point pairs.",
            regions=regions,
            suspicious_regions=regions,
            metrics={"cluster_count": total_clusters, "peak_matches": max_matches}
        )
        return detail, [indicator]


    def _fuse_signals(
        self,
        signals: Dict[str, TamperingSignalDetail],
        indicators: List[TamperingIndicator],
        warnings: List[str],
        consistency_debug: Optional[Dict[str, Any]] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ) -> TamperingResult:
        """Computes weighted overall risk score, evidence coverage, and maps to categorical risk levels with corroboration."""
        # Core unique signal keys to prevent double-counting aliases
        primary_keys = ["ela", "metadata", "noise", "edge_texture", "copy_move"]
        if "document_consistency" in signals and signals["document_consistency"].evaluated:
            primary_keys.append("document_consistency")

        total_configured_weight = 0.0
        evaluated_weight = 0.0
        weighted_sum = 0.0

        for key in primary_keys:
            if key in signals:
                sig = signals[key]
                total_configured_weight += sig.weight
                if sig.evaluated:
                    evaluated_weight += sig.weight
                    weighted_sum += (sig.weight * sig.score)

        if total_configured_weight > 0.0:
            evidence_coverage = round(min(1.0, evaluated_weight / total_configured_weight), 4)
        else:
            evidence_coverage = 1.0

        if evaluated_weight > 0.0:
            raw_risk = weighted_sum / evaluated_weight
        else:
            raw_risk = 0.0

        risk_score = round(min(1.0, max(0.0, raw_risk)), 4)

        # Count elevated independent signals (score >= low_threshold)
        elevated_signals = [
            sig for k, sig in signals.items()
            if k in primary_keys and sig.evaluated and sig.score >= self.low_threshold
        ]
        has_very_strong_signal = any(sig.score >= 0.80 for sig in elevated_signals)

        if risk_score >= self.high_threshold:
            # Corroboration rule: HIGH risk requires >= 2 elevated signals or 1 strong signal (>= 0.80)
            if len(elevated_signals) >= 2 or has_very_strong_signal:
                risk_level = "HIGH"
            else:
                risk_level = "MEDIUM"
        elif risk_score >= self.low_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Sort indicators by score descending
        indicators.sort(key=lambda ind: -ind.score)

        return TamperingResult(
            tampering_risk_score=risk_score,
            risk_level=risk_level,
            evidence_coverage=evidence_coverage,
            signals=signals,
            indicators=indicators,
            warnings=warnings,
            consistency_debug=consistency_debug,
            debug=debug_info,
            disclaimer="Tampering risk represents image-forensic anomalies and does not constitute absolute proof of forgery or authenticity."
        )

