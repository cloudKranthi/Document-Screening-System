from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.config import settings
from app.models.schemas import DocumentTypeEnum, OCRExtractResponse, TamperingResult
from app.services.confidence_service import ConfidenceService
from app.services.document_service import DocumentService
from app.services.image_service import ImageService
from app.services.mrz_service import MRZService
from app.services.ocr_service import OCRService
from app.services.tampering_service import TamperingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR Extraction & Document Screening"])

# Instantiate service singletons
ocr_service = OCRService()
mrz_service = MRZService()
document_service = DocumentService()
tampering_service = TamperingService()


def get_ocr_service() -> OCRService:
    return ocr_service


def get_mrz_service() -> MRZService:
    return mrz_service


def get_document_service() -> DocumentService:
    return document_service


def get_tampering_service() -> TamperingService:
    return tampering_service



@router.post(
    "/extract",
    response_model=OCRExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract and validate identity document information",
    description="Accepts an identity document image (Passport, Visa, National ID), preprocesses it, extracts OCR text, parses and validates MRZ check digits if present, and returns structured key-value fields."
)
async def extract_document_info(
    file: UploadFile = File(..., description="Identity document image file (JPG, PNG, WebP, BMP, TIFF)"),
    document_type: str = Form(
        default="auto",
        description="Target document type: 'passport', 'visa', 'national_id', or 'auto' for automatic detection."
    ),
    include_debug: bool = Form(
        default=False,
        description="Whether to include MRZ candidate scores and pipeline debug metadata in response."
    ),
    detect_tampering: bool = Form(
        default=False,
        description="Whether to run multi-signal tampering detection analysis on the document."
    ),
    ocr_srv: OCRService = Depends(get_ocr_service),
    doc_srv: DocumentService = Depends(get_document_service),
    tampering_srv: TamperingService = Depends(get_tampering_service),
) -> OCRExtractResponse:
    """Processes uploaded document image and returns extracted fields and verification metadata."""
    logger.info(f"Received extraction request for file: {file.filename} with requested type: {document_type}, tampering: {detect_tampering}")

    # Validate document_type parameter
    valid_types = [t.value for t in DocumentTypeEnum]
    if document_type.lower() not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document_type '{document_type}'. Supported values: {', '.join(valid_types)}"
        )

    # Step 1 & 2: Safely validate and read file contents
    raw_bytes = await ImageService.validate_and_read_upload(file)

    def _sync_extract():
        import time
        # Step 3 to 12: Image preprocessing pipeline (decode, resize, boundary detect, warp, enhance)
        t_pre_start = time.perf_counter()
        original_img, ocr_optimized_img, processing_meta = ImageService.process_document_image(raw_bytes)
        t_pre = (time.perf_counter() - t_pre_start) * 1000.0
        logger.info(f"[TIMING] preprocess completed in {t_pre:.2f}ms")

        # Step 13: General OCR Execution (Full Document)
        try:
            t_prim_start = time.perf_counter()
            ocr_result = ocr_srv.extract(ocr_optimized_img)
            t_prim = (time.perf_counter() - t_prim_start) * 1000.0
            logger.info(f"[TIMING] primary OCR completed in {t_prim:.2f}ms (conf: {ocr_result.average_confidence:.3f})")
        except Exception as e:
            logger.error(f"General OCR execution failed: {str(e)}")
            raise e

        region_texts = [r.text for r in ocr_result.regions]
        should_debug = include_debug or settings.DEBUG

        # Step 14: Staged MRZ Candidate Generation & ICAO 9303 Check Digit Validation
        # Stage 1: Initial evaluation on full-document OCR
        mrz_result, mrz_fields, mrz_debug_info = MRZService.extract_and_validate_mrz(
            raw_ocr_text=ocr_result.raw_text,
            ocr_lines=region_texts,
            mrz_candidate_texts=[],
            include_debug=should_debug
        )

        mrz_candidate_sources: list[tuple[str, str]] = []

        # Stage 2: Single targeted MRZ crop only if MRZ not detected or invalid
        if not (mrz_result.detected and mrz_result.overall_valid):
            t_mrz_start = time.perf_counter()
            try:
                from app.utils.image_utils import extract_mrz_region, preprocess_mrz_crop
                target_crop = extract_mrz_region(ocr_optimized_img, bottom_ratio=0.35)
                prep_variants = preprocess_mrz_crop(target_crop, scale_factor=settings.MRZ_UPSCALE_FACTOR)
                if prep_variants:
                    clahe_name, clahe_img = prep_variants[0]
                    try:
                        mrz_ocr_res = ocr_srv.extract_mrz(clahe_img, psm=6)
                        t_mrz = (time.perf_counter() - t_mrz_start) * 1000.0
                        logger.info(f"[TIMING] MRZ OCR completed in {t_mrz:.2f}ms")
                        if mrz_ocr_res.raw_text.strip():
                            mrz_candidate_sources.append((f"crop_0.35_{clahe_name}_psm6", mrz_ocr_res.raw_text))
                            mrz_result, mrz_fields, mrz_debug_info = MRZService.extract_and_validate_mrz(
                                raw_ocr_text=ocr_result.raw_text,
                                ocr_lines=region_texts,
                                mrz_candidate_texts=mrz_candidate_sources,
                                include_debug=should_debug
                            )
                    except TimeoutError as te:
                        logger.warning(f"Targeted MRZ OCR pass timed out: {te}")
                    except Exception as e:
                        logger.warning(f"Targeted MRZ crop OCR failed: {e}")
            except Exception as crop_err:
                logger.warning(f"Targeted MRZ crop generation error: {crop_err}")

        # Stage 3: Limited fallback variants only if MRZ is STILL not valid
        if not (mrz_result.detected and mrz_result.overall_valid):
            from app.utils.image_utils import extract_mrz_region, preprocess_mrz_crop
            max_fallbacks = getattr(settings, "MAX_MRZ_FALLBACK_ATTEMPTS", 2)
            fallback_attempts = 0
            for ratio in [0.30, 0.40]:
                if fallback_attempts >= max_fallbacks or (mrz_result.detected and mrz_result.overall_valid):
                    break
                try:
                    crop = extract_mrz_region(ocr_optimized_img, bottom_ratio=ratio)
                    variants = preprocess_mrz_crop(crop, scale_factor=settings.MRZ_UPSCALE_FACTOR)
                    for v_name, v_img in variants[1:]:  # Otsu, adaptive
                        if fallback_attempts >= max_fallbacks:
                            break
                        fallback_attempts += 1
                        t_fb_start = time.perf_counter()
                        try:
                            mrz_ocr_res = ocr_srv.extract_mrz(v_img, psm=6)
                            t_fb = (time.perf_counter() - t_fb_start) * 1000.0
                            logger.info(f"[TIMING] fallback OCR attempt #{fallback_attempts} completed in {t_fb:.2f}ms")
                            if mrz_ocr_res.raw_text.strip():
                                mrz_candidate_sources.append((f"fallback_{ratio}_{v_name}_psm6", mrz_ocr_res.raw_text))
                                mrz_result, mrz_fields, mrz_debug_info = MRZService.extract_and_validate_mrz(
                                    raw_ocr_text=ocr_result.raw_text,
                                    ocr_lines=region_texts,
                                    mrz_candidate_texts=mrz_candidate_sources,
                                    include_debug=should_debug
                                )
                                if mrz_result.detected and mrz_result.overall_valid:
                                    break
                        except TimeoutError as te:
                            logger.warning(f"Fallback OCR pass timed out: {te}")
                        except Exception:
                            pass
                except Exception:
                    pass




        # Step 16: Merge OCR outputs & regions (Preserving both general and MRZ outputs)
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

        # Step 17: Document Classification & Field Extraction
        effective_doc_type, extracted_fields, field_confs, extraction_warnings, field_debug_info, field_sources_info = doc_srv.process_extraction(
            requested_type=document_type,
            ocr_text=combined_text,
            mrz_result=mrz_result,
            mrz_fields=mrz_fields,
            ocr_regions=all_regions,
            document_image=ocr_optimized_img,
            ocr_service=ocr_srv,
            include_debug=should_debug
        )

        # Step 18: Optional Tampering Detection
        tampering_res = None
        if detect_tampering:
            try:
                vis_fields_to_pass = getattr(doc_srv, "last_visual_fields", None) or extracted_fields
                vis_confs_to_pass = getattr(doc_srv, "last_visual_confs", None) or field_confs
                tampering_res = tampering_srv.analyze_document(
                    image_bytes=raw_bytes,
                    document_image=original_img,
                    visual_fields=vis_fields_to_pass,
                    mrz_fields=mrz_fields,
                    field_confidences=vis_confs_to_pass,
                    layout_regions=all_regions
                )
            except Exception as e:
                logger.error(f"Tampering detection failed: {str(e)}")

        logger.info(f"Extraction completed: document_type={effective_doc_type}, mrz_detected={mrz_result.detected}, avg_conf={avg_conf}")

        return OCRExtractResponse(
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
            warnings=extraction_warnings,
            tampering=tampering_res,
            mrz_debug=mrz_debug_info if should_debug else None,
            field_debug=field_debug_info if should_debug else None
        )

    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_sync_extract)


@router.post(
    "/tampering",
    response_model=TamperingResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze document image for digital tampering & manipulation",
    description="Evaluates 6 independent forensic signals (Cross-zone semantic field consistency, JPEG ELA compression inconsistency, noise distribution, edge/texture sharpness discontinuity, copy-move duplication, metadata editor tags) and returns an explainable risk assessment."
)
async def analyze_document_tampering(
    file: UploadFile = File(..., description="Identity document image file (JPG, PNG, WebP, BMP, TIFF)"),
    tampering_srv: TamperingService = Depends(get_tampering_service),
    ocr_srv: OCRService = Depends(get_ocr_service),
    mrz_srv: MRZService = Depends(get_mrz_service),
    doc_srv: DocumentService = Depends(get_document_service),
) -> TamperingResult:

    """Standalone endpoint for document tampering and forgery analysis."""
    logger.info(f"Received tampering analysis request for file: {file.filename}")
    raw_bytes = await ImageService.validate_and_read_upload(file)

    def _sync_tampering():
        original_img, ocr_optimized_img, _ = ImageService.process_document_image(raw_bytes)

        # 1. Run OCR and MRZ extraction to enable cross-zone consistency analysis
        vis_fields_to_pass = {}
        vis_confs_to_pass = {}
        mrz_fields = {}
        all_regions = []

        try:
            ocr_result = ocr_srv.extract(ocr_optimized_img)
            all_regions = list(ocr_result.regions)

            mrz_candidate_sources = []
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
                            mrz_ocr_res = ocr_srv.extract_mrz(var_img, psm=psm)
                            if mrz_ocr_res.raw_text.strip():
                                mrz_candidate_sources.append((src_label, mrz_ocr_res.raw_text))
                        except Exception:
                            pass
            except Exception:
                pass

            region_texts = [r.text for r in ocr_result.regions]
            mrz_result, mrz_fields, _ = MRZService.extract_and_validate_mrz(
                raw_ocr_text=ocr_result.raw_text,
                ocr_lines=region_texts,
                mrz_candidate_texts=mrz_candidate_sources,
                include_debug=False
            )

            if mrz_result.detected and mrz_result.line1 and mrz_result.line2:
                mrz_lines = [mrz_result.line1, mrz_result.line2]
                if mrz_result.line3:
                    mrz_lines.append(mrz_result.line3)
                mrz_block = "\n".join(mrz_lines)
                combined_text = ocr_result.raw_text + "\n" + mrz_block if mrz_result.line1 not in ocr_result.raw_text else ocr_result.raw_text
            else:
                combined_text = ocr_result.raw_text

            _, extracted_fields, field_confs, _, _, _ = doc_srv.process_extraction(
                requested_type="auto",
                ocr_text=combined_text,
                mrz_result=mrz_result,
                mrz_fields=mrz_fields,
                ocr_regions=all_regions,
                document_image=ocr_optimized_img,
                ocr_service=ocr_srv,
                include_debug=False
            )

            vis_fields_to_pass = getattr(doc_srv, "last_visual_fields", None) or extracted_fields
            vis_confs_to_pass = getattr(doc_srv, "last_visual_confs", None) or field_confs
        except Exception as e:
            logger.warning(f"OCR/MRZ extraction during tampering analysis skipped or failed: {str(e)}")

        result = tampering_srv.analyze_document(
            image_bytes=raw_bytes,
            document_image=original_img,
            visual_fields=vis_fields_to_pass,
            mrz_fields=mrz_fields,
            field_confidences=vis_confs_to_pass,
            layout_regions=all_regions
        )
        return result

    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_sync_tampering)










