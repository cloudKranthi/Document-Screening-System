from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.config import settings
from app.models.schemas import DocumentTypeEnum, OCRExtractResponse
from app.services.confidence_service import ConfidenceService
from app.services.document_service import DocumentService
from app.services.image_service import ImageService
from app.services.mrz_service import MRZService
from app.services.ocr_service import OCRService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR Extraction"])

# Instantiate service singletons
ocr_service = OCRService()
document_service = DocumentService()


def get_ocr_service() -> OCRService:
    return ocr_service


def get_document_service() -> DocumentService:
    return document_service


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
    ocr_srv: OCRService = Depends(get_ocr_service),
    doc_srv: DocumentService = Depends(get_document_service),
) -> OCRExtractResponse:
    """Processes uploaded document image and returns extracted fields and verification metadata."""
    logger.info(f"Received extraction request for file: {file.filename} with requested type: {document_type}")

    # Validate document_type parameter
    valid_types = [t.value for t in DocumentTypeEnum]
    if document_type.lower() not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document_type '{document_type}'. Supported values: {', '.join(valid_types)}"
        )

    # Step 1 & 2: Safely validate and read file contents
    raw_bytes = await ImageService.validate_and_read_upload(file)

    # Step 3 to 12: Image preprocessing pipeline (decode, resize, boundary detect, warp, enhance)
    original_img, ocr_optimized_img, processing_meta = ImageService.process_document_image(raw_bytes)

    # Step 13: General OCR Execution (Full Document)
    try:
        ocr_result = ocr_srv.extract(ocr_optimized_img)
    except Exception as e:
        logger.error(f"General OCR execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR engine failure: {str(e)}"
        )

    # Step 14: Multi-Ratio Crop & Multi-Variant Dedicated MRZ OCR Execution
    # Evaluates bottom 20%, 25%, 30%, 35%, 40% crops across CLAHE, Otsu, Adaptive, and Blackhat variants
    mrz_candidate_sources: list[tuple[str, str]] = []
    mrz_regions_combined = []
    
    try:
        crop_variants = ImageService.get_all_mrz_crop_variants(
            ocr_optimized_img,
            ratios=settings.MRZ_CROP_RATIOS,
            scale_factor=settings.MRZ_UPSCALE_FACTOR
        )
        
        # Test configurations: PSM 6 (uniform block), PSM 4 (single column), PSM 11 (sparse text)
        psm_modes = [6, 4, 11]
        
        for var_name, ratio, var_img in crop_variants:
            for psm in psm_modes:
                src_label = f"{var_name}_psm{psm}"
                try:
                    mrz_ocr_res = ocr_srv.extract_mrz(var_img, psm=psm)
                    if mrz_ocr_res.raw_text.strip():
                        mrz_candidate_sources.append((src_label, mrz_ocr_res.raw_text))
                        if mrz_ocr_res.regions:
                            mrz_regions_combined.extend(mrz_ocr_res.regions)
                except Exception as mrz_err:
                    logger.warning(f"MRZ OCR pass ({src_label}) failed: {str(mrz_err)}")
    except Exception as e:
        logger.warning(f"MRZ crop candidate generation failed: {str(e)}")

    # Step 15: Passport MRZ Candidate Scoring, Detection & ICAO 9303 Check Digit Validation
    region_texts = [r.text for r in ocr_result.regions]
    should_debug = include_debug or settings.DEBUG
    
    mrz_result, mrz_fields, mrz_debug_info = MRZService.extract_and_validate_mrz(
        raw_ocr_text=ocr_result.raw_text,
        ocr_lines=region_texts,
        mrz_candidate_texts=mrz_candidate_sources,
        include_debug=should_debug
    )

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
        mrz_debug=mrz_debug_info if should_debug else None,
        field_debug=field_debug_info if should_debug else None
    )






