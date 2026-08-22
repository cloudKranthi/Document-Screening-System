"""OCR extraction and document verification route."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

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

    # Step 13: OCR Execution
    try:
        ocr_result = ocr_srv.extract(ocr_optimized_img)
    except Exception as e:
        logger.error(f"OCR execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR engine failure: {str(e)}"
        )

    # Step 14: Passport MRZ Detection & ICAO 9303 Check Digit Validation
    region_texts = [r.text for r in ocr_result.regions]
    mrz_result, mrz_fields = MRZService.extract_and_validate_mrz(ocr_result.raw_text, region_texts)

    # Step 15: Document Classification & Field Extraction
    effective_doc_type, extracted_fields = doc_srv.process_extraction(
        requested_type=document_type,
        ocr_text=ocr_result.raw_text,
        mrz_result=mrz_result,
        mrz_fields=mrz_fields
    )

    logger.info(f"Extraction successful: document_type={effective_doc_type}, mrz_detected={mrz_result.detected}, avg_conf={ocr_result.average_confidence}")

    return OCRExtractResponse(
        success=True,
        document_type=effective_doc_type,
        average_confidence=ocr_result.average_confidence,
        extracted_text=ocr_result.raw_text,
        fields=extracted_fields,
        mrz=mrz_result,
        ocr_regions=ocr_result.regions,
        processing=processing_meta
    )
