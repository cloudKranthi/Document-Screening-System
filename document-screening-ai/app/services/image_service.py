"""Image processing service handling ingestion, validation, enhancement, and 4-point perspective cropping."""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.models.schemas import ProcessingMetadata
from app.utils.image_utils import (
    deskew_image,
    detect_document_contour,
    enhance_for_ocr,
    extract_mrz_region,
    four_point_transform,
    get_field_candidate_crops,
    get_mrz_candidate_crops,
    preprocess_field_crop,
    preprocess_mrz_crop,
    resize_image_if_needed,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ImageService:
    """Service for safe ingestion, preprocessing, boundary detection, and cropping of document images."""

    @staticmethod
    async def validate_and_read_upload(file: UploadFile) -> bytes:
        """Validates uploaded file size and content type, and safely reads bytes."""
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename."
            )

        # Check extension
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Read contents
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
            )

        return contents

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Decodes raw bytes into an OpenCV BGR numpy image safely."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError("Decoded image is empty or invalid.")
            return img
        except Exception as e:
            logger.error(f"Image decoding failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted or invalid image file. Unable to decode."
            )

    @classmethod
    def process_document_image(cls, image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, ProcessingMetadata]:
        """Executes the full image processing pipeline.
        
        Returns:
            Tuple of:
              - original_image: unmodified RGB/BGR numpy array
              - ocr_optimized_image: enhanced, cropped/deskewed grayscale image for OCR
              - metadata: ProcessingMetadata describing transformations applied
        """
        original_img = cls.decode_image(image_bytes)
        orig_h, orig_w = original_img.shape[:2]
        
        # Resize if dimensions exceed threshold
        resized_img, _ = resize_image_if_needed(
            original_img,
            max_dim=settings.MAX_IMAGE_DIMENSION,
            min_dim=settings.MIN_IMAGE_DIMENSION
        )
        
        # Detect 4-point document boundaries
        quad_points = detect_document_contour(resized_img)
        boundary_detected = quad_points is not None
        crop_success = False
        
        if boundary_detected:
            try:
                cropped = four_point_transform(resized_img, quad_points)
                # Ensure cropped region is not degenerate
                if cropped.shape[0] > 50 and cropped.shape[1] > 50:
                    working_img = cropped
                    crop_success = True
                else:
                    working_img = resized_img
            except Exception as e:
                logger.warning(f"Perspective crop failed, falling back to original: {str(e)}")
                working_img = resized_img
        else:
            working_img = resized_img
            
        # Enhance for OCR: Grayscale, Bilateral noise reduction, CLAHE contrast enhancement
        enhanced = enhance_for_ocr(
            working_img,
            clip_limit=settings.CLAHE_CLIP_LIMIT,
            tile_grid_size=settings.CLAHE_TILE_GRID_SIZE
        )
        
        # Deskew if needed
        ocr_optimized = deskew_image(enhanced)
        
        proc_h, proc_w = ocr_optimized.shape[:2]
        
        metadata = ProcessingMetadata(
            crop_success=crop_success,
            preprocessing_applied=True,
            original_dimensions=[orig_w, orig_h],
            processed_dimensions=[proc_w, proc_h],
            boundary_detected=boundary_detected
        )
        
        return original_img, ocr_optimized, metadata

    @classmethod
    def get_mrz_preprocessed_variants(cls, image: np.ndarray, bottom_ratio: float = 0.35) -> list[tuple[str, np.ndarray]]:
        """Extracts the bottom MRZ region and returns multiple preprocessed image variants."""
        mrz_crop = extract_mrz_region(image, bottom_ratio=bottom_ratio)
        return preprocess_mrz_crop(mrz_crop, scale_factor=settings.MRZ_UPSCALE_FACTOR)

    @classmethod
    def get_all_mrz_crop_variants(
        cls,
        image: np.ndarray,
        ratios: Optional[list[float]] = None,
        scale_factor: float = 3.0
    ) -> list[tuple[str, float, np.ndarray]]:
        """Generates all combinations of crop ratios and preprocessing variants.
        
        Returns:
            List of tuples: (variant_name, ratio, processed_image)
        """
        if ratios is None:
            ratios = settings.MRZ_CROP_RATIOS
            
        candidate_crops = get_mrz_candidate_crops(image, ratios=ratios)
        all_variants = []
        for ratio, crop in candidate_crops:
            variants = preprocess_mrz_crop(crop, scale_factor=scale_factor)
            for var_name, var_img in variants:
                all_variants.append((f"{var_name}_r{int(ratio*100)}", ratio, var_img))
        return all_variants

    @classmethod
    def get_field_crop_variants(
        cls,
        image: np.ndarray,
        label_bbox: list[int],
        field_type: str = "dob",
        scale_factor: float = 3.0
    ) -> list[tuple[str, str, np.ndarray, list[int]]]:
        """Generates all combinations of field candidate crop geometries and preprocessing variants.
        
        Returns:
            List of tuples: (crop_name, prep_name, preprocessed_image, crop_bbox)
        """
        candidate_crops = get_field_candidate_crops(image, label_bbox=label_bbox, field_type=field_type)
        all_variants = []
        for crop_name, crop_img, crop_bbox in candidate_crops:
            preprocessed_list = preprocess_field_crop(crop_img, scale_factor=scale_factor)
            for prep_name, prep_img in preprocessed_list:
                all_variants.append((crop_name, prep_name, prep_img, crop_bbox))
        return all_variants



