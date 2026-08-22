"""Unit tests for image processing, boundary detection, and enhancements."""

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from app.services.image_service import ImageService
from app.utils.image_utils import (
    deskew_image,
    detect_document_contour,
    enhance_for_ocr,
    four_point_transform,
    order_points,
    resize_image_if_needed,
)


class TestImageUtils:
    """Tests for OpenCV low-level image utilities."""

    def test_order_points(self):
        # 4 unordered points representing a rectangle: (10, 10), (100, 10), (100, 50), (10, 50)
        pts = np.array([[100, 50], [10, 10], [100, 10], [10, 50]], dtype="float32")
        ordered = order_points(pts)
        
        # [tl, tr, br, bl]
        np.testing.assert_array_equal(ordered[0], [10, 10])   # top-left
        np.testing.assert_array_equal(ordered[1], [100, 10])  # top-right
        np.testing.assert_array_equal(ordered[2], [100, 50])  # bottom-right
        np.testing.assert_array_equal(ordered[3], [10, 50])   # bottom-left

    def test_four_point_transform(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a white inner box
        cv2.rectangle(img, (20, 20), (180, 150), (255, 255, 255), -1)
        
        pts = np.array([[20, 20], [180, 20], [180, 150], [20, 150]], dtype="float32")
        warped = four_point_transform(img, pts)
        
        assert warped is not None
        assert warped.shape[0] > 100
        assert warped.shape[1] > 100

    def test_resize_image_if_needed(self):
        large_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        resized, scale = resize_image_if_needed(large_img, max_dim=2000)
        assert max(resized.shape[:2]) == 2000
        assert scale == 2000 / 4000

    def test_enhance_for_ocr(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        enhanced = enhance_for_ocr(img)
        assert len(enhanced.shape) == 2  # Grayscale
        assert enhanced.shape[:2] == (200, 200)

    def test_detect_document_contour_synthetic_quad(self, create_perspective_tilted_card):
        img_bytes = create_perspective_tilted_card()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        quad = detect_document_contour(img, min_area_ratio=0.10)
        assert quad is not None
        assert len(quad) == 4

    def test_detect_document_contour_fallback_blank(self):
        blank_img = np.ones((500, 500, 3), dtype=np.uint8) * 200
        quad = detect_document_contour(blank_img)
        assert quad is None


class TestImageService:
    """Tests for ImageService pipeline."""

    def test_decode_valid_image(self, create_dummy_image):
        img_bytes = create_dummy_image()
        decoded = ImageService.decode_image(img_bytes)
        assert decoded is not None
        assert decoded.shape[0] == 600
        assert decoded.shape[1] == 800

    def test_decode_invalid_corrupted_image(self):
        corrupted_bytes = b"NOT_A_VALID_IMAGE_DATA_12345"
        with pytest.raises(HTTPException) as exc_info:
            ImageService.decode_image(corrupted_bytes)
        assert exc_info.value.status_code == 400
        assert "Corrupted or invalid image" in exc_info.value.detail

    def test_process_document_image_pipeline(self, create_dummy_image):
        img_bytes = create_dummy_image(text="PASSPORT TEST")
        orig, ocr_opt, meta = ImageService.process_document_image(img_bytes)
        
        assert orig is not None
        assert ocr_opt is not None
        assert meta.preprocessing_applied is True
        assert meta.original_dimensions == [800, 600]
        assert meta.processed_dimensions is not None

    def test_mrz_region_crop_and_preprocessing_variants(self, create_dummy_image):
        img_bytes = create_dummy_image(text="MRZ BOTTOM TEST")
        decoded = ImageService.decode_image(img_bytes)
        
        variants = ImageService.get_mrz_preprocessed_variants(decoded, bottom_ratio=0.35)
        assert len(variants) == 4
        
        var_names = [v[0] for v in variants]
        assert "clahe" in var_names
        assert "otsu" in var_names
        assert "adaptive" in var_names
        assert "blackhat" in var_names
        
        for name, img in variants:
            assert img is not None
            # Verify upscale factor was applied
            assert img.shape[1] == int(decoded.shape[1] * 3.0)
            assert len(img.shape) == 2  # Grayscale/Binarized

    def test_get_all_mrz_crop_variants(self, create_dummy_image):
        img_bytes = create_dummy_image(text="MULTI CROP TEST")
        decoded = ImageService.decode_image(img_bytes)
        
        all_crops = ImageService.get_all_mrz_crop_variants(decoded, ratios=[0.20, 0.35], scale_factor=3.0)
        assert len(all_crops) == 2 * 4  # 2 ratios x 4 variants
        for label, ratio, img in all_crops:
            assert ratio in [0.20, 0.35]
            assert img is not None

    def test_get_field_candidate_crops_and_preprocessing(self, create_dummy_image):
        """Test generation of targeted dynamic ROI crops and preprocessing variants around a field label."""
        img_bytes = create_dummy_image(text="FIELD CROP TEST")
        decoded = ImageService.decode_image(img_bytes)
        
        label_bbox = [50, 100, 100, 130] # label [lx1, ly1, lx2, ly2]
        field_variants = ImageService.get_field_crop_variants(decoded, label_bbox=label_bbox, field_type="dob", scale_factor=3.0)
        
        assert len(field_variants) > 0
        crop_names = [v[0] for v in field_variants]
        prep_names = [v[1] for v in field_variants]
        
        assert any("right" in c for c in crop_names)
        assert any("below" in c for c in crop_names)
        assert "clahe_sharp" in prep_names
        assert "otsu" in prep_names
        assert "adaptive" in prep_names
        
        for crop_name, prep_name, prep_img, crop_bbox in field_variants:
            assert prep_img is not None
            assert len(prep_img.shape) == 2
            assert prep_img.shape[0] > 10
            assert prep_img.shape[1] > 10



