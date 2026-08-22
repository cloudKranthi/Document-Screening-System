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
