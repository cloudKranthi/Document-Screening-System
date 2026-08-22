"""Pytest fixtures and test helpers for SIH OCR Microservice."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
from app.models.schemas import OCRRegion


@pytest.fixture
def test_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def valid_td3_mrz_sample():
    """Standard ICAO 9303 Doc 9303 Part 4 sample test vector."""
    return {
        "line1": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "line2": "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        "expected_surname": "ERIKSSON",
        "expected_given_names": "ANNA MARIA",
        "expected_passport_number": "L898902C3",
        "expected_nationality": "UTO",
        "expected_dob": "740812",
        "expected_sex": "F",
        "expected_expiry": "120415",
        "expected_personal_number": "ZE184226B",
        "expected_overall_valid": True,
    }


@pytest.fixture
def create_dummy_image():
    """Helper fixture to create a valid JPEG byte buffer with drawn text/shapes."""
    def _create(width=800, height=600, text="SAMPLE DOCUMENT"):
        img = np.ones((height, width, 3), dtype=np.uint8) * 240
        # Draw a simulated card border
        cv2.rectangle(img, (50, 50), (width - 50, height - 50), (40, 40, 40), 3)
        # Put text
        cv2.putText(img, text, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        _, buf = cv2.imencode('.jpg', img)
        return buf.tobytes()
    return _create


@pytest.fixture
def create_perspective_tilted_card():
    """Creates an image with a rotated/tilted quadrilateral card on a background."""
    def _create(width=1000, height=800):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Quad corners inside image
        pts = np.array([[150, 120], [820, 180], [780, 680], [120, 620]], np.int32)
        cv2.fillPoly(img, [pts], (230, 230, 230))
        cv2.polylines(img, [pts], True, (20, 20, 20), 4)
        _, buf = cv2.imencode('.png', img)
        return buf.tobytes()
    return _create
