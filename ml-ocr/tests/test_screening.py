"""Integration tests for Unified Document Screening Endpoint (POST /api/v1/screen)."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routes.face_verification import get_face_verification_service
from app.api.routes.ocr import get_ocr_service, get_tampering_service
from app.config import settings
from app.main import app
from app.models.schemas import FaceVerificationResult, OCRRegion, TamperingResult, UnifiedScreeningResult
from app.services.face_embedding_service import FaceEmbeddingService, MockFaceEmbeddingEngine
from app.services.face_verification_service import FaceVerificationService
from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
from app.services.tampering_service import TamperingService
from app.utils.image_utils import resize_image_if_needed


def _draw_synthetic_document_image(text: str = "PASSPORT", size: int = 400) -> bytes:
    """Helper to draw a valid synthetic document image with portrait face and text."""
    img = np.full((size, size, 3), 245, dtype=np.uint8)

    # Document border
    cv2.rectangle(img, (20, 20), (size - 20, size - 20), (100, 100, 100), 2)

    # Synthetic portrait face on the left
    face_center = (90, 120)
    skin_color = (140, 175, 220)  # BGR
    cv2.ellipse(img, face_center, (35, 45), 0, 0, 360, skin_color, -1)
    cv2.ellipse(img, face_center, (35, 45), 0, 0, 360, (70, 60, 50), 2)
    # Eyes & mouth
    cv2.circle(img, (75, 110), 4, (40, 30, 20), -1)
    cv2.circle(img, (105, 110), 4, (40, 30, 20), -1)
    cv2.line(img, (90, 115), (90, 130), (80, 60, 50), 2)
    cv2.ellipse(img, (90, 145), (15, 6), 0, 0, 180, (60, 40, 40), 2)

    # Text
    cv2.putText(img, text, (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    cv2.putText(img, "REPUBLIC OF UTOPIA", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _draw_synthetic_selfie_image(size: int = 300, add_blur: bool = False) -> bytes:
    """Helper to draw a valid synthetic selfie image."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    center = (size // 2, size // 2)

    skin_color = (140, 175, 220)
    cv2.ellipse(img, center, (65, 85), 0, 0, 360, skin_color, -1)
    cv2.ellipse(img, center, (65, 85), 0, 0, 360, (70, 60, 50), 2)

    cv2.circle(img, (center[0] - 25, center[1] - 15), 6, (40, 30, 20), -1)
    cv2.circle(img, (center[0] + 25, center[1] - 15), 6, (40, 30, 20), -1)
    cv2.line(img, (center[0], center[1] - 5), (center[0], center[1] + 15), (80, 60, 50), 2)
    cv2.ellipse(img, (center[0], center[1] + 35), (20, 8), 0, 0, 180, (60, 40, 40), 2)

    if add_blur:
        img = cv2.GaussianBlur(img, (41, 41), 25.0)

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestUnifiedScreeningEndpoint:
    """Comprehensive test suite for POST /api/v1/screen covering all specified scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def mock_passport_ocr(self):
        mock_engine = MockOCREngine()
        regions = [
            OCRRegion(text="PASSPORT", confidence=0.99, bbox=[50, 50, 200, 80]),
            OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.98, bbox=[50, 300, 750, 330]),
            OCRRegion(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", confidence=0.98, bbox=[50, 340, 750, 370]),
        ]
        mock_engine.set_predefined_result(
            OCRResult(
                raw_text="PASSPORT\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10",
                regions=regions,
                average_confidence=0.98
            )
        )
        custom_ocr_service = OCRService(engine=mock_engine)
        app.dependency_overrides[get_ocr_service] = lambda: custom_ocr_service
        yield mock_engine
        app.dependency_overrides.pop(get_ocr_service, None)

    @pytest.fixture
    def mock_visa_ocr(self):
        mock_engine = MockOCREngine()
        raw_visa = "VISA\nVJDEHE5CK1LAU5931014<11706262AUS<<<<<<<<<<<<<\nVNAUSSTEVENS<<JOHN<PAUL<<<<<<<<<<<<<<<<<<<<<"
        regions = [
            OCRRegion(text="VISA", confidence=0.98, bbox=[50, 50, 150, 80]),
            OCRRegion(text="VJDEHE5CK1LAU5931014<11706262AUS<<<<<<<<<<<<<", confidence=0.95, bbox=[50, 300, 750, 330]),
            OCRRegion(text="VNAUSSTEVENS<<JOHN<PAUL<<<<<<<<<<<<<<<<<<<<<", confidence=0.95, bbox=[50, 340, 750, 370]),
        ]
        mock_engine.set_predefined_result(
            OCRResult(raw_text=raw_visa, regions=regions, average_confidence=0.96)
        )
        custom_ocr_service = OCRService(engine=mock_engine)
        app.dependency_overrides[get_ocr_service] = lambda: custom_ocr_service
        yield mock_engine
        app.dependency_overrides.pop(get_ocr_service, None)

    @pytest.fixture
    def mock_national_id_ocr(self):
        mock_engine = MockOCREngine()
        raw_nid = "NATIONAL IDENTITY CARD\nIDUTO123456784<<<<<<<<<<<<<<<\n7408122F1204159UTO<<<<<<<<<<<8\nERIKSSON<<ANNA<MARIA<<<<<<<<<<"
        regions = [
            OCRRegion(text="NATIONAL IDENTITY CARD", confidence=0.98, bbox=[50, 50, 300, 80]),
            OCRRegion(text="IDUTO123456784<<<<<<<<<<<<<<<", confidence=0.96, bbox=[50, 300, 750, 330]),
            OCRRegion(text="7408122F1204159UTO<<<<<<<<<<<8", confidence=0.96, bbox=[50, 335, 750, 365]),
            OCRRegion(text="ERIKSSON<<ANNA<MARIA<<<<<<<<<<", confidence=0.96, bbox=[50, 370, 750, 400]),
        ]
        mock_engine.set_predefined_result(
            OCRResult(raw_text=raw_nid, regions=regions, average_confidence=0.97)
        )
        custom_ocr_service = OCRService(engine=mock_engine)
        app.dependency_overrides[get_ocr_service] = lambda: custom_ocr_service
        yield mock_engine
        app.dependency_overrides.pop(get_ocr_service, None)

    @pytest.fixture
    def mock_matching_face_service(self):
        vec = np.ones(128, dtype=np.float32)
        custom_face_service = FaceVerificationService(
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine(predefined_embedding=vec)),
            threshold=0.75
        )
        app.dependency_overrides[get_face_verification_service] = lambda: custom_face_service
        yield custom_face_service
        app.dependency_overrides.pop(get_face_verification_service, None)

    @pytest.fixture
    def mock_different_face_service(self):
        vec_doc = np.zeros(128, dtype=np.float32)
        vec_doc[0] = 1.0
        vec_selfie = np.zeros(128, dtype=np.float32)
        vec_selfie[1] = 1.0  # Orthogonal

        class AlternatingEngine(MockFaceEmbeddingEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extract_embedding(self, aligned_face):
                self.calls += 1
                return vec_doc if self.calls % 2 == 1 else vec_selfie

        custom_face_service = FaceVerificationService(
            embedding_service=FaceEmbeddingService(engine=AlternatingEngine()),
            threshold=0.75
        )
        app.dependency_overrides[get_face_verification_service] = lambda: custom_face_service
        yield custom_face_service
        app.dependency_overrides.pop(get_face_verification_service, None)

    def test_screen_passport_with_selfie_full_screening(self, client, mock_passport_ocr, mock_matching_face_service):
        """Scenario 1: Passport + selfie with detect_tampering=True and verify_face=True runs full pipeline."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")
        selfie_bytes = _draw_synthetic_selfie_image()

        response = client.post(
            "/api/v1/screen",
            data={
                "document_type": "passport",
                "run_ocr": "true",
                "detect_tampering": "true",
                "verify_face": "true"
            },
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["document_type"] == "passport"
        assert "ocr" in data
        assert data["ocr"]["success"] is True
        assert data["ocr"]["mrz"]["detected"] is True
        assert data["ocr"]["fields"]["passport_number"] == "L898902C3"
        assert "tampering" in data
        assert data["tampering"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert data["face_verification"]["match"] is True
        assert data["face_verification"]["similarity_score"] >= 0.75
        assert data["face_verification"]["match_band"] in ["STRONG_MATCH", "BORDERLINE_MATCH"]
        assert data["face_verification"]["ui_color"] in ["GREEN", "YELLOW"]

    def test_screen_ocr_only_default_behavior(self, client, mock_passport_ocr):
        """Scenario 2: Default flags run OCR only, skipping tampering and face verification."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")

        response = client.post(
            "/api/v1/screen",
            data={"document_type": "passport"},
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["ocr"]["success"] is True
        assert data["ocr"]["fields"]["passport_number"] == "L898902C3"
        assert data["tampering"] is None
        assert data["face_verification"]["status"] == "NOT_EVALUATED"

    def test_screen_tampering_disabled_explicit(self, client, mock_passport_ocr):
        """Scenario 3: Explicitly setting detect_tampering=False prevents tampering computation."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")

        response = client.post(
            "/api/v1/screen",
            data={"document_type": "passport", "detect_tampering": "false"},
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tampering"] is None

    def test_screen_face_disabled_explicit_even_with_selfie(self, client, mock_passport_ocr):
        """Scenario 4: Setting verify_face=False skips face verification even when selfie is provided."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")
        selfie_bytes = _draw_synthetic_selfie_image()

        response = client.post(
            "/api/v1/screen",
            data={"document_type": "passport", "verify_face": "false"},
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["face_verification"]["status"] == "NOT_EVALUATED"
        assert data["face_verification"]["match"] is None

    def test_screen_visa_with_selfie_success(self, client, mock_visa_ocr, mock_matching_face_service):
        """Scenario 5: Visa + selfie with verify_face=True processes MRV lines and face verification."""
        doc_bytes = _draw_synthetic_document_image("VISA")
        selfie_bytes = _draw_synthetic_selfie_image()

        response = client.post(
            "/api/v1/screen",
            data={"document_type": "visa", "verify_face": "true"},
            files={
                "document_image": ("visa.jpg", doc_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["document_type"] == "visa"
        assert data["ocr"]["success"] is True
        assert data["face_verification"]["match"] is True

    def test_screen_national_id_with_selfie_success(self, client, mock_national_id_ocr, mock_matching_face_service):
        """Scenario 6: National ID + selfie with verify_face=True processes TD1 fields and face verification."""
        doc_bytes = _draw_synthetic_document_image("NATIONAL ID")
        selfie_bytes = _draw_synthetic_selfie_image()

        response = client.post(
            "/api/v1/screen",
            data={"document_type": "national_id", "verify_face": "true"},
            files={
                "document_image": ("national_id.jpg", doc_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["document_type"] == "national_id"
        assert data["ocr"]["success"] is True
        assert data["face_verification"]["match"] is True

    def test_screen_document_type_auto_detection(self, client, mock_passport_ocr):
        """Scenario 7: document_type=auto automatically classifies document as passport."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")

        response = client.post(
            "/api/v1/screen",
            data={"document_type": "auto"},
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["document_type"] == "passport"
        assert data["ocr"]["success"] is True

    def test_face_verification_no_match(self, client, mock_passport_ocr, mock_different_face_service):
        """Scenario 8: Different subject selfie returns NO_MATCH, match=False, match_band=NO_MATCH, ui_color=RED."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")
        selfie_bytes = _draw_synthetic_selfie_image()

        response = client.post(
            "/api/v1/screen",
            data={"verify_face": "true"},
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_verification"]["status"] == "NO_MATCH"
        assert data["face_verification"]["match"] is False
        assert data["face_verification"]["similarity_score"] < 0.75
        assert data["face_verification"]["match_band"] == "NO_MATCH"
        assert data["face_verification"]["ui_color"] == "RED"

    def test_face_verification_insufficient_quality(self, client, mock_passport_ocr):
        """Scenario 9: Blurred selfie yields INSUFFICIENT_QUALITY without failing whole screening request."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")
        blurred_selfie = _draw_synthetic_selfie_image(add_blur=True)

        response = client.post(
            "/api/v1/screen",
            data={"verify_face": "true"},
            files={
                "document_image": ("passport.jpg", doc_bytes, "image/jpeg"),
                "selfie_image": ("blurred_selfie.jpg", blurred_selfie, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["ocr"]["success"] is True
        assert data["face_verification"]["status"] in ["INSUFFICIENT_QUALITY", "NO_FACE"]
        assert data["face_verification"]["match"] is None

    def test_tampering_low_result_on_clean_document(self, client, mock_passport_ocr):
        """Scenario 10: Tampering analysis on clean document produces LOW risk level when enabled."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")

        response = client.post(
            "/api/v1/screen",
            data={"detect_tampering": "true"},
            files={
                "document_image": ("clean_doc.jpg", doc_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "tampering" in data
        assert data["tampering"]["risk_level"] == "LOW"
        assert data["tampering"]["tampering_risk_score"] < 0.30

    def test_tampering_module_failure_tolerance(self, client, mock_passport_ocr):
        """Scenario 11: If tampering module fails, request succeeds with OCR, warnings, and fallback tampering status."""
        class CrashingTamperingService(TamperingService):
            def analyze_document(self, *args, **kwargs):
                raise RuntimeError("Forensic ELA engine out of memory")

        app.dependency_overrides[get_tampering_service] = lambda: CrashingTamperingService()

        try:
            doc_bytes = _draw_synthetic_document_image("PASSPORT")
            response = client.post(
                "/api/v1/screen",
                data={"detect_tampering": "true"},
                files={
                    "document_image": ("passport.jpg", doc_bytes, "image/jpeg")
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["ocr"]["success"] is True
            assert data["tampering"]["risk_level"] == "UNKNOWN"
            assert any("Tampering" in w for w in data["warnings"])
        finally:
            app.dependency_overrides.pop(get_tampering_service, None)

    def test_missing_document_file_returns_422(self, client):
        """Scenario 12: Missing required document_image field returns HTTP 422 Unprocessable Entity."""
        selfie_bytes = _draw_synthetic_selfie_image()
        response = client.post(
            "/api/v1/screen",
            files={
                "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
            }
        )
        assert response.status_code == 422

    def test_invalid_document_type_returns_400(self, client):
        """Scenario 13: Supplying an unsupported document_type returns HTTP 400 Bad Request."""
        doc_bytes = _draw_synthetic_document_image("PASSPORT")
        response = client.post(
            "/api/v1/screen",
            data={"document_type": "alien_id"},
            files={
                "document_image": ("doc.jpg", doc_bytes, "image/jpeg")
            }
        )
        assert response.status_code == 400
        assert "Invalid document_type" in response.json()["detail"]

    def test_corrupted_document_returns_400(self, client):
        """Scenario 14: Uploading an unreadable or non-image document returns HTTP 400 Bad Request."""
        response = client.post(
            "/api/v1/screen",
            files={
                "document_image": ("corrupt.txt", b"Non-image text content", "text/plain")
            }
        )
        assert response.status_code == 400

    def test_health_check_is_lightweight_and_fast(self, client):
        """Scenario 15: Verifies GET /health returns static 200 OK immediately with no blocking operations."""
        import time
        start_time = time.perf_counter()
        response = client.get("/health")
        elapsed = time.perf_counter() - start_time

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "service" in data
        assert "available_ocr_engines" in data
        assert elapsed < 0.100

    def test_face_verification_module_failure_tolerance(self, client, mock_passport_ocr):
        """Scenario 16: If face verification module raises an unexpected exception, screening still succeeds with OCR."""
        class CrashingFaceService(FaceVerificationService):
            def verify_faces(self, *args, **kwargs):
                raise RuntimeError("Face embedding engine timeout")

        app.dependency_overrides[get_face_verification_service] = lambda: CrashingFaceService()

        try:
            doc_bytes = _draw_synthetic_document_image("PASSPORT")
            selfie_bytes = _draw_synthetic_selfie_image()

            response = client.post(
                "/api/v1/screen",
                data={"verify_face": "true"},
                files={
                    "document_image": ("passport.jpg", doc_bytes, "image/jpeg"),
                    "selfie_image": ("selfie.jpg", selfie_bytes, "image/jpeg")
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["ocr"]["success"] is True
            assert data["face_verification"]["status"] == "PROCESSING_ERROR"
            assert any("Face verification" in w for w in data["warnings"])
        finally:
            app.dependency_overrides.pop(get_face_verification_service, None)

    def test_large_image_downscale_preserves_aspect_ratio_and_succeeds(self, client, mock_passport_ocr):
        """Scenario 17: Large image (> 1400 px) is safely downscaled to max 1400 px without upscaling small images."""
        large_img = np.full((2400, 1800, 3), 245, dtype=np.uint8)
        resized, scale = resize_image_if_needed(large_img, max_dim=1400, min_dim=0)
        assert max(resized.shape[:2]) == 1400
        assert scale < 1.0

        # Small image is not upscaled
        small_img = np.full((300, 200, 3), 245, dtype=np.uint8)
        small_resized, small_scale = resize_image_if_needed(small_img, max_dim=1400, min_dim=0)
        assert small_resized.shape == small_img.shape
        assert small_scale == 1.0

    def test_mrz_staged_ocr_fallback_limits(self, client):
        """Scenario 18: Staged OCR does not exceed MAX_MRZ_FALLBACK_ATTEMPTS passes when MRZ is missing."""
        calls = {"extract": 0, "extract_mrz": 0}

        class CountingMockEngine(MockOCREngine):
            def extract_text(self, image):
                calls["extract"] += 1
                return OCRResult(raw_text="NOTHING TO SEE HERE", regions=[], average_confidence=0.1)

            def extract_mrz_text(self, image, psm=6):
                calls["extract_mrz"] += 1
                return OCRResult(raw_text="", regions=[], average_confidence=0.0)

        custom_ocr_service = OCRService(engine=CountingMockEngine())
        app.dependency_overrides[get_ocr_service] = lambda: custom_ocr_service

        try:
            doc_bytes = _draw_synthetic_document_image("PASSPORT")
            response = client.post(
                "/api/v1/screen",
                data={"document_type": "passport"},
                files={"document_image": ("doc.jpg", doc_bytes, "image/jpeg")}
            )
            assert response.status_code == 200
            # 1 primary OCR + 1 targeted crop + at most MAX_MRZ_FALLBACK_ATTEMPTS (2) = <= 4 calls total
            assert calls["extract"] == 1
            assert calls["extract_mrz"] <= (1 + settings.MAX_MRZ_FALLBACK_ATTEMPTS)
        finally:
            app.dependency_overrides.pop(get_ocr_service, None)

    def test_ocr_pass_timeout_returns_clean_module_error(self, client):
        """Scenario 19: If an individual OCR pass times out, screening returns 200 with clean module error info."""
        class TimingOutEngine(MockOCREngine):
            def extract_text(self, image):
                raise TimeoutError("Tesseract process timeout (30s)")

        custom_ocr_service = OCRService(engine=TimingOutEngine())
        app.dependency_overrides[get_ocr_service] = lambda: custom_ocr_service

        try:
            doc_bytes = _draw_synthetic_document_image("PASSPORT")
            response = client.post(
                "/api/v1/screen",
                data={"document_type": "passport"},
                files={"document_image": ("doc.jpg", doc_bytes, "image/jpeg")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["ocr"]["success"] is False
            assert "timeout" in data["ocr"]["error"].lower()
            assert any("timed out" in w.lower() for w in data["warnings"])
        finally:
            app.dependency_overrides.pop(get_ocr_service, None)

