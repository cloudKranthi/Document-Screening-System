"""Comprehensive unit and API integration tests for Biometric Face Verification module."""

import io
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.models.schemas import FaceVerificationResult, FaceDetail
from app.services.face_detection_service import FaceDetectionService
from app.services.face_alignment_service import FaceAlignmentService, STANDARD_5_LANDMARKS_112
from app.services.face_embedding_service import (
    FaceEmbeddingService,
    MockFaceEmbeddingEngine,
    SFaceEmbeddingEngine
)
from app.services.face_verification_service import (
    FaceVerificationService,
    compute_cosine_similarity,
    normalize_similarity_to_public_score,
    classify_similarity_band
)
from app.api.routes.face_verification import get_face_verification_service



def _create_synthetic_face_image(name: str = "Test Subject", add_blur: bool = False, size: int = 300) -> np.ndarray:
    """Helper to draw a valid synthetic face with facial features."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    center = (size // 2, size // 2)
    # Head
    cv2.ellipse(img, center, (int(size * 0.25), int(size * 0.35)), 0, 0, 360, (200, 180, 160), -1)
    # Eyes
    cv2.circle(img, (int(size * 0.40), int(size * 0.45)), int(size * 0.03), (40, 30, 20), -1)
    cv2.circle(img, (int(size * 0.60), int(size * 0.45)), int(size * 0.03), (40, 30, 20), -1)
    # Nose
    cv2.line(img, (int(size * 0.50), int(size * 0.48)), (int(size * 0.50), int(size * 0.58)), (80, 60, 50), 2)
    # Mouth
    cv2.ellipse(img, (int(size * 0.50), int(size * 0.68)), (int(size * 0.08), int(size * 0.04)), 0, 0, 180, (60, 40, 40), 2)

    if add_blur:
        img = cv2.GaussianBlur(img, (31, 31), 15.0)

    cv2.putText(img, name, (10, size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    return img


class MockFaceDetector(FaceDetectionService):
    """Custom mock face detector for fine-grained scenario testing."""

    def __init__(self, faces_to_return=None):
        super().__init__()
        self.faces_to_return = faces_to_return

    def detect_faces(self, image, score_threshold=None):
        if self.faces_to_return is not None:
            return self.faces_to_return
        # Default: 1 valid synthetic face in center
        h, w = image.shape[:2]
        bx, by, bw, bh = int(w * 0.25), int(h * 0.20), int(w * 0.50), int(h * 0.60)
        landmarks = np.array([
            [bx + bw * 0.3, by + bh * 0.4],
            [bx + bw * 0.7, by + bh * 0.4],
            [bx + bw * 0.5, by + bh * 0.6],
            [bx + bw * 0.35, by + bh * 0.8],
            [bx + bw * 0.65, by + bh * 0.8]
        ], dtype=np.float32)
        return [{
            "bbox": [bx, by, bw, bh],
            "confidence": 0.98,
            "landmarks": landmarks,
            "raw_face": np.array([bx, by, bw, bh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.98], dtype=np.float32)
        }]


class TestCosineSimilarityAndMath:
    """Unit tests for mathematical cosine similarity and normalization logic."""

    def test_cosine_similarity_identical_embeddings(self):
        """Identical vectors must return exact cosine similarity of 1.0."""
        vec = np.random.randn(128).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        sim = compute_cosine_similarity(vec, vec)
        assert pytest.approx(sim, abs=1e-5) == 1.0

    def test_highly_similar_embeddings(self):
        """Slightly perturbed vectors return high similarity above 0.75."""
        vec1 = np.random.randn(128).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        rand_dir = np.random.randn(128).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        vec2 = 0.95 * vec1 + 0.05 * rand_dir
        vec2 = vec2 / np.linalg.norm(vec2)

        sim = compute_cosine_similarity(vec1, vec2)
        assert sim >= 0.75
        assert sim <= 1.0


    def test_different_embeddings(self):
        """Orthogonal or opposing vectors return low or negative similarity."""
        vec1 = np.zeros(128, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(128, dtype=np.float32)
        vec2[1] = 1.0  # Orthogonal

        sim = compute_cosine_similarity(vec1, vec2)
        assert pytest.approx(sim, abs=1e-5) == 0.0

    def test_score_remains_0_1(self):
        """Public score must always be strictly clamped within [0.0, 1.0]."""
        assert normalize_similarity_to_public_score(1.5) == 1.0
        assert normalize_similarity_to_public_score(-0.8) == 0.0
        assert normalize_similarity_to_public_score(0.82345) == 0.8235
        assert normalize_similarity_to_public_score(0.0) == 0.0

    def test_threshold_exactly_0_75(self):
        """Threshold defaults to 0.75 from settings and determines match correctly."""
        service = FaceVerificationService()
        assert service.threshold == 0.75

    def test_score_ge_threshold_match(self):
        """Score >= 0.75 returns match=True and status=MATCH."""
        vec1 = np.ones(128, dtype=np.float32)
        mock_engine = MockFaceEmbeddingEngine(predefined_embedding=vec1)
        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=mock_engine),
            threshold=0.75
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.success is True
        assert res.status == "MATCH"
        assert res.match is True
        assert res.similarity_score >= 0.75

    def test_score_lt_threshold_no_match(self):
        """Score < 0.75 returns match=False and status=NO_MATCH."""
        vec_doc = np.zeros(128, dtype=np.float32)
        vec_doc[0] = 1.0
        vec_selfie = np.zeros(128, dtype=np.float32)
        vec_selfie[1] = 1.0  # Orthogonal

        class AlternatingMockEngine(MockFaceEmbeddingEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extract_embedding(self, aligned_face):
                self.calls += 1
                return vec_doc if self.calls % 2 == 1 else vec_selfie

        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=AlternatingMockEngine()),
            threshold=0.75
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.success is True
        assert res.status == "NO_MATCH"
        assert res.match is False
        assert res.similarity_score < 0.75


class TestFaceQualityAndDetectionScenarios:
    """Tests for edge cases: missing faces, multiple faces, blur, low resolution, corruption."""

    def test_no_document_face(self):
        """If identity document has no face, return NO_FACE status and match=None."""
        service = FaceVerificationService(
            detection_service=MockFaceDetector(faces_to_return=[]),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine())
        )
        blank = np.full((300, 300, 3), 255, dtype=np.uint8)
        img = _create_synthetic_face_image()

        res = service.verify_faces(blank, img)
        assert res.status == "NO_FACE"
        assert res.match is None
        assert res.similarity_score is None
        assert res.document_face.detected is False

    def test_no_selfie_face(self):
        """If selfie has no face, return NO_FACE status and match=None."""
        class DocOnlyDetector(FaceDetectionService):
            def detect_document_portrait(self, img):
                return {"bbox": [50, 50, 100, 120], "confidence": 0.95, "landmarks": STANDARD_5_LANDMARKS_112}, {"quality_passed": True, "quality_score": 0.9, "warnings": []}, []

            def detect_selfie_face(self, img):
                return None, None, ["No face detected in selfie."], "NO_FACE"

        service = FaceVerificationService(
            detection_service=DocOnlyDetector(),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine())
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.status == "NO_FACE"
        assert res.match is None
        assert res.selfie_face.detected is False

    def test_multiple_selfie_faces(self):
        """Multiple prominent faces in selfie returns MULTIPLE_FACES status."""
        class MultiFaceDetector(FaceDetectionService):
            def detect_document_portrait(self, img):
                return {"bbox": [50, 50, 100, 120], "confidence": 0.95, "landmarks": STANDARD_5_LANDMARKS_112}, {"quality_passed": True, "quality_score": 0.9, "warnings": []}, []

            def detect_selfie_face(self, img):
                return None, None, ["Multiple faces detected in selfie."], "MULTIPLE_FACES"

        service = FaceVerificationService(
            detection_service=MultiFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine())
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.status == "MULTIPLE_FACES"
        assert res.match is None
        assert any("Multiple faces" in w for w in res.warnings)

    def test_blurred_selfie(self):
        """Heavily blurred selfie returns INSUFFICIENT_QUALITY status."""
        detector = FaceDetectionService()
        blurred_selfie = _create_synthetic_face_image(add_blur=True)
        # Quality check directly on blurred image
        quality = detector.evaluate_face_quality(blurred_selfie, [50, 50, 200, 200])
        assert quality["quality_passed"] is False
        assert any("blurred" in w for w in quality["warnings"])

    def test_low_resolution_document_portrait(self):
        """Microscopic portrait below min size returns INSUFFICIENT_QUALITY warning."""
        detector = FaceDetectionService()
        small_face = np.full((100, 100, 3), 200, dtype=np.uint8)
        quality = detector.evaluate_face_quality(small_face, [10, 10, 30, 35])  # 30x35 px < 60x60
        assert quality["quality_passed"] is False
        assert any("too small" in w for w in quality["warnings"])

    def test_invalid_image(self):
        """Unreadable or None image returns PROCESSING_ERROR safely without crashing."""
        service = FaceVerificationService()
        res = service.verify_faces(None, np.full((100, 100, 3), 255, dtype=np.uint8))
        assert res.status == "PROCESSING_ERROR"
        assert res.match is None

    def test_deterministic_result_for_same_embeddings(self):
        """Same input pair evaluated repeatedly yields identical similarity score and status."""
        vec = np.random.randn(128).astype(np.float32)
        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine(predefined_embedding=vec))
        )
        img = _create_synthetic_face_image()
        res1 = service.verify_faces(img, img)
        res2 = service.verify_faces(img, img)

        assert res1.similarity_score == res2.similarity_score
        assert res1.match == res2.match
        assert res1.status == res2.status

    def test_model_exception_handled_safely(self):
        """Embedding extraction exceptions fail safely with PROCESSING_ERROR."""
        class CrashingEngine(MockFaceEmbeddingEngine):
            def extract_embedding(self, aligned_face):
                raise RuntimeError("Hardware failure during feature extraction")

        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=CrashingEngine())
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.status == "PROCESSING_ERROR"
        assert res.match is None


class TestFaceVerificationAPI:
    """Integration tests for POST /api/v1/face/verify endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def override_same_person_service(self):
        vec = np.ones(128, dtype=np.float32)
        custom_service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine(predefined_embedding=vec)),
            threshold=0.75
        )
        app.dependency_overrides[get_face_verification_service] = lambda: custom_service
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def override_diff_person_service(self):
        vec_doc = np.zeros(128, dtype=np.float32)
        vec_doc[0] = 1.0
        vec_selfie = np.zeros(128, dtype=np.float32)
        vec_selfie[1] = 1.0

        class AlternatingEngine(MockFaceEmbeddingEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extract_embedding(self, aligned_face):
                self.calls += 1
                return vec_doc if self.calls % 2 == 1 else vec_selfie

        custom_service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=AlternatingEngine()),
            threshold=0.75
        )
        app.dependency_overrides[get_face_verification_service] = lambda: custom_service
        yield
        app.dependency_overrides.clear()

    def test_api_same_person_match(self, client, override_same_person_service):
        """API POST /api/v1/face/verify on matching subject returns HTTP 200 with match=true and status=MATCH."""
        doc_img = _create_synthetic_face_image("Alice Doc")
        selfie_img = _create_synthetic_face_image("Alice Selfie")

        _, doc_buf = cv2.imencode(".jpg", doc_img)
        _, selfie_buf = cv2.imencode(".jpg", selfie_img)

        response = client.post(
            "/api/v1/face/verify",
            files={
                "document_image": ("doc.jpg", doc_buf.tobytes(), "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_buf.tobytes(), "image/jpeg")
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "MATCH"
        assert data["match"] is True
        assert data["similarity_score"] >= 0.75
        assert data["threshold"] == 0.75
        assert data["match_band"] == "STRONG_MATCH"
        assert data["ui_color"] == "GREEN"
        assert data["document_face"]["detected"] is True
        assert data["selfie_face"]["detected"] is True
        assert "disclaimer" in data

    def test_api_different_person_no_match(self, client, override_diff_person_service):
        """API POST /api/v1/face/verify on different subjects returns HTTP 200 with match=false and status=NO_MATCH."""
        doc_img = _create_synthetic_face_image("Alice Doc")
        selfie_img = _create_synthetic_face_image("Bob Selfie")

        _, doc_buf = cv2.imencode(".jpg", doc_img)
        _, selfie_buf = cv2.imencode(".jpg", selfie_img)

        response = client.post(
            "/api/v1/face/verify",
            files={
                "document_image": ("doc.jpg", doc_buf.tobytes(), "image/jpeg"),
                "selfie_image": ("selfie.jpg", selfie_buf.tobytes(), "image/jpeg")
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "NO_MATCH"
        assert data["match"] is False
        assert data["similarity_score"] < 0.75
        assert data["match_band"] == "NO_MATCH"
        assert data["ui_color"] == "RED"


    def test_api_missing_document_image(self, client):
        """Missing required document image field returns HTTP 422 Unprocessable Entity."""
        selfie_img = _create_synthetic_face_image()
        _, selfie_buf = cv2.imencode(".jpg", selfie_img)

        response = client.post(
            "/api/v1/face/verify",
            files={
                "selfie_image": ("selfie.jpg", selfie_buf.tobytes(), "image/jpeg")
            }
        )
        assert response.status_code == 422

    def test_api_missing_selfie_image(self, client):
        """Missing required selfie image field returns HTTP 422 Unprocessable Entity."""
        doc_img = _create_synthetic_face_image()
        _, doc_buf = cv2.imencode(".jpg", doc_img)

        response = client.post(
            "/api/v1/face/verify",
            files={
                "document_image": ("doc.jpg", doc_buf.tobytes(), "image/jpeg")
            }
        )
        assert response.status_code == 422

    def test_api_invalid_image_format(self, client):
        """Uploading text/plain file returns HTTP 400 Bad Request."""
        response = client.post(
            "/api/v1/face/verify",
            files={
                "document_image": ("doc.txt", b"not an image", "text/plain"),
                "selfie_image": ("selfie.txt", b"not an image", "text/plain")
            }
        )
        assert response.status_code == 400

    def test_evaluate_face_verification_script(self, tmp_path):
        """Benchmark evaluation harness computes dataset metrics and outputs structured report."""
        from evaluation.evaluate_face_verification import run_face_verification_benchmark
        gen_dir = str(tmp_path / "genuine")
        imp_dir = str(tmp_path / "impostor")

        report = run_face_verification_benchmark(
            genuine_dir=gen_dir,
            impostor_dir=imp_dir,
            synthetic_samples=2,
            generate_synthetic_if_missing=True
        )
        assert "sample_counts" in report
        assert "mean_scores" in report
        assert "classification_metrics" in report
        assert report["sample_counts"]["genuine_pairs"] == 2
        assert report["sample_counts"]["impostor_pairs"] == 2

    def test_download_face_models_script(self, tmp_path):
        """Deployment model download script creates directory and skips existing models properly."""
        from scripts.download_face_models import download_model, MODEL_DEFINITIONS
        target_dir = tmp_path / "models"
        target_dir.mkdir()

        # Test with dummy model file
        dummy_config = {
            "filename": "test_model.onnx",
            "url": "http://invalid-url-should-skip.local",
            "min_size_bytes": 10,
            "description": "Test Model"
        }
        test_file = target_dir / "test_model.onnx"
        test_file.write_bytes(b"0123456789ABCDEF")

        # When file exists with >= min_size, download_model should return True without contacting network
        success = download_model("test", dummy_config, target_dir)
        assert success is True



class TestSimilarityBands:
    """Explicit tests for similarity bands, UI colors, and boundary conditions."""

    def test_strong_match_boundary_and_above(self):
        """Scores >= 0.80 must be classified as STRONG_MATCH with UI color GREEN."""
        assert classify_similarity_band(1.0) == ("STRONG_MATCH", "GREEN")
        assert classify_similarity_band(0.8658) == ("STRONG_MATCH", "GREEN")
        assert classify_similarity_band(0.80) == ("STRONG_MATCH", "GREEN")
        assert classify_similarity_band(0.8000) == ("STRONG_MATCH", "GREEN")

    def test_borderline_match_boundary_and_range(self):
        """Scores in [0.75, 0.7999] must be classified as BORDERLINE_MATCH with UI color YELLOW."""
        assert classify_similarity_band(0.7999) == ("BORDERLINE_MATCH", "YELLOW")
        assert classify_similarity_band(0.78) == ("BORDERLINE_MATCH", "YELLOW")
        assert classify_similarity_band(0.75) == ("BORDERLINE_MATCH", "YELLOW")
        assert classify_similarity_band(0.7500) == ("BORDERLINE_MATCH", "YELLOW")

    def test_manual_review_boundary_and_range(self):
        """Scores in [0.60, 0.7499] must be classified as MANUAL_REVIEW with UI color ORANGE."""
        assert classify_similarity_band(0.7499) == ("MANUAL_REVIEW", "ORANGE")
        assert classify_similarity_band(0.72) == ("MANUAL_REVIEW", "ORANGE")
        assert classify_similarity_band(0.65) == ("MANUAL_REVIEW", "ORANGE")
        assert classify_similarity_band(0.60) == ("MANUAL_REVIEW", "ORANGE")
        assert classify_similarity_band(0.6000) == ("MANUAL_REVIEW", "ORANGE")

    def test_no_match_boundary_and_below(self):
        """Scores < 0.60 must be classified as NO_MATCH with UI color RED."""
        assert classify_similarity_band(0.5999) == ("NO_MATCH", "RED")
        assert classify_similarity_band(0.50) == ("NO_MATCH", "RED")
        assert classify_similarity_band(0.42) == ("NO_MATCH", "RED")
        assert classify_similarity_band(0.0) == ("NO_MATCH", "RED")

    def test_not_evaluated_when_none(self):
        """None score must be classified as NOT_EVALUATED with UI color GRAY."""
        assert classify_similarity_band(None) == ("NOT_EVALUATED", "GRAY")

    def test_end_to_end_strong_match_service_result(self):
        """Service produces match=True, match_band=STRONG_MATCH, ui_color=GREEN when score is 0.85."""
        vec = np.ones(128, dtype=np.float32)
        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine(predefined_embedding=vec)),
            threshold=0.75
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.match is True
        assert res.similarity_score == 1.0
        assert res.match_band == "STRONG_MATCH"
        assert res.ui_color == "GREEN"

    def test_end_to_end_borderline_match_service_result(self):
        """Service produces match=True, match_band=BORDERLINE_MATCH, ui_color=YELLOW when score is 0.77."""
        # Create vectors with dot product ~ 0.77
        vec_doc = np.zeros(128, dtype=np.float32)
        vec_doc[0] = 1.0
        vec_selfie = np.zeros(128, dtype=np.float32)
        vec_selfie[0] = 0.77
        vec_selfie[1] = np.sqrt(1.0 - 0.77**2)

        class FixedScoreEngine(MockFaceEmbeddingEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extract_embedding(self, aligned_face):
                self.calls += 1
                return vec_doc if self.calls % 2 == 1 else vec_selfie

        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=FixedScoreEngine()),
            threshold=0.75
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.match is True
        assert res.similarity_score == 0.77
        assert res.match_band == "BORDERLINE_MATCH"
        assert res.ui_color == "YELLOW"

    def test_end_to_end_manual_review_service_result(self):
        """Service produces match=False, match_band=MANUAL_REVIEW, ui_color=ORANGE when score is 0.68."""
        vec_doc = np.zeros(128, dtype=np.float32)
        vec_doc[0] = 1.0
        vec_selfie = np.zeros(128, dtype=np.float32)
        vec_selfie[0] = 0.68
        vec_selfie[1] = np.sqrt(1.0 - 0.68**2)

        class FixedScoreEngine(MockFaceEmbeddingEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extract_embedding(self, aligned_face):
                self.calls += 1
                return vec_doc if self.calls % 2 == 1 else vec_selfie

        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=FixedScoreEngine()),
            threshold=0.75
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.match is False
        assert res.similarity_score == 0.68
        assert res.match_band == "MANUAL_REVIEW"
        assert res.ui_color == "ORANGE"

    def test_end_to_end_no_match_service_result(self):
        """Service produces match=False, match_band=NO_MATCH, ui_color=RED when score is 0.45."""
        vec_doc = np.zeros(128, dtype=np.float32)
        vec_doc[0] = 1.0
        vec_selfie = np.zeros(128, dtype=np.float32)
        vec_selfie[0] = 0.45
        vec_selfie[1] = np.sqrt(1.0 - 0.45**2)

        class FixedScoreEngine(MockFaceEmbeddingEngine):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extract_embedding(self, aligned_face):
                self.calls += 1
                return vec_doc if self.calls % 2 == 1 else vec_selfie

        service = FaceVerificationService(
            detection_service=MockFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=FixedScoreEngine()),
            threshold=0.75
        )
        img = _create_synthetic_face_image()
        res = service.verify_faces(img, img)

        assert res.match is False
        assert res.similarity_score == 0.45
        assert res.match_band == "NO_MATCH"
        assert res.ui_color == "RED"

    def test_error_states_produce_not_evaluated_gray(self):
        """Error states (NO_FACE, MULTIPLE_FACES, INSUFFICIENT_QUALITY, PROCESSING_ERROR) return NOT_EVALUATED and GRAY."""
        # 1. NO_FACE
        service_no_face = FaceVerificationService(
            detection_service=MockFaceDetector(faces_to_return=[]),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine())
        )
        blank = np.full((300, 300, 3), 255, dtype=np.uint8)
        img = _create_synthetic_face_image()
        res_no_face = service_no_face.verify_faces(blank, img)
        assert res_no_face.status == "NO_FACE"
        assert res_no_face.match_band == "NOT_EVALUATED"
        assert res_no_face.ui_color == "GRAY"

        # 2. MULTIPLE_FACES
        class MultiFaceDetector(FaceDetectionService):
            def detect_document_portrait(self, img):
                return {"bbox": [50, 50, 100, 120], "confidence": 0.95, "landmarks": STANDARD_5_LANDMARKS_112}, {"quality_passed": True, "quality_score": 0.9, "warnings": []}, []

            def detect_selfie_face(self, img):
                return None, None, ["Multiple faces detected in selfie."], "MULTIPLE_FACES"

        service_multi = FaceVerificationService(
            detection_service=MultiFaceDetector(),
            embedding_service=FaceEmbeddingService(engine=MockFaceEmbeddingEngine())
        )
        res_multi = service_multi.verify_faces(img, img)
        assert res_multi.status == "MULTIPLE_FACES"
        assert res_multi.match_band == "NOT_EVALUATED"
        assert res_multi.ui_color == "GRAY"

        # 3. PROCESSING_ERROR
        res_error = service_no_face.verify_faces(None, img)
        assert res_error.status == "PROCESSING_ERROR"
        assert res_error.match_band == "NOT_EVALUATED"
        assert res_error.ui_color == "GRAY"


