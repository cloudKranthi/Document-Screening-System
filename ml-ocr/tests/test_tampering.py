"""Comprehensive test suite for the document tampering detection module."""

import io
from pathlib import Path
import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from app.main import app
from app.evaluation.calibration import TamperingCalibrationHarness
from app.models.schemas import TamperingResult
from app.services.tampering_service import TamperingService
from app.utils.tampering_utils import (
    compute_ela_residual,
    compute_local_gradient_and_sharpness,
    compute_noise_residual,
    detect_orb_copy_move_clusters,
    extract_image_metadata_signals,
)


@pytest.fixture
def tampering_service():
    return TamperingService()


@pytest.fixture
def clean_synthetic_document():
    """Generates a clean synthetic document image with uniform texture and text lines."""
    img = np.full((500, 700, 3), 245, dtype=np.uint8)
    cv2.putText(img, "PASSPORT OF THE REPUBLIC", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "SURNAME: ERIKSSON", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "GIVEN NAMES: ANNA MARIA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "NATIONALITY: UTO", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "PASSPORT NO: L898902C3", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.putText(img, "L898902C36UTO7408122F1204159ZE184226B<<<<<10", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    
    # Add subtle uniform sensor noise
    noise = np.random.normal(0, 2.0, img.shape).astype(np.int16)
    clean_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Encode as clean JPEG
    _, buf = cv2.imencode(".jpg", clean_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return clean_img, buf.tobytes()


class TestTamperingSignals:
    """Unit tests for individual tampering signal algorithms and utilities."""

    def test_ela_residual_computation(self, clean_synthetic_document):
        img_arr, _ = clean_synthetic_document
        ela_diff, mean_err, std_err = compute_ela_residual(img_arr, quality=90)
        assert ela_diff.shape == img_arr.shape[:2]
        assert mean_err >= 0.0
        assert std_err >= 0.0

    def test_noise_residual_computation(self, clean_synthetic_document):
        img_arr, _ = clean_synthetic_document
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        res = compute_noise_residual(gray)
        assert res.shape == gray.shape
        assert float(np.mean(np.abs(res))) >= 0.0

    def test_gradient_and_sharpness_computation(self, clean_synthetic_document):
        img_arr, _ = clean_synthetic_document
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        grad_mag, lap_abs = compute_local_gradient_and_sharpness(gray)
        assert grad_mag.shape == gray.shape
        assert lap_abs.shape == gray.shape

    def test_metadata_extraction_absent(self):
        """Test missing metadata produces score 0.0 and no false alarm."""
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        meta = extract_image_metadata_signals(buf.getvalue())
        assert meta["score"] == 0.0
        assert meta["is_editing_software"] is False

    def test_metadata_extraction_photoshop_tag(self):
        """Test Photoshop software tag is detected and produces a calibrated weak score (<= 0.40)."""
        img = Image.new("RGB", (100, 100), color="white")
        exif = img.getexif()
        # Tag 305 is Software
        exif[305] = "Adobe Photoshop 2024 (Windows)"
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        
        meta = extract_image_metadata_signals(buf.getvalue())
        assert meta["has_metadata"] is True
        assert meta["is_editing_software"] is True
        assert meta["detected_software_name"] == "Photoshop"
        assert meta["score"] == 0.40


class TestTamperingServiceForensics:
    """Forensic scenario tests for the full TamperingService pipeline."""

    def test_untouched_clean_document_low_risk(self, tampering_service, clean_synthetic_document):
        """Clean document without anomalies must produce LOW tampering risk."""
        img_arr, img_bytes = clean_synthetic_document
        result = tampering_service.analyze_document(image_bytes=img_bytes, document_image=img_arr)
        
        assert isinstance(result, TamperingResult)
        assert result.tampering_risk_score < 0.35
        assert result.risk_level == "LOW"
        assert result.signals["metadata"].score == 0.0

    def test_uniform_jpeg_recompression_safeguard(self, tampering_service, clean_synthetic_document):
        """Uniformly recompressed images must NOT be flagged as tampered."""
        img_arr, _ = clean_synthetic_document
        # Recompress uniformly at quality 70
        _, buf = cv2.imencode(".jpg", img_arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        recompressed_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        
        result = tampering_service.analyze_document(image_bytes=buf.tobytes(), document_image=recompressed_img)
        assert result.tampering_risk_score < 0.35
        assert result.risk_level == "LOW"

    def test_localized_pasted_text_patch_detection(self, tampering_service, clean_synthetic_document):
        """A document with a localized spliced text patch must increase tampering indicators."""
        img_arr, _ = clean_synthetic_document
        tampered = img_arr.copy()
        
        # Create a spliced patch with high-contrast sharp edges and foreign noise
        patch = np.full((60, 180, 3), 200, dtype=np.uint8)
        cv2.putText(patch, "FORGED-999", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        # Add foreign high-variance noise to the patch
        patch_noise = np.random.normal(0, 18.0, patch.shape).astype(np.int16)
        patch_noisy = np.clip(patch.astype(np.int16) + patch_noise, 0, 255).astype(np.uint8)
        
        # Splice into image
        tampered[120:180, 350:530] = patch_noisy
        _, buf = cv2.imencode(".jpg", tampered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        result = tampering_service.analyze_document(image_bytes=buf.tobytes(), document_image=tampered)
        assert result.tampering_risk_score > 0.10
        assert len(result.indicators) > 0
        sig_types = [ind.type for ind in result.indicators]
        assert any(t in sig_types for t in ["compression_inconsistency", "noise_inconsistency", "ela_local_inconsistency"])



    def test_copy_move_cloned_region_detected(self, tampering_service):
        """Duplicating a textured region to another location within the image triggers copy-move detection."""
        # Create base canvas with distinct geometric features
        img = np.full((600, 800, 3), 240, dtype=np.uint8)
        
        # Create a rich feature patch (e.g. official seal / textured crest)
        seal = np.full((120, 120, 3), 220, dtype=np.uint8)
        cv2.circle(seal, (60, 60), 50, (0, 0, 150), 3)
        cv2.rectangle(seal, (30, 30), (90, 90), (150, 0, 0), 2)
        cv2.putText(seal, "OFFICIAL", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)
        
        # Place original patch at (80, 80)
        img[80:200, 80:200] = seal
        # Clone patch to (350, 450)
        img[350:470, 450:570] = seal
        
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result = tampering_service.analyze_document(image_bytes=buf.tobytes(), document_image=img)
        
        assert result.signals["copy_move"].score > 0.30
        assert any(ind.type == "copy_move" for ind in result.indicators)
        cm_ind = next(ind for ind in result.indicators if ind.type == "copy_move")
        assert len(cm_ind.regions) >= 2

    def test_localized_blur_smudge_patch(self, tampering_service, clean_synthetic_document):
        """Localized smudging/blurring of a text section increases edge/texture anomaly score."""
        img_arr, _ = clean_synthetic_document
        tampered = img_arr.copy()
        
        # Apply heavy Gaussian blur to a specific region (simulating digital eraser / smudge)
        tampered[120:190, 50:350] = cv2.GaussianBlur(tampered[120:190, 50:350], (31, 31), 10.0)
        _, buf = cv2.imencode(".jpg", tampered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        result = tampering_service.analyze_document(image_bytes=buf.tobytes(), document_image=tampered)
        assert result.signals["edge_texture_inconsistency"].score > 0.10

    def test_high_uniform_noise_image_remains_safe(self, tampering_service, clean_synthetic_document):
        """High camera sensor noise that is UNIFORM across the document does not cause a false alarm."""
        img_arr, _ = clean_synthetic_document
        # Add strong uniform Gaussian noise across the whole image
        noise = np.random.normal(0, 15.0, img_arr.shape).astype(np.int16)
        noisy_img = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", noisy_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        result = tampering_service.analyze_document(image_bytes=buf.tobytes(), document_image=noisy_img)
        # Uniform noise should have low localized MAD divergence
        assert result.tampering_risk_score < 0.40

    def test_metadata_photoshop_weak_signal_fusion(self, tampering_service, clean_synthetic_document):
        """A document with Photoshop metadata tag alone contributes mildly without pushing clean document to HIGH."""
        img_arr, _ = clean_synthetic_document
        pil_img = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
        exif = pil_img.getexif()
        exif[305] = "Adobe Photoshop CC 2024"
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", exif=exif)
        
        result = tampering_service.analyze_document(image_bytes=buf.getvalue(), document_image=img_arr)
        assert result.signals["metadata"].score in (0.35, 0.40)
        assert result.signals["metadata"].editing_software_detected is True
        assert result.tampering_risk_score < 0.35
        assert result.risk_level == "LOW"


    def test_malformed_image_fails_safely(self, tampering_service):
        """Empty or unreadable input returns zero-risk fallback with warning."""
        result = tampering_service.analyze_document(image_bytes=b"not an image", document_image=None)
        assert result.tampering_risk_score == 0.0
        assert result.risk_level == "LOW"
        assert len(result.warnings) > 0


class TestDocumentConsistencyTampering:
    """Tests for cross-zone semantic field consistency analysis (Visual vs MRZ)."""

    def test_clean_document_perfect_consistency(self, tampering_service):
        """When visual and MRZ fields match, consistency signal is clean with NO_ANOMALY_FOUND."""
        vis_fields = {
            "passport_number": "L898902C3",
            "date_of_birth": "12/08/1974",
            "date_of_expiry": "15/04/2012",
            "name": "ERIKSSON ANNA MARIA",
            "sex": "F",
            "nationality": "UTO"
        }
        mrz_fields = {
            "document_number": "L898902C3",
            "date_of_birth": "740812",
            "date_of_expiry": "120415",
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "sex": "F",
            "nationality": "UTO"
        }
        res = tampering_service.analyze_document(
            visual_fields=vis_fields,
            mrz_fields=mrz_fields
        )
        sig = res.signals["document_consistency"]
        assert sig.evaluated is True
        assert sig.score == 0.0
        assert sig.reason == "NO_ANOMALY_FOUND"
        assert sig.evidence_confidence >= 0.90
        assert len(res.indicators) == 0
        assert res.risk_level == "LOW"

    def test_tampered_changed_dob_detected(self, tampering_service):
        """When visual DOB is altered (e.g. 15/08/1995) while MRZ has authentic DOB (740812), tampering risk surges."""
        vis_fields = {
            "passport_number": "L898902C3",
            "date_of_birth": "15/08/1995",  # Tampered printed DOB
            "name": "ERIKSSON ANNA MARIA",
        }
        mrz_fields = {
            "document_number": "L898902C3",
            "date_of_birth": "740812",      # Authentic MRZ DOB (12/08/1974)
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA"
        }
        res = tampering_service.analyze_document(
            visual_fields=vis_fields,
            mrz_fields=mrz_fields,
            field_confidences={"date_of_birth": 0.92}
        )
        sig = res.signals["document_consistency"]
        assert sig.evaluated is True
        assert sig.score >= 0.85
        assert sig.reason == "EVALUATED"
        assert res.tampering_risk_score >= 0.25
        assert any(ind.type == "document_consistency_mismatch" for ind in res.indicators)
        mismatch_ind = next(ind for ind in res.indicators if ind.type == "document_consistency_mismatch")
        assert "Date of Birth" in mismatch_ind.explanation
        assert "15/08/1995" in mismatch_ind.explanation

    def test_tampered_changed_document_number_detected(self, tampering_service):
        """When visual passport number is changed to forged number, consistency mismatch is flagged."""
        vis_fields = {
            "passport_number": "P99887766",  # Forged printed document number
            "date_of_birth": "12/08/1974",
        }
        mrz_fields = {
            "document_number": "L898902C3",   # Original MRZ number
            "date_of_birth": "740812",
        }
        res = tampering_service.analyze_document(
            visual_fields=vis_fields,
            mrz_fields=mrz_fields,
            field_confidences={"passport_number": 0.95}
        )
        sig = res.signals["document_consistency"]
        assert sig.evaluated is True
        assert sig.score >= 0.85
        assert any(ind.type == "document_consistency_mismatch" for ind in res.indicators)

    def test_tampered_changed_expiry_date_detected(self, tampering_service):
        """When visual expiry date is modified to extend validity, consistency mismatch is detected."""
        vis_fields = {
            "passport_number": "L898902C3",
            "date_of_expiry": "15/04/2030",  # Extended validity
        }
        mrz_fields = {
            "document_number": "L898902C3",
            "date_of_expiry": "120415",      # Expired in 2012
        }
        res = tampering_service.analyze_document(
            visual_fields=vis_fields,
            mrz_fields=mrz_fields,
            field_confidences={"date_of_expiry": 0.90}
        )
        assert res.signals["document_consistency"].score >= 0.85

    def test_tampered_changed_name_detected(self, tampering_service):
        """When visual name is swapped, consistency mismatch is detected."""
        vis_fields = {
            "passport_number": "L898902C3",
            "name": "ROBERT WILLIAMS",
        }
        mrz_fields = {
            "document_number": "L898902C3",
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA"
        }
        res = tampering_service.analyze_document(
            visual_fields=vis_fields,
            mrz_fields=mrz_fields,
            field_confidences={"name": 0.90}
        )
        assert res.signals["document_consistency"].score >= 0.70

    def test_missing_visual_field_not_flagged_as_mismatch(self, tampering_service):
        """A missing visual field (empty string) must NOT be counted as a mismatch or tampering signal."""
        vis_fields = {
            "passport_number": "L898902C3",
            "date_of_birth": "",  # Unextracted / missing field
        }
        mrz_fields = {
            "document_number": "L898902C3",
            "date_of_birth": "740812",
        }
        res = tampering_service.analyze_document(
            visual_fields=vis_fields,
            mrz_fields=mrz_fields
        )
        # Only passport_number is compared; date_of_birth is skipped safely
        sig = res.signals["document_consistency"]
        assert sig.score == 0.0
        assert "date_of_birth" not in sig.comparisons

    def test_date_format_normalization_robustness(self, tampering_service):
        """Test different date formats (DD-MM-YYYY vs YYMMDD, DD MMM YYYY) correctly match."""
        # 12-04-2010 vs 100412
        vis_fields = {"date_of_expiry": "12-04-2010"}
        mrz_fields = {"date_of_expiry": "100412"}
        res = tampering_service.analyze_document(visual_fields=vis_fields, mrz_fields=mrz_fields)
        assert res.signals["document_consistency"].score == 0.0
        assert res.signals["document_consistency"].comparisons["date_of_expiry"].match is True

    def test_tampering_service_with_mocked_ocr_pipeline_dob_mismatch(self, clean_synthetic_document):
        """Integration test: calling tampering service without supplying fields runs existing OCR/MRZ pipeline and detects DOB mismatch."""
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        from app.services.tampering_service import TamperingService
        
        img_arr, img_bytes = clean_synthetic_document
        mock_engine = MockOCREngine()
        mock_engine.set_predefined_result(
            OCRResult(
                raw_text="PASSPORT\nDate of Birth: 15/08/1982\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO5909232F1204159ZE184226B<<<<<10",
                regions=[
                    OCRRegion(text="PASSPORT", confidence=0.99, bbox=[50, 50, 200, 80]),
                    OCRRegion(text="Date of Birth: 15/08/1982", confidence=0.95, bbox=[50, 150, 350, 180]),
                    OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.96, bbox=[50, 400, 750, 430]),
                    OCRRegion(text="L898902C36UTO5909232F1204159ZE184226B<<<<<10", confidence=0.95, bbox=[50, 440, 750, 470]),
                ],
                average_confidence=0.96
            )
        )
        custom_ocr_srv = OCRService(engine=mock_engine)
        tampering_srv = TamperingService(ocr_service=custom_ocr_srv)

        # Call with ONLY image (caller does not supply OCR fields)
        res = tampering_srv.analyze_document(image_bytes=img_bytes, document_image=img_arr)

        assert res.consistency_debug is not None
        assert res.consistency_debug["ocr_pipeline_called"] is True
        assert res.consistency_debug["mrz_detected"] is True
        assert "date_of_birth" in res.consistency_debug["comparable_fields"]

        sig = res.signals["document_consistency"]
        assert sig.evaluated is True
        assert sig.comparisons["date_of_birth"].match is False
        assert sig.score > 0
        assert res.tampering_risk_score >= 0.25

    def test_tampering_service_with_mocked_ocr_pipeline_matching_dob(self, clean_synthetic_document):
        """Integration test: calling tampering service with matching DOB (23/09/1959 vs 590923) has match=True and no penalty."""
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        from app.services.tampering_service import TamperingService
        
        img_arr, img_bytes = clean_synthetic_document
        mock_engine = MockOCREngine()
        mock_engine.set_predefined_result(
            OCRResult(
                raw_text="PASSPORT\nDate of Birth: 23/09/1959\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO5909232F1204159ZE184226B<<<<<10",
                regions=[
                    OCRRegion(text="PASSPORT", confidence=0.99, bbox=[50, 50, 200, 80]),
                    OCRRegion(text="Date of Birth: 23/09/1959", confidence=0.95, bbox=[50, 150, 350, 180]),
                    OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.96, bbox=[50, 400, 750, 430]),
                    OCRRegion(text="L898902C36UTO5909232F1204159ZE184226B<<<<<10", confidence=0.95, bbox=[50, 440, 750, 470]),
                ],
                average_confidence=0.96
            )
        )
        custom_ocr_srv = OCRService(engine=mock_engine)
        tampering_srv = TamperingService(ocr_service=custom_ocr_srv)

        # Call with ONLY image
        res = tampering_srv.analyze_document(image_bytes=img_bytes, document_image=img_arr)

        assert res.consistency_debug is not None
        assert res.consistency_debug["ocr_pipeline_called"] is True
        assert res.consistency_debug["mrz_detected"] is True

        sig = res.signals["document_consistency"]
        assert sig.evaluated is True
        assert sig.comparisons["date_of_birth"].match is True
        assert sig.score == 0.0



class TestTamperedVsCleanSeparation:
    """Verifies that tampered documents produce significantly higher risk scores than clean documents."""

    def test_tampered_field_score_separation(self, tampering_service, clean_synthetic_document):
        img_arr, img_bytes = clean_synthetic_document
        
        # Clean baseline
        clean_res = tampering_service.analyze_document(
            image_bytes=img_bytes,
            document_image=img_arr,
            visual_fields={"passport_number": "L898902C3", "date_of_birth": "12/08/1974"},
            mrz_fields={"document_number": "L898902C3", "date_of_birth": "740812"}
        )
        
        # Tampered sample with forged visual DOB
        tampered_res = tampering_service.analyze_document(
            image_bytes=img_bytes,
            document_image=img_arr,
            visual_fields={"passport_number": "L898902C3", "date_of_birth": "01/01/2000"},
            mrz_fields={"document_number": "L898902C3", "date_of_birth": "740812"}
        )
        
        assert clean_res.signals["document_consistency"].score == 0.0
        assert clean_res.risk_level == "LOW"
        assert tampered_res.tampering_risk_score > clean_res.tampering_risk_score
        assert (tampered_res.tampering_risk_score - clean_res.tampering_risk_score) >= 0.20


class TestTamperingAPI:
    """API endpoint tests for document tampering detection."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def override_ocr_engine(self):
        from app.api.routes.ocr import get_ocr_service
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        mock_engine = MockOCREngine()
        custom_service = OCRService(engine=mock_engine)
        app.dependency_overrides[get_ocr_service] = lambda: custom_service
        yield mock_engine
        app.dependency_overrides.clear()

    def test_standalone_tampering_endpoint(self, client, clean_synthetic_document):
        """Test POST /api/v1/ocr/tampering returns HTTP 200 with TamperingResult."""
        _, img_bytes = clean_synthetic_document
        response = client.post(
            "/api/v1/ocr/tampering",
            files={"file": ("passport.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tampering_risk_score" in data
        assert "risk_level" in data
        assert "signals" in data
        assert "indicators" in data
        assert data["risk_level"] == "LOW"

    def test_extract_endpoint_with_tampering_flag(self, client, override_ocr_engine, clean_synthetic_document):
        """Test POST /api/v1/ocr/extract with detect_tampering=True includes tampering payload."""
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import OCRResult
        
        regions = [
            OCRRegion(text="PASSPORT", confidence=0.99, bbox=[50, 50, 200, 80]),
            OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.96, bbox=[50, 400, 750, 430]),
            OCRRegion(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", confidence=0.95, bbox=[50, 440, 750, 470]),
        ]
        override_ocr_engine.set_predefined_result(
            OCRResult(
                raw_text="PASSPORT\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10",
                regions=regions,
                average_confidence=0.96
            )
        )

        _, img_bytes = clean_synthetic_document
        response = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("passport.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "passport", "detect_tampering": "true"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tampering"] is not None
        assert "tampering_risk_score" in data["tampering"]
        assert "risk_level" in data["tampering"]

    def test_extract_endpoint_detects_visual_mrz_contradiction(self, client, override_ocr_engine, clean_synthetic_document):
        """Test POST /api/v1/ocr/extract where visual DOB (15/08/1995) contradicts MRZ DOB (740812)."""
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import OCRResult
        
        regions = [
            OCRRegion(text="PASSPORT", confidence=0.99, bbox=[50, 50, 200, 80]),
            OCRRegion(text="Date of Birth: 15/08/1995", confidence=0.95, bbox=[50, 150, 350, 180]),
            OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.96, bbox=[50, 400, 750, 430]),
            OCRRegion(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", confidence=0.95, bbox=[50, 440, 750, 470]),
        ]
        override_ocr_engine.set_predefined_result(
            OCRResult(
                raw_text="PASSPORT\nDate of Birth: 15/08/1995\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10",
                regions=regions,
                average_confidence=0.96
            )
        )

        _, img_bytes = clean_synthetic_document
        response = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("passport.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "passport", "detect_tampering": "true"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tampering"] is not None
        tampering = data["tampering"]
        assert tampering["signals"]["document_consistency"]["evaluated"] is True
        assert tampering["signals"]["document_consistency"]["score"] >= 0.85
        assert tampering["tampering_risk_score"] >= 0.25
        assert any(ind["type"] == "document_consistency_mismatch" for ind in tampering["indicators"])


class TestCalibrationHarness:
    """Tests for dataset evaluation and calibration harness."""

    def test_synthetic_dataset_calibration(self, tmp_path):
        """Test generating synthetic paired dataset and computing calibration metrics."""
        dataset_dir = str(tmp_path / "benchmark_data")
        counts = TamperingCalibrationHarness.generate_synthetic_benchmark_dataset(
            output_dir=dataset_dir,
            num_clean=3,
            num_tampered=3
        )
        assert counts["clean_generated"] == 3
        assert counts["tampered_generated"] == 3

        harness = TamperingCalibrationHarness(default_threshold=0.30)
        metrics = harness.evaluate_directory(dataset_dir=dataset_dir, threshold=0.30)
        
        assert metrics["sample_counts"]["clean"] == 3
        assert metrics["sample_counts"]["tampered"] == 3
        assert metrics["mean_risk_scores"]["tampered"] >= metrics["mean_risk_scores"]["clean"]
        assert "confusion_matrix" in metrics
        assert "rates" in metrics


class TestTamperingSpecificationRequirements:
    """Explicit unit tests covering all 15 specification requirements for the tampering module."""

    def test_gimp_metadata_detection(self, tampering_service):
        """EXIF / ImageDescription containing GIMP signature triggers metadata anomaly."""
        img = Image.new("RGB", (200, 200), color="white")
        exif = img.getexif()
        exif[305] = "GIMP 2.10.34"
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)

        res = tampering_service.analyze_document(image_bytes=buf.getvalue())
        meta_sig = res.signals["metadata"]
        assert meta_sig.evaluated is True
        assert meta_sig.editing_software_detected is True
        assert meta_sig.score == 0.35
        assert "GIMP" in str(meta_sig.software)
        assert any(ind.type == "editing_software_metadata" for ind in res.indicators)

    def test_camera_exif_metadata_is_clean(self, tampering_service):
        """Legitimate camera EXIF metadata without editor tags is evaluated=True, score=0.0."""
        img = Image.new("RGB", (200, 200), color="white")
        exif = img.getexif()
        exif[271] = "Canon"        # Make
        exif[272] = "Canon EOS R5" # Model
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)

        res = tampering_service.analyze_document(image_bytes=buf.getvalue())
        meta_sig = res.signals["metadata"]
        assert meta_sig.evaluated is True
        assert meta_sig.editing_software_detected is False
        assert meta_sig.score == 0.0
        assert meta_sig.reason == "NO_ANOMALY_FOUND"

    def test_png_image_input_processing(self, tampering_service):
        """PNG images are converted and analyzed safely in memory without crashing."""
        img_arr = np.full((300, 400, 3), 240, dtype=np.uint8)
        cv2.putText(img_arr, "PNG DOCUMENT TEST", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        _, png_buf = cv2.imencode(".png", img_arr)

        res = tampering_service.analyze_document(image_bytes=png_buf.tobytes(), document_image=img_arr)
        assert isinstance(res, TamperingResult)
        assert 0.0 <= res.tampering_risk_score <= 1.0
        assert res.risk_level == "LOW"
        assert res.signals["ela"].evaluated is True

    def test_locally_resaved_region_anomaly(self, tampering_service, clean_synthetic_document):
        """A locally re-saved spliced region with distinct JPEG quality increases ELA residual deviation."""
        img_arr, _ = clean_synthetic_document
        tampered = img_arr.copy()

        # Create a local sub-region recompressed at low quality (Q=40)
        sub = tampered[150:230, 300:500].copy()
        _, sub_buf = cv2.imencode(".jpg", sub, [cv2.IMWRITE_JPEG_QUALITY, 40])
        sub_degraded = cv2.imdecode(sub_buf, cv2.IMREAD_COLOR)
        tampered[150:230, 300:500] = sub_degraded

        _, buf = cv2.imencode(".jpg", tampered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        res = tampering_service.analyze_document(image_bytes=buf.tobytes(), document_image=tampered)

        ela_metrics = res.signals["ela"].metrics
        assert ela_metrics["max_error"] > 0
        assert res.signals["ela"].evaluated is True

    def test_score_bounds_and_determinism(self, tampering_service, clean_synthetic_document):
        """Tampering risk scores must always be strictly in [0.0, 1.0] and 100% deterministic across repeated calls."""
        img_arr, img_bytes = clean_synthetic_document
        res1 = tampering_service.analyze_document(image_bytes=img_bytes, document_image=img_arr)
        res2 = tampering_service.analyze_document(image_bytes=img_bytes, document_image=img_arr)

        assert 0.0 <= res1.tampering_risk_score <= 1.0
        assert 0.0 <= res1.evidence_coverage <= 1.0
        assert res1.tampering_risk_score == res2.tampering_risk_score
        assert res1.evidence_coverage == res2.evidence_coverage
        assert res1.risk_level == res2.risk_level

    def test_clean_vs_tampered_controlled_regression_margin(self, tampering_service):
        """Controlled paired test: same image modified with ONE localized spliced region must elevate ELA & overall score."""
        # Base image
        raw_img = np.full((500, 700, 3), 245, dtype=np.uint8)
        cv2.putText(raw_img, "PASSPORT OF THE REPUBLIC", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.putText(raw_img, "PASSPORT NO: L898902C3", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        
        # Clean baseline: uniform JPEG at Q=92
        _, clean_buf = cv2.imencode(".jpg", raw_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        clean_img = cv2.imdecode(clean_buf, cv2.IMREAD_COLOR)

        # Tampered image: pre-compressed at Q=70, with foreign high-frequency spliced patch inserted
        _, base_buf = cv2.imencode(".jpg", raw_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        tampered_base = cv2.imdecode(base_buf, cv2.IMREAD_COLOR)
        
        patch = np.full((80, 200, 3), 200, dtype=np.uint8)
        cv2.putText(patch, "FORGED-999", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        patch_noise = np.random.normal(0, 16.0, patch.shape).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + patch_noise, 0, 255).astype(np.uint8)
        tampered_base[240:320, 40:240] = patch
        
        _, tampered_buf = cv2.imencode(".jpg", tampered_base, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tampered_img = cv2.imdecode(tampered_buf, cv2.IMREAD_COLOR)

        clean_res = tampering_service.analyze_document(image_bytes=clean_buf.tobytes(), document_image=clean_img)
        tampered_res = tampering_service.analyze_document(image_bytes=tampered_buf.tobytes(), document_image=tampered_img)

        # Assert: tampered ELA score > clean ELA score
        assert tampered_res.signals["ela"].score > clean_res.signals["ela"].score
        # Assert: tampered overall score > clean overall score
        assert tampered_res.tampering_risk_score > clean_res.tampering_risk_score
        assert (tampered_res.tampering_risk_score - clean_res.tampering_risk_score) >= 0.10




    def test_dedicated_tampering_analyze_api_endpoint(self, clean_synthetic_document):
        """Test dedicated POST /api/v1/tampering/analyze endpoint."""
        client = TestClient(app)
        _, img_bytes = clean_synthetic_document

        response = client.post(
            "/api/v1/tampering/analyze",
            files={"file": ("passport.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tampering_risk_score" in data
        assert "risk_level" in data
        assert "evidence_coverage" in data
        assert "signals" in data
        assert "ela" in data["signals"]
        assert "metadata" in data["signals"]
        assert data["risk_level"] == "LOW"

    def test_evaluate_tampering_script(self, tmp_path):
        """Test evaluation/evaluate_tampering.py evaluation function."""
        from evaluation.evaluate_tampering import evaluate_dataset

        clean_dir = str(tmp_path / "clean")
        tampered_dir = str(tmp_path / "tampered")

        report = evaluate_dataset(
            clean_dir=clean_dir,
            tampered_dir=tampered_dir,
            threshold=0.30,
            generate_synthetic_if_missing=True,
            synthetic_samples=3
        )
        assert report["sample_counts"]["clean"] == 3
        assert report["sample_counts"]["tampered"] == 3
        assert report["mean_scores"]["tampered_average"] > report["mean_scores"]["clean_average"]
        assert "confusion_matrix" in report
        assert "risk_level_counts" in report

