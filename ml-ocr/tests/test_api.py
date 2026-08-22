"""API integration tests for SIH OCR Microservice."""

import io
import pytest
from fastapi.testclient import TestClient

from app.api.routes.ocr import get_ocr_service
from app.main import app
from app.models.schemas import OCRRegion
from app.services.ocr_service import MockOCREngine, OCRResult, OCRService


@pytest.fixture
def override_ocr_engine():
    """Overrides the OCR engine in the API with a mock engine for deterministic testing."""
    mock_engine = MockOCREngine()
    custom_service = OCRService(engine=mock_engine)
    app.dependency_overrides[get_ocr_service] = lambda: custom_service
    yield mock_engine
    app.dependency_overrides.clear()


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check_root(self, test_client: TestClient):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "available_ocr_engines" in data

    def test_health_check_v1(self, test_client: TestClient):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestOCRExtractEndpoint:
    """Integration tests for POST /api/v1/ocr/extract."""

    def test_extract_passport_success(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        # Configure mock OCR output for valid TD3 passport
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

        img_bytes = create_dummy_image(text="PASSPORT")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("passport.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "passport"}
        )

        assert response.status_code == 200
        data = response.json()
        
        # Verify JSON structure matches specifications
        assert data["success"] is True
        assert data["document_type"] == "passport"
        assert 0.0 <= data["average_confidence"] <= 1.0
        assert "ERIKSSON" in data["extracted_text"]
        
        # Fields
        fields = data["fields"]
        assert fields["surname"] == "ERIKSSON"
        assert fields["given_names"] == "ANNA MARIA"
        assert fields["passport_number"] == "L898902C3"
        assert fields["nationality"] == "UTO"
        assert fields["date_of_birth"] == "740812"
        assert fields["sex"] == "F"
        assert fields["date_of_expiry"] == "120415"
        
        # MRZ checks
        mrz = data["mrz"]
        assert mrz["detected"] is True
        assert mrz["valid_format"] is True
        assert mrz["overall_valid"] is True
        assert mrz["check_digits"]["passport_number"] is True
        assert mrz["check_digits"]["date_of_birth"] is True
        assert mrz["check_digits"]["date_of_expiry"] is True
        assert mrz["check_digits"]["composite"] is True

        # Regions and processing
        assert len(data["ocr_regions"]) == 3
        assert data["processing"]["preprocessing_applied"] is True

    def test_extract_auto_detection_passport(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        # When document_type is 'auto' and MRZ contains P<
        regions = [
            OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.96, bbox=[50, 400, 750, 430]),
            OCRRegion(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", confidence=0.95, bbox=[50, 440, 750, 470]),
        ]
        override_ocr_engine.set_predefined_result(
            OCRResult(
                raw_text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10",
                regions=regions,
                average_confidence=0.95
            )
        )

        img_bytes = create_dummy_image(text="SAMPLE")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("doc.png", img_bytes, "image/png")},
            data={"document_type": "auto"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "passport"

    def test_extract_visa_document(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        visa_text = (
            "UNITED STATES OF AMERICA VISA\n"
            "Control Number: 202312345678\n"
            "Visa No: V9876543\n"
            "Name: SMITH JOHN\n"
            "Passport No: P12345678\n"
            "Nationality: USA\n"
            "Entries: MULTIPLE\n"
            "Type / Class: B1/B2\n"
            "Issue Date: 15/01/2023\n"
            "Expiry Date: 14/01/2033\n"
            "Authority: US EMBASSY LONDON"
        )
        regions = [OCRRegion(text=line, confidence=0.94, bbox=[10, 10 + i * 20, 200, 30 + i * 20]) for i, line in enumerate(visa_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=visa_text, regions=regions, average_confidence=0.94)
        )

        img_bytes = create_dummy_image(text="USA VISA")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("visa.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "visa"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "visa"
        assert data["fields"]["visa_number"] == "V9876543" or "202312345678" in data["fields"]["visa_number"]
        assert data["fields"]["name"] == "SMITH JOHN"
        assert data["fields"]["passport_number"] == "P12345678"
        assert data["fields"]["visa_type"] == "B1/B2"
        assert data["fields"]["entries"] == "MULTIPLE"
        assert data["mrz"]["detected"] is False

    def test_extract_national_id_document(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        id_text = (
            "GOVERNMENT IDENTITY CARD\n"
            "Name: ROBERT WILLIAMS\n"
            "ID No: 9876 5432 1098\n"
            "DOB: 12/08/1985\n"
            "Gender: MALE\n"
            "Nationality: INDIAN\n"
            "Address: 123 Baker Street, London"
        )
        regions = [OCRRegion(text=line, confidence=0.92, bbox=[10, 10 + i * 20, 200, 30 + i * 20]) for i, line in enumerate(id_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=id_text, regions=regions, average_confidence=0.92)
        )

        img_bytes = create_dummy_image(text="NATIONAL ID")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("id.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "national_id"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "national_id"
        assert data["fields"]["name"] == "ROBERT WILLIAMS"
        assert data["fields"]["id_number"] == "9876 5432 1098"
        assert data["fields"]["date_of_birth"] == "12/08/1985"
        assert data["fields"]["gender"] == "MALE"
        assert data["fields"]["nationality"] == "INDIAN"
        assert "123 Baker Street" in data["fields"]["address"]

    def test_error_invalid_document_type(self, test_client: TestClient, create_dummy_image):
        img_bytes = create_dummy_image()
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("doc.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "invalid_type_name"}
        )
        assert response.status_code == 400
        assert "Invalid document_type" in response.json()["detail"]

    def test_error_empty_upload(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"document_type": "auto"}
        )
        assert response.status_code == 400
        assert "Uploaded file is empty" in response.json()["detail"]

    def test_error_unsupported_file_extension(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("script.py", b"print('hello')", "text/x-python")},
            data={"document_type": "auto"}
        )
        assert response.status_code == 400
        assert "Unsupported file extension" in response.json()["detail"]

    def test_error_corrupted_image(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("corrupted.jpg", b"SOME_CORRUPTED_BYTES_HERE", "image/jpeg")},
            data={"document_type": "auto"}
        )
        assert response.status_code == 400
        assert "Corrupted or invalid image file" in response.json()["detail"]

    def test_extract_passport_with_debug_mode(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        regions = [
            OCRRegion(text="PASSPORT", confidence=0.99, bbox=[10, 10, 100, 30]),
            OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.98, bbox=[10, 400, 700, 430]),
            OCRRegion(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", confidence=0.97, bbox=[10, 440, 700, 470]),
        ]
        override_ocr_engine.set_predefined_result(
            OCRResult(
                raw_text="PASSPORT\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10",
                regions=regions,
                average_confidence=0.96
            )
        )

        img_bytes = create_dummy_image(text="PASSPORT")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("passport.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "passport", "include_debug": "true"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mrz_debug"] is not None
        assert "candidate_count" in data["mrz_debug"]
        assert "best_candidate" in data["mrz_debug"]
        assert "candidate_scores" in data["mrz_debug"]
        assert data["mrz_debug"]["best_candidate"]["valid_format"] is True

    def test_extract_visa_document_with_mrva(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction for a visa containing MRV-A machine-readable lines."""
        visa_text = (
            "UNITED STATES VISA\n"
            "VISA TYPE: B1/B2\n"
            "ENTRIES: M\n"
            "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO7408122F1204159<<<<<<<<<<<<<<<<"
        )
        regions = [OCRRegion(text=line, confidence=0.95, bbox=[10, 10 + i * 20, 400, 30 + i * 20]) for i, line in enumerate(visa_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=visa_text, regions=regions, average_confidence=0.95)
        )

        img_bytes = create_dummy_image(text="VISA MRVA")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("visa.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "auto"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "visa"
        assert data["mrz"]["detected"] is True
        assert data["mrz"]["format"] == "MRVA"
        assert data["mrz"]["overall_valid"] is True
        assert data["fields"]["surname"] == "ERIKSSON"
        assert data["fields"]["visa_number"] == "L898902C3"
        assert data["fields"]["visa_type"] == "B1/B2"
        assert "field_sources" in data and data["field_sources"] is not None
        assert data["field_sources"]["visa_type"]["source"] == "visual_ocr"
        assert data["field_sources"]["visa_number"]["source"] == "mrz"

    def test_extract_pure_visual_visa_without_mrz(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction for a visa without MRZ lines using visual-zone key-value extraction."""
        visa_text = (
            "UNITED STATES OF AMERICA VISA\n"
            "Control Number: 202315904812\n"
            "Visa Type / Class: B1/B2\n"
            "Entries: MULTIPLE\n"
            "Issue Date: 15/01/2023\n"
            "Expiry Date: 14/01/2033\n"
            "Passport No: P98765432\n"
            "Name: ROBERT WILLIAMS\n"
            "Authority: US EMBASSY LONDON"
        )
        regions = [OCRRegion(text=line, confidence=0.94, bbox=[10, 10 + i * 20, 400, 30 + i * 20]) for i, line in enumerate(visa_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=visa_text, regions=regions, average_confidence=0.94)
        )

        img_bytes = create_dummy_image(text="VISUAL VISA ONLY")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("visa.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "visa"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "visa"
        assert data["mrz"]["detected"] is False
        assert data["fields"]["visa_number"] == "202315904812"
        assert data["fields"]["visa_type"] == "B1/B2"
        assert data["fields"]["entries"] == "MULTIPLE"
        assert data["fields"]["issue_date"] == "15/01/2023"
        assert data["fields"]["expiry_date"] == "14/01/2033"
        assert data["fields"]["passport_number"] == "P98765432"
        assert data["fields"]["name"] == "ROBERT WILLIAMS"
        assert data["field_sources"]["visa_type"]["source"] == "visual_ocr"
        assert data["field_sources"]["visa_number"]["source"] == "visual_ocr"


    def test_extract_visa_document_with_invalid_mrz_fields(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction for a visa containing MRZ with invalid nationality, DOB and expiry fields."""
        visa_text = (
            "UNITED STATES OF AMERICA VISA\n"
            "V<USAJOHNSON<<ROBERT<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "VJDEHE5CK1LAUS931014<11706262AUS<<<<<<<<<<<<"
        )
        regions = [OCRRegion(text=line, confidence=0.92, bbox=[10, 10 + i * 20, 400, 30 + i * 20]) for i, line in enumerate(visa_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=visa_text, regions=regions, average_confidence=0.92)
        )

        img_bytes = create_dummy_image(text="VISA INVALID FIELDS")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("visa.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "visa"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "visa"
        assert data["mrz"]["detected"] is True
        assert data["mrz"]["format"] == "MRVA"
        assert data["mrz"]["overall_valid"] is False
        assert data["mrz"]["field_validation"]["nationality"]["valid"] is False
        assert data["mrz"]["field_validation"]["nationality"]["value"] == "LAU"
        assert data["mrz"]["field_validation"]["date_of_birth"]["valid"] is False
        assert data["mrz"]["field_validation"]["date_of_birth"]["value"] == "S93101"
        assert data["mrz"]["field_validation"]["date_of_expiry"]["valid"] is False
        assert data["mrz"]["field_validation"]["date_of_expiry"]["value"] == "117062"


    def test_extract_national_id_with_td1_mrz(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction for a National ID containing TD1 3-line MRZ."""
        id_text = (
            "EUROPEAN UNION IDENTITY CARD\n"
            "I<UTOD231458907<<<<<<<<<<<<<<<\n"
            "7408122F1204159UTO<<<<<<<<<<<6\n"
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"
        )
        regions = [OCRRegion(text=line, confidence=0.94, bbox=[10, 10 + i * 20, 300, 30 + i * 20]) for i, line in enumerate(id_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=id_text, regions=regions, average_confidence=0.94)
        )

        img_bytes = create_dummy_image(text="NATIONAL ID TD1")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("id.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "auto"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "national_id"
        assert data["mrz"]["detected"] is True
        assert data["mrz"]["format"] == "TD1"
        assert data["mrz"]["overall_valid"] is True
        assert data["mrz"]["line3"] is not None
        assert data["fields"]["id_number"] == "D23145890"

    def test_extract_national_id_without_mrz_has_warnings(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction for a National ID without MRZ returns fields and informational warnings."""
        id_text = (
            "GOVERNMENT IDENTITY CARD\n"
            "Name: ROBERT WILLIAMS\n"
            "ID No: 9876 5432 1098\n"
            "DOB: 12/08/1985\n"
            "Gender: MALE\n"
            "Nationality: INDIAN\n"
            "Address: 123 Baker Street, London"
        )
        regions = [OCRRegion(text=line, confidence=0.92, bbox=[10, 10 + i * 20, 200, 30 + i * 20]) for i, line in enumerate(id_text.splitlines())]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=id_text, regions=regions, average_confidence=0.92)
        )

        img_bytes = create_dummy_image(text="NATIONAL ID NO MRZ")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("id.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "national_id"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "national_id"
        assert data["mrz"]["detected"] is False
        assert len(data["warnings"]) > 0
        assert "MRZ not detected" in data["warnings"][0]

    def test_extract_real_national_id_layout_api(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction for a real-world Aadhaar/National ID card layout with unlabeled name."""
        id_text = (
            "GOVERNMENT OF INDIA\n"
            "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\n"
            "Sriram Mamundi\n"
            "DOB: 15/08/1990\n"
            "Male\n"
            "8416 1590 3267"
        )
        regions = [
            OCRRegion(text="GOVERNMENT OF INDIA", confidence=0.98, bbox=[50, 20, 450, 50]),
            OCRRegion(text="UNIQUE IDENTIFICATION AUTHORITY OF INDIA", confidence=0.96, bbox=[50, 55, 480, 80]),
            OCRRegion(text="Sriram Mamundi", confidence=0.94, bbox=[50, 100, 280, 130]),
            OCRRegion(text="DOB: 15/08/1990", confidence=0.95, bbox=[50, 140, 250, 165]),
            OCRRegion(text="Male", confidence=0.97, bbox=[50, 175, 120, 200]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 220, 320, 255]),
        ]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=id_text, regions=regions, average_confidence=0.965)
        )

        img_bytes = create_dummy_image(text="AADHAAR REAL SAMPLE")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("aadhaar.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "auto"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "national_id"
        assert data["mrz"]["detected"] is False
        assert data["fields"]["name"] == "Sriram Mamundi"
        assert data["fields"]["date_of_birth"] == "15/08/1990"
        assert data["fields"]["gender"] == "MALE"
        assert data["fields"]["sex"] == "M"
        assert data["fields"]["id_number"] == "8416 1590 3267"
        assert data["fields"]["issuing_authority"] == "Government of India"
        assert data["field_confidences"]["name"] == 0.94
        assert data["field_confidences"]["id_number"] == 0.99

    def test_extract_national_id_with_field_debug_api(self, test_client: TestClient, override_ocr_engine: MockOCREngine, create_dummy_image):
        """Test API extraction with include_debug=true returns field_debug with candidate lists."""
        id_text = (
            "GOVERNMENT OF INDIA\n"
            "Sriram Mamundi\n"
            "DOB: 15/08/1990\n"
            "Male\n"
            "8416 1590 3267"
        )
        regions = [
            OCRRegion(text="GOVERNMENT OF INDIA", confidence=0.98, bbox=[50, 20, 450, 50]),
            OCRRegion(text="Sriram Mamundi", confidence=0.94, bbox=[50, 100, 280, 130]),
            OCRRegion(text="DOB", confidence=0.95, bbox=[50, 140, 90, 165]),
            OCRRegion(text="15/08/1990", confidence=0.96, bbox=[100, 140, 220, 165]),
            OCRRegion(text="Gender: Male", confidence=0.97, bbox=[50, 175, 180, 200]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 220, 320, 255]),
        ]
        override_ocr_engine.set_predefined_result(
            OCRResult(raw_text=id_text, regions=regions, average_confidence=0.96)
        )

        img_bytes = create_dummy_image(text="FIELD DEBUG SAMPLE")
        response = test_client.post(
            "/api/v1/ocr/extract",
            files={"file": ("aadhaar_debug.jpg", img_bytes, "image/jpeg")},
            data={"document_type": "national_id", "include_debug": "true"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["field_debug"] is not None
        assert "dob_candidates" in data["field_debug"]
        assert "gender_candidates" in data["field_debug"]




