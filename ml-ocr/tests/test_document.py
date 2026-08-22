"""Tests for DocumentService classification and field extraction across Passports, Visas, and National IDs."""

import pytest
from app.models.schemas import DocumentTypeEnum, MRZResult
from app.services.document_service import DocumentService, PassportExtractor, VisaExtractor, NationalIDExtractor


@pytest.fixture
def doc_service():
    return DocumentService()


class TestDocumentClassification:
    """Tests for explainable document type classification."""

    def test_classify_passport_by_mrz_format(self, doc_service):
        mrz_res = MRZResult(
            detected=True,
            format="TD3",
            document_code="P<",
            valid_format=True,
            overall_valid=True
        )
        doc_type, conf = doc_service.detect_document_type("Some OCR text", mrz_res)
        assert doc_type == DocumentTypeEnum.PASSPORT.value
        assert conf >= 0.95

    def test_classify_visa_by_mrz_format(self, doc_service):
        mrz_res = MRZResult(
            detected=True,
            format="MRVA",
            document_code="V<",
            valid_format=True,
            overall_valid=True
        )
        doc_type, conf = doc_service.detect_document_type("Some OCR text", mrz_res)
        assert doc_type == DocumentTypeEnum.VISA.value
        assert conf >= 0.95

    def test_classify_national_id_by_td1_mrz(self, doc_service):
        mrz_res = MRZResult(
            detected=True,
            format="TD1",
            document_code="I<",
            valid_format=True,
            overall_valid=True
        )
        doc_type, conf = doc_service.detect_document_type("Some OCR text", mrz_res)
        assert doc_type == DocumentTypeEnum.NATIONAL_ID.value
        assert conf >= 0.95

    def test_classify_visa_by_keywords(self, doc_service):
        mrz_res = MRZResult(detected=False, valid_format=False, overall_valid=False)
        text = "UNITED STATES VISA\nCONTROL NUMBER: 2024098124\nENTRIES: MULTIPLE\nVISA TYPE: B1/B2"
        doc_type, conf = doc_service.detect_document_type(text, mrz_res)
        assert doc_type == DocumentTypeEnum.VISA.value
        assert conf >= 0.90

    def test_classify_passport_by_keywords(self, doc_service):
        mrz_res = MRZResult(detected=False, valid_format=False, overall_valid=False)
        text = "PASSEPORT / PASSPORT\nREPUBLIC OF INDIA\nPASSPORT NO: Z1234567"
        doc_type, conf = doc_service.detect_document_type(text, mrz_res)
        assert doc_type == DocumentTypeEnum.PASSPORT.value
        assert conf >= 0.90

    def test_classify_national_id_by_keywords(self, doc_service):
        mrz_res = MRZResult(detected=False, valid_format=False, overall_valid=False)
        text = "GOVERNMENT OF INDIA\nUNIQUE IDENTIFICATION AUTHORITY\nAADHAAR CARD\n9876 5432 1098"
        doc_type, conf = doc_service.detect_document_type(text, mrz_res)
        assert doc_type == DocumentTypeEnum.NATIONAL_ID.value
        assert conf >= 0.90

    def test_classify_unknown_fallback(self, doc_service):
        mrz_res = MRZResult(detected=False, valid_format=False, overall_valid=False)
        text = "SOME ARBITRARY RECEIPT TEXT WITH NO RECOGNIZABLE HEADERS"
        doc_type, conf = doc_service.detect_document_type(text, mrz_res)
        assert doc_type == DocumentTypeEnum.NATIONAL_ID.value
        assert conf <= 0.60


class TestTypeSpecificExtractors:
    """Tests for PassportExtractor, VisaExtractor, and NationalIDExtractor."""

    def test_passport_extractor_mrz_primary(self):
        extractor = PassportExtractor()
        mrz_fields = {
            "surname": "SMITH",
            "given_names": "JOHN",
            "passport_number": "A12345678",
            "nationality": "USA",
            "date_of_birth": "800101",
            "sex": "M",
            "date_of_expiry": "300101",
            "issuing_state": "USA"
        }
        fields, confs = extractor.extract_fields("Visual text not used when MRZ present", mrz_fields, mrz_format="TD3")
        assert fields["surname"] == "SMITH"
        assert fields["passport_number"] == "A12345678"
        assert fields["mrz_format"] == "TD3"

    def test_visa_extractor_with_mrv_and_visual_merge(self):
        extractor = VisaExtractor()
        mrz_fields = {
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "document_number": "L898902C3",
            "nationality": "UTO",
            "date_of_birth": "740812",
            "sex": "F",
            "date_of_expiry": "120415",
            "issuing_state": "UTO"
        }
        ocr_text = "VISA TYPE: B1/B2\nENTRIES: MULTIPLE\nISSUE DATE: 12/04/2010\nISSUING POST: LONDON"
        fields, confs = extractor.extract_fields(ocr_text, mrz_fields, mrz_format="MRVA")
        assert fields["name"] == "ERIKSSON ANNA MARIA"
        assert fields["visa_number"] == "L898902C3"
        assert fields["visa_type"] == "B1/B2"
        assert fields["entries"] == "MULTIPLE"
        assert fields["issue_date"] == "12/04/2010"
        assert fields["issuing_authority"] == "LONDON"
        assert fields["mrz_format"] == "MRVA"

    def test_national_id_extractor_with_td1_mrz(self):
        extractor = NationalIDExtractor()
        mrz_fields = {
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "document_number": "D23145890",
            "nationality": "UTO",
            "date_of_birth": "740812",
            "sex": "F",
            "gender": "F",
            "date_of_expiry": "120415",
            "issuing_state": "UTO"
        }
        fields, confs = extractor.extract_fields("Visual text", mrz_fields, mrz_format="TD1")
        assert fields["name"] == "ERIKSSON ANNA MARIA"
        assert fields["id_number"] == "D23145890"
        assert fields["gender"] == "F"
        assert fields["mrz_format"] == "TD1"

    def test_national_id_extractor_real_sample_unlabeled_name(self):
        """Test extraction for real-world National ID layout with unlabeled name and 12-digit grouped ID."""
        extractor = NationalIDExtractor()
        ocr_text = (
            "GOVERNMENT OF INDIA\n"
            "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\n"
            "Sriram Mamundi\n"
            "DOB: 15/08/1990\n"
            "Male\n"
            "8416 1590 3267"
        )
        from app.models.schemas import OCRRegion
        regions = [
            OCRRegion(text="GOVERNMENT OF INDIA", confidence=0.98, bbox=[50, 20, 450, 50]),
            OCRRegion(text="UNIQUE IDENTIFICATION AUTHORITY OF INDIA", confidence=0.96, bbox=[50, 55, 480, 80]),
            OCRRegion(text="Sriram Mamundi", confidence=0.94, bbox=[50, 100, 280, 130]),
            OCRRegion(text="DOB: 15/08/1990", confidence=0.95, bbox=[50, 140, 250, 165]),
            OCRRegion(text="Male", confidence=0.97, bbox=[50, 175, 120, 200]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 220, 320, 255]),
        ]
        fields, confs = extractor.extract_fields(ocr_text, mrz_fields={}, mrz_format=None, ocr_regions=regions)
        assert fields["name"] == "Sriram Mamundi"
        assert fields["date_of_birth"] == "15/08/1990"
        assert fields["gender"] == "MALE"
        assert fields["sex"] == "M"
        assert fields["id_number"] == "8416 1590 3267"
        assert fields["issuing_authority"] == "Government of India"
        assert confs["name"] == 0.94
        assert confs["id_number"] == 0.99

    def test_national_id_extractor_spatial_adjacent_labels(self):
        """Test extraction when labels and values are located in separate bounding boxes."""
        extractor = NationalIDExtractor()
        from app.models.schemas import OCRRegion
        regions = [
            OCRRegion(text="NATIONAL IDENTITY CARD", confidence=0.95, bbox=[50, 10, 300, 35]),
            # Label on left, Value to the right
            OCRRegion(text="Full Name:", confidence=0.95, bbox=[50, 60, 150, 85]),
            OCRRegion(text="Elena Rostova", confidence=0.92, bbox=[160, 60, 320, 85]),
            # Label on left, DOB to the right
            OCRRegion(text="DOB:", confidence=0.95, bbox=[50, 95, 100, 120]),
            OCRRegion(text="24-11-1988", confidence=0.93, bbox=[110, 95, 230, 120]),
            # Label on left, Sex to the right
            OCRRegion(text="Gender:", confidence=0.95, bbox=[50, 130, 120, 155]),
            OCRRegion(text="Female", confidence=0.96, bbox=[130, 130, 200, 155]),
            # ID number below
            OCRRegion(text="Card No: ID-78901234", confidence=0.94, bbox=[50, 170, 280, 195]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        fields, confs = extractor.extract_fields(ocr_text, mrz_fields={}, mrz_format=None, ocr_regions=regions)
        assert fields["name"] == "Elena Rostova"
        assert fields["date_of_birth"] == "24-11-1988"
        assert fields["gender"] == "FEMALE"
        assert fields["sex"] == "F"
        assert fields["id_number"] == "ID-78901234"

    def test_national_id_date_formats(self):
        """Test handling of various valid date formats (DD/MM/YYYY, YYYY-MM-DD, YYYY)."""
        extractor = NationalIDExtractor()
        # YYYY-MM-DD
        f1, _ = extractor.extract_fields("ID: 12345\nBirth Date: 1994-06-25\nSex: F", {})
        assert f1["date_of_birth"] == "1994-06-25"
        
        # DD.MM.YYYY
        f2, _ = extractor.extract_fields("ID: 12345\nDOB: 12.04.1982\nSex: M", {})
        assert f2["date_of_birth"] == "12.04.1982"
        
        # Year of Birth only
        f3, _ = extractor.extract_fields("ID: 12345\nYear of Birth: 1978\nGender: Male", {})
        assert f3["date_of_birth"] == "1978"

    def test_national_id_missing_fields_not_fabricated(self):
        """Ensure missing fields are returned as empty strings and not fabricated."""
        extractor = NationalIDExtractor()
        f, _ = extractor.extract_fields("GOVERNMENT CARD\nDOCUMENT NUMBER: 99887766", {})
        assert f["id_number"] == "99887766"
        assert f["name"] == ""
        assert f["date_of_birth"] == ""
        assert f["gender"] == ""

    def test_high_confidence_english_name_vs_low_confidence_noisy_token(self):
        """Test that English-first candidate ranking prefers high-confidence grouped name over low-confidence noise."""
        extractor = NationalIDExtractor(min_confidence=0.50)
        from app.models.schemas import OCRRegion
        regions = [
            # Header
            OCRRegion(text="GOVERNMENT OF INDIA", confidence=0.98, bbox=[50, 20, 450, 50]),
            OCRRegion(text="UNIQUE IDENTIFICATION AUTHORITY OF INDIA", confidence=0.96, bbox=[50, 55, 480, 80]),
            # Low-confidence noise artifact positioned higher on card
            OCRRegion(text="wher", confidence=0.10, bbox=[20, 85, 60, 95]),
            # High-confidence personal name split into two word tokens on the same horizontal line
            OCRRegion(text="Sriram", confidence=0.78, bbox=[50, 100, 150, 130]),
            OCRRegion(text="Mamundi", confidence=0.92, bbox=[160, 100, 280, 130]),
            # DOB & Gender
            OCRRegion(text="DOB: 15/08/1990", confidence=0.95, bbox=[50, 140, 250, 165]),
            OCRRegion(text="Male", confidence=0.97, bbox=[50, 175, 120, 200]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 220, 320, 255]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        fields, confs = extractor.extract_fields(ocr_text, mrz_fields={}, mrz_format=None, ocr_regions=regions)
        assert fields["name"] == "Sriram Mamundi"
        assert confs["name"] >= 0.75
        assert fields["name"] != "wher"

    def test_government_headers_rejected_as_names(self):
        """Ensure institutional and government headers are rejected and never extracted as personal names."""
        extractor = NationalIDExtractor(min_confidence=0.50)
        from app.models.schemas import OCRRegion
        regions = [
            OCRRegion(text="REPUBLIC OF INDIA", confidence=0.99, bbox=[50, 10, 300, 35]),
            OCRRegion(text="ELECTION COMMISSION OF INDIA", confidence=0.98, bbox=[50, 40, 400, 65]),
            OCRRegion(text="VOTER IDENTITY CARD", confidence=0.97, bbox=[50, 70, 350, 95]),
            OCRRegion(text="NATIONAL IDENTITY CARD", confidence=0.98, bbox=[50, 100, 320, 125]),
            OCRRegion(text="Card No: 12345678", confidence=0.95, bbox=[50, 140, 250, 165]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        fields, confs = extractor.extract_fields(ocr_text, mrz_fields={}, mrz_format=None, ocr_regions=regions)
        assert fields["name"] == ""
        assert fields["id_number"] == "12345678"

    def test_dob_extraction_variations(self):
        """Test DOB extraction across formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, and grouped tokens."""
        extractor = NationalIDExtractor()
        from app.models.schemas import OCRRegion
        
        # Test 1: Separated DOB label and date value tokens
        regions = [
            OCRRegion(text="DOB", confidence=0.95, bbox=[50, 100, 90, 125]),
            OCRRegion(text="15/08/1990", confidence=0.96, bbox=[100, 100, 220, 125]),
        ]
        fields, confs = extractor.extract_fields("DOB 15/08/1990", {}, ocr_regions=regions)
        assert fields["date_of_birth"] == "15/08/1990"
        
        # Test 2: DD-MM-YYYY
        f2, _ = extractor.extract_fields("Date of Birth: 25-12-1985\nMale", {})
        assert f2["date_of_birth"] == "25-12-1985"

        # Test 3: YYYY-MM-DD
        f3, _ = extractor.extract_fields("Birth Date: 1992-04-18\nFemale", {})
        assert f3["date_of_birth"] == "1992-04-18"

    def test_gender_extraction_variations(self):
        """Test gender extraction across English keywords: MALE, FEMALE, M, F, TRANSGENDER."""
        extractor = NationalIDExtractor()
        
        f1, _ = extractor.extract_fields("Name: John Doe\nGender: Male", {})
        assert f1["gender"] == "MALE"
        assert f1["sex"] == "M"

        f2, _ = extractor.extract_fields("Name: Jane Doe\nSex: Female", {})
        assert f2["gender"] == "FEMALE"
        assert f2["sex"] == "F"

        f3, _ = extractor.extract_fields("Name: Alex Doe\nSex: Transgender", {})
        assert f3["gender"] == "TRANSGENDER"
        assert f3["sex"] == "X"

    def test_dob_beside_label_and_reject_id_number_overlap(self):
        """Test valid DOB extracted beside label, and ID number fragment (8416) rejected as DOB."""
        extractor = NationalIDExtractor()
        from app.models.schemas import OCRRegion
        
        # Real-world sample with separate DOB token and ID number
        regions = [
            OCRRegion(text="DOB", confidence=0.95, bbox=[50, 140, 90, 165]),
            OCRRegion(text="15/08/1990", confidence=0.96, bbox=[100, 140, 220, 165]),
            OCRRegion(text="Male", confidence=0.97, bbox=[50, 175, 120, 200]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 220, 320, 255]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        fields, confs = extractor.extract_fields(ocr_text, {}, ocr_regions=regions)
        assert fields["date_of_birth"] == "15/08/1990"
        assert fields["id_number"] == "8416 1590 3267"
        assert fields["date_of_birth"] != "8416"

    def test_year_only_yob_vs_arbitrary_4_digit_rejection(self):
        """Test that explicit YOB (1990) is accepted, but arbitrary 4-digit token near DOB (8416) is rejected."""
        extractor = NationalIDExtractor()
        from app.models.schemas import OCRRegion

        # Case 1: Explicit Year of Birth label with valid year
        f1, _ = extractor.extract_fields("Year of Birth: 1990\nMale", {})
        assert f1["date_of_birth"] == "1990"

        # Case 2: Explicit YOB label with valid year
        f2, _ = extractor.extract_fields("YOB: 1984\nFemale", {})
        assert f2["date_of_birth"] == "1984"

        # Case 3: Arbitrary 4-digit number (8416) near DOB label without year-of-birth label -> REJECTED
        regions_bad = [
            OCRRegion(text="DOB", confidence=0.95, bbox=[50, 140, 90, 165]),
            OCRRegion(text="8416", confidence=0.95, bbox=[100, 140, 150, 165]),
            OCRRegion(text="1590 3267", confidence=0.99, bbox=[160, 140, 280, 165]),
        ]
        f3, _ = extractor.extract_fields("DOB 8416 1590 3267", {}, ocr_regions=regions_bad)
        assert f3["date_of_birth"] == ""
        assert f3["id_number"] == "8416 1590 3267"

    def test_malformed_dates_rejected(self):
        """Ensure invalid calendar days (>31) or invalid months (>12) are rejected."""
        extractor = NationalIDExtractor()
        # Invalid day 35
        f1, _ = extractor.extract_fields("DOB: 35/10/1990\nMale", {})
        assert f1["date_of_birth"] == ""

        # Invalid month 15
        f2, _ = extractor.extract_fields("DOB: 12/15/1990\nFemale", {})
        assert f2["date_of_birth"] == ""

    def test_unrelated_single_letter_ocr_noise_not_gender(self):
        """Ensure isolated single letters (M or F) elsewhere in document are NOT treated as gender."""
        extractor = NationalIDExtractor()
        from app.models.schemas import OCRRegion
        
        # Document with middle initial M and random OCR artifact F at top, but NO gender label
        regions = [
            OCRRegion(text="F", confidence=0.70, bbox=[10, 10, 20, 20]), # Random noise top left
            OCRRegion(text="Robert M Williams", confidence=0.95, bbox=[50, 60, 250, 85]),
            OCRRegion(text="DOB: 12/04/1985", confidence=0.96, bbox=[50, 95, 200, 120]),
            OCRRegion(text="ID: 98765432", confidence=0.95, bbox=[50, 130, 180, 155]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        fields, confs = extractor.extract_fields(ocr_text, {}, ocr_regions=regions)
        assert fields["gender"] == ""
        assert fields["sex"] == ""

    def test_gender_labeled_and_slash_patterns(self):
        """Test gender extraction from labeled single letters and multilingual slash formats."""
        extractor = NationalIDExtractor()
        
        # Labeled M
        f1, _ = extractor.extract_fields("Gender: M\nDOB: 15/08/1990", {})
        assert f1["gender"] == "M"
        assert f1["sex"] == "M"

        # Labeled F
        f2, _ = extractor.extract_fields("Sex: F\nDOB: 15/08/1990", {})
        assert f2["gender"] == "F"
        assert f2["sex"] == "F"

        # Multilingual slash format: पुरुष / MALE
        f3, _ = extractor.extract_fields("पुरुष / MALE\nDOB: 15/08/1990", {})
        assert f3["gender"] == "MALE"
        assert f3["sex"] == "M"



class TestDocumentServiceOrchestration:
    """Tests for full process_extraction pipeline including warnings and field debug."""

    def test_process_extraction_national_id_without_mrz_generates_warning(self, doc_service):
        mrz_res = MRZResult(detected=False, valid_format=False, overall_valid=False)
        ocr_text = "IDENTITY CARD\nName: ALICE WONDER\nID No: ID-998877"
        
        eff_type, fields, confs, warnings, field_debug, field_sources = doc_service.process_extraction(
            requested_type="national_id",
            ocr_text=ocr_text,
            mrz_result=mrz_res,
            mrz_fields={}
        )
        assert eff_type == "national_id"
        assert fields["name"] == "ALICE WONDER"
        assert fields["id_number"] == "ID-998877"
        assert len(warnings) > 0
        assert "MRZ not detected" in warnings[0]
        assert "name" in field_sources
        assert field_sources["name"]["source"] == "visual_ocr"


class TestVisaDualSourceExtraction:
    """Tests for Dual-Source Visa extraction (Visual Zone OCR + MRV MRZ OCR)."""

    def test_visa_visual_zone_only_extraction(self):
        """Test extraction of all visual fields when no MRV is detected."""
        extractor = VisaExtractor()
        ocr_text = (
            "UNITED STATES OF AMERICA\n"
            "VISA\n"
            "Control Number: 202315904812\n"
            "Visa Type / Class: B1/B2\n"
            "Entries: MULTIPLE\n"
            "Issue Date: 15/01/2023\n"
            "Expiry Date: 14/01/2033\n"
            "Passport No: P98765432\n"
            "Name: ROBERT WILLIAMS\n"
            "Authority: US EMBASSY LONDON"
        )
        fields, confs = extractor.extract_fields(ocr_text=ocr_text, mrz_fields={})
        assert fields["visa_number"] == "202315904812"
        assert fields["visa_type"] == "B1/B2"
        assert fields["entries"] == "MULTIPLE"
        assert fields["issue_date"] == "15/01/2023"
        assert fields["expiry_date"] == "14/01/2033"
        assert fields["passport_number"] == "P98765432"
        assert fields["name"] == "ROBERT WILLIAMS"
        assert fields["issuing_authority"] == "US EMBASSY LONDON"

        # Check field_sources
        sources = extractor.last_field_sources
        assert sources["visa_type"]["source"] == "visual_ocr"
        assert sources["visa_type"]["value"] == "B1/B2"
        assert sources["entries"]["source"] == "visual_ocr"
        assert sources["entries"]["value"] == "MULTIPLE"

    def test_visa_controlled_entries_normalization(self):
        """Test normalization of controlled entries values: MULTIPLE, SINGLE, DOUBLE, M, S, 1, 2."""
        extractor = VisaExtractor()
        
        # MULT
        f1, _ = extractor.extract_fields("Entries: MULT\nVisa Type: B1", {})
        assert f1["entries"] == "MULTIPLE"

        # M
        f2, _ = extractor.extract_fields("Entries: M\nVisa Type: B1", {})
        assert f2["entries"] == "MULTIPLE"

        # SINGLE
        f3, _ = extractor.extract_fields("Entries: SINGLE\nVisa Type: B1", {})
        assert f3["entries"] == "SINGLE"

        # S
        f4, _ = extractor.extract_fields("Entries: S\nVisa Type: B1", {})
        assert f4["entries"] == "SINGLE"

        # DOUBLE
        f5, _ = extractor.extract_fields("Entries: DOUBLE\nVisa Type: B1", {})
        assert f5["entries"] == "DOUBLE"

        # 1
        f6, _ = extractor.extract_fields("Entries: 1\nVisa Type: B1", {})
        assert f6["entries"] == "SINGLE"

    def test_visa_date_validation_and_rejection(self):
        """Test valid calendar dates are accepted and invalid ones are rejected."""
        extractor = VisaExtractor()
        
        # Valid date formats
        f1, _ = extractor.extract_fields("Issue Date: 12-05-2021\nExpiry Date: 11-05-2031", {})
        assert f1["issue_date"] == "12-05-2021"
        assert f1["expiry_date"] == "11-05-2031"

        # Invalid month (month 15)
        f2, _ = extractor.extract_fields("Issue Date: 12/15/2021\nExpiry Date: 35/05/2031", {})
        assert f2["issue_date"] == ""
        assert f2["expiry_date"] == ""

    def test_visa_dual_source_fusion_with_mrva(self):
        """Test fusion where identity fields come from MRV-A and printed fields come from visual zone."""
        extractor = VisaExtractor()
        mrz_fields = {
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "document_number": "L898902C3",
            "nationality": "UTO",
            "date_of_birth": "740812",
            "sex": "F",
            "date_of_expiry": "120415",
            "issuing_state": "UTO"
        }
        ocr_text = (
            "UNITED STATES VISA\n"
            "Visa Type / Class: B1/B2\n"
            "Entries: MULTIPLE\n"
            "Issue Date: 16/04/2002\n"
            "Expiry Date: 15/04/2012\n"
            "Passport No: P12345678\n"
            "Authority: LONDON"
        )
        fields, confs = extractor.extract_fields(ocr_text=ocr_text, mrz_fields=mrz_fields, mrz_format="MRVA")
        
        # MRZ-preferred identity fields
        assert fields["visa_number"] == "L898902C3"
        assert fields["name"] == "ERIKSSON ANNA MARIA"
        assert fields["nationality"] == "UTO"
        assert fields["date_of_birth"] == "740812"
        assert fields["sex"] == "F"
        assert fields["issuing_state"] == "UTO"
        assert fields["mrz_format"] == "MRVA"

        # Visual-preferred printed fields
        assert fields["visa_type"] == "B1/B2"
        assert fields["entries"] == "MULTIPLE"
        assert fields["issue_date"] == "16/04/2002"
        assert fields["expiry_date"] == "15/04/2012"
        assert fields["passport_number"] == "P12345678"
        assert fields["issuing_authority"] == "LONDON"

        # Verify sources
        sources = extractor.last_field_sources
        assert sources["visa_number"]["source"] == "mrz"
        assert sources["name"]["source"] == "mrz"
        assert sources["visa_type"]["source"] == "visual_ocr"
        assert sources["issue_date"]["source"] == "visual_ocr"

    def test_visa_discrepancy_warning_on_mismatched_data(self):
        """Test warning generation when visual OCR document number disagrees with MRV."""
        extractor = VisaExtractor()
        mrz_fields = {
            "surname": "JOHNSON",
            "given_names": "ROBERT",
            "document_number": "L898902C3",
            "nationality": "USA",
            "date_of_birth": "800101",
            "sex": "M",
            "date_of_expiry": "300101",
            "issuing_state": "USA"
        }
        ocr_text = (
            "VISA\n"
            "Control Number: 999999999\n"  # Completely different from L898902C3
            "Name: EMILY BLUNT\n"          # Completely different from ROBERT JOHNSON
            "Visa Type: B1/B2"
        )
        fields, confs = extractor.extract_fields(ocr_text=ocr_text, mrz_fields=mrz_fields, mrz_format="MRVA")
        assert len(extractor.last_warnings) >= 2
        assert any("Control Number" in w or "document_number" in w or "999999999" in w for w in extractor.last_warnings)
        assert any("name" in w.lower() or "EMILY BLUNT" in w for w in extractor.last_warnings)

    def test_visa_mrz_name_cleaning_strips_filler_noise(self):
        """Test that OCR filler confusion noise (SRSSSESSESSSS, K, E) is removed from parsed MRZ names."""
        from app.utils.mrz_utils import parse_mrva_mrz
        
        # Line 1 with OCR filler confusion artifacts at the end of given names
        l1_noisy = "V<UTOERIKSSON<<ANNA<MARIA<SRSSSESSESSSS<<<<<"
        l2_valid = "L898902C36UTO7408122F1204159<<<<<<<<<<<<<<<<"
        
        parsed = parse_mrva_mrz(l1_noisy, l2_valid)
        assert parsed["valid_format"] is True
        assert parsed["fields"]["surname"] == "ERIKSSON"
        assert parsed["fields"]["given_names"] == "ANNA MARIA"
        assert "SRSSSESSESSSS" not in parsed["fields"]["given_names"]
        # Raw MRZ line is preserved untouched
        assert parsed["raw_line1"] == l1_noisy



class TestSecondPassFieldOCR:
    """Targeted field-specific OCR crop, preprocessing, and candidate scoring tests."""

    def test_second_pass_clearly_readable_dob_and_gender(self):
        """Test targeted field OCR correctly extracts DOB and Gender from dynamic image crops."""
        import numpy as np
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        
        # Synthetic document image
        img = np.full((300, 500, 3), 240, dtype=np.uint8)
        
        # Mock OCR service returning valid DOB and Gender for field crops
        mock_engine = MockOCREngine()
        mock_engine.extract_field_text = lambda image, field_type, psm=7: (
            OCRResult(raw_text="15/08/1990", regions=[OCRRegion(text="15/08/1990", confidence=0.96, bbox=[0, 0, 100, 30])], average_confidence=0.96)
            if field_type == "dob"
            else OCRResult(raw_text="FEMALE", regions=[OCRRegion(text="FEMALE", confidence=0.95, bbox=[0, 0, 80, 30])], average_confidence=0.95)
        )
        ocr_srv = OCRService(engine=mock_engine)

        extractor = NationalIDExtractor()
        regions = [
            OCRRegion(text="Sriram Mamundi", confidence=0.92, bbox=[50, 60, 250, 85]),
            OCRRegion(text="DOB", confidence=0.95, bbox=[50, 100, 90, 125]),
            OCRRegion(text="Gender", confidence=0.95, bbox=[50, 140, 120, 165]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 180, 320, 210]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        
        fields, confs = extractor.extract_fields(
            ocr_text=ocr_text,
            mrz_fields={},
            ocr_regions=regions,
            document_image=img,
            ocr_service=ocr_srv,
            include_debug=True
        )
        
        assert fields["date_of_birth"] == "15/08/1990"
        assert fields["gender"] == "FEMALE"
        assert fields["sex"] == "F"
        assert fields["name"] == "Sriram Mamundi"
        assert fields["id_number"] == "8416 1590 3267"
        assert len(extractor.last_field_debug["dob_candidates"]) > 0
        assert len(extractor.last_field_debug["gender_candidates"]) > 0

    def test_second_pass_dob_beside_id_never_extracts_id_digits(self):
        """Ensure second pass rejects any DOB crop that returns ID number digits (8416)."""
        import numpy as np
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        
        img = np.full((300, 500, 3), 240, dtype=np.uint8)
        
        mock_engine = MockOCREngine()
        # Simulate crop returning ID number digits
        mock_engine.extract_field_text = lambda image, field_type, psm=7: OCRResult(
            raw_text="8416",
            regions=[OCRRegion(text="8416", confidence=0.98, bbox=[0, 0, 60, 30])],
            average_confidence=0.98
        )
        ocr_srv = OCRService(engine=mock_engine)

        extractor = NationalIDExtractor()
        regions = [
            OCRRegion(text="DOB", confidence=0.95, bbox=[50, 100, 90, 125]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 180, 320, 210]),
        ]
        ocr_text = "\n".join(r.text for r in regions)
        
        fields, _ = extractor.extract_fields(
            ocr_text=ocr_text,
            mrz_fields={},
            ocr_regions=regions,
            document_image=img,
            ocr_service=ocr_srv
        )
        # 8416 must be rejected and date_of_birth must remain empty
        assert fields["date_of_birth"] == ""
        assert fields["id_number"] == "8416 1590 3267"

    def test_second_pass_male_and_female_recognition(self):
        """Test extraction of MALE and FEMALE tokens."""
        import numpy as np
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        
        img = np.full((300, 500, 3), 240, dtype=np.uint8)
        
        # Test MALE
        mock_engine_male = MockOCREngine()
        mock_engine_male.extract_field_text = lambda image, field_type, psm=7: OCRResult(
            raw_text="MALE",
            regions=[OCRRegion(text="MALE", confidence=0.95, bbox=[0, 0, 50, 25])],
            average_confidence=0.95
        )
        extractor = NationalIDExtractor()
        regions = [OCRRegion(text="Sex", confidence=0.95, bbox=[50, 100, 90, 125])]
        f_male, _ = extractor.extract_fields(
            ocr_text="Sex",
            mrz_fields={},
            ocr_regions=regions,
            document_image=img,
            ocr_service=OCRService(engine=mock_engine_male)
        )
        assert f_male["gender"] == "MALE"
        assert f_male["sex"] == "M"

    def test_second_pass_noisy_gender_region_rejected(self):
        """Ensure random noisy OCR text in crop (e.g. 'XXKJ9') is not accepted as gender."""
        import numpy as np
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRResult, OCRService
        
        img = np.full((300, 500, 3), 240, dtype=np.uint8)
        
        mock_engine_noise = MockOCREngine()
        mock_engine_noise.extract_field_text = lambda image, field_type, psm=7: OCRResult(
            raw_text="XXKJ9",
            regions=[OCRRegion(text="XXKJ9", confidence=0.80, bbox=[0, 0, 60, 25])],
            average_confidence=0.80
        )
        extractor = NationalIDExtractor()
        regions = [OCRRegion(text="Gender", confidence=0.95, bbox=[50, 100, 110, 125])]
        f_noise, _ = extractor.extract_fields(
            ocr_text="Gender",
            mrz_fields={},
            ocr_regions=regions,
            document_image=img,
            ocr_service=OCRService(engine=mock_engine_noise)
        )
        assert f_noise["gender"] == ""
        assert f_noise["sex"] == ""

    def test_second_pass_missing_values_return_empty_strings(self):
        """Ensure documents without DOB or Gender labels return empty strings without hallucination."""
        import numpy as np
        from app.models.schemas import OCRRegion
        from app.services.ocr_service import MockOCREngine, OCRService
        
        img = np.full((300, 500, 3), 240, dtype=np.uint8)
        extractor = NationalIDExtractor()
        regions = [
            OCRRegion(text="GOVERNMENT OF INDIA", confidence=0.98, bbox=[50, 20, 450, 50]),
            OCRRegion(text="8416 1590 3267", confidence=0.99, bbox=[50, 180, 320, 210]),
        ]
        fields, _ = extractor.extract_fields(
            ocr_text="\n".join(r.text for r in regions),
            mrz_fields={},
            ocr_regions=regions,
            document_image=img,
            ocr_service=OCRService(engine=MockOCREngine())
        )
        assert fields["date_of_birth"] == ""
        assert fields["gender"] == ""
        assert fields["id_number"] == "8416 1590 3267"



