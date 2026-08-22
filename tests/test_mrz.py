"""Unit tests for ICAO 9303 MRZ parsing and check digit validation."""

import pytest
from app.services.mrz_service import MRZService
from app.utils.mrz_utils import (
    calculate_check_digit,
    char_to_mrz_value,
    normalize_mrz_line,
    normalize_ocr_alpha,
    normalize_ocr_digits,
    parse_td3_mrz,
    validate_check_digit,
)


class TestICAOCheckDigit:
    """Tests for the ICAO Doc 9303 Check Digit Algorithm (Weights 7, 3, 1)."""

    def test_char_to_mrz_value(self):
        # Digits 0-9
        assert char_to_mrz_value('0') == 0
        assert char_to_mrz_value('9') == 9
        # Letters A-Z
        assert char_to_mrz_value('A') == 10
        assert char_to_mrz_value('B') == 11
        assert char_to_mrz_value('Z') == 35
        # Filler '<'
        assert char_to_mrz_value('<') == 0

    def test_calculate_check_digit_simple(self):
        # Passport number: L898902C3 -> 6
        # L(21)*7 + 8*3 + 9*1 + 8*7 + 9*3 + 0*1 + 2*7 + C(12)*3 + 3*1 = 316 -> 6
        assert calculate_check_digit("L898902C3") == 6
        
        # DOB: 740812 -> 2
        # 7*7 + 4*3 + 0*1 + 8*7 + 1*3 + 2*1 = 49 + 12 + 0 + 56 + 3 + 2 = 122 -> 2
        assert calculate_check_digit("740812") == 2
        
        # Expiry: 120415 -> 9
        # 1*7 + 2*3 + 0*1 + 4*7 + 1*3 + 5*1 = 7 + 6 + 0 + 28 + 3 + 5 = 49 -> 9
        assert calculate_check_digit("120415") == 9

    def test_validate_check_digit(self):
        assert validate_check_digit("L898902C3", "6") is True
        assert validate_check_digit("L898902C3", 6) is True
        assert validate_check_digit("740812", "2") is True
        assert validate_check_digit("120415", "9") is True
        
        # Intentionally invalid check digits
        assert validate_check_digit("L898902C3", "5") is False
        assert validate_check_digit("740812", "7") is False
        assert validate_check_digit("120415", "0") is False

    def test_validate_check_digit_edge_cases(self):
        # Empty / non-numeric expected digit
        assert validate_check_digit("12345", "") is False
        assert validate_check_digit("12345", None) is False
        assert validate_check_digit("12345", "X") is False
        # Filler handling
        assert validate_check_digit("<<<<<<<<<", "<") is True


class TestMRZLineNormalization:
    """Tests for raw OCR text cleaning and normalization."""

    def test_normalize_mrz_line_artifacts(self):
        raw = "P<UTO ERIKSSON««ANNA(MARIA[<<<<<<<<<<<<<<<<<"
        normalized = normalize_mrz_line(raw)
        assert " " not in normalized
        assert "«" not in normalized
        assert "(" not in normalized
        assert "[" not in normalized
        assert normalized.startswith("P<UTO<ERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<")

    def test_normalize_ocr_digits(self):
        raw_date = "74O8l2"  # 'O' instead of 0, 'l' instead of 1
        normalized = normalize_ocr_digits(raw_date)
        assert normalized == "740812"

    def test_normalize_ocr_alpha(self):
        raw_country = "UT0"  # '0' instead of O
        normalized = normalize_ocr_alpha(raw_country)
        assert normalized == "UTO"


class TestTD3MRZParsing:
    """Tests for parsing 2-line 44-character TD3 Passport MRZ."""

    def test_valid_td3_sample_parsing(self, valid_td3_mrz_sample):
        l1 = valid_td3_mrz_sample["line1"]
        l2 = valid_td3_mrz_sample["line2"]
        
        res = parse_td3_mrz(l1, l2)
        assert res["valid_format"] is True
        assert res["overall_valid"] is True
        assert res["fields"]["surname"] == valid_td3_mrz_sample["expected_surname"]
        assert res["fields"]["given_names"] == valid_td3_mrz_sample["expected_given_names"]
        assert res["fields"]["passport_number"] == valid_td3_mrz_sample["expected_passport_number"]
        assert res["fields"]["nationality"] == valid_td3_mrz_sample["expected_nationality"]
        assert res["fields"]["date_of_birth"] == valid_td3_mrz_sample["expected_dob"]
        assert res["fields"]["sex"] == valid_td3_mrz_sample["expected_sex"]
        assert res["fields"]["date_of_expiry"] == valid_td3_mrz_sample["expected_expiry"]
        assert res["fields"]["personal_number"] == valid_td3_mrz_sample["expected_personal_number"]
        
        # Check individual check digit flags
        cd = res["check_digits"]
        assert cd["passport_number"] is True
        assert cd["date_of_birth"] is True
        assert cd["date_of_expiry"] is True
        assert cd["personal_number"] is True
        assert cd["composite"] is True

    def test_invalid_line_length(self):
        l1 = "P<UTOERIKSSON<<ANNA"  # only 19 chars
        l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        res = parse_td3_mrz(l1, l2)
        assert res["valid_format"] is False
        assert res["overall_valid"] is False
        assert "Invalid TD3 line lengths" in res["error"]

    def test_tampered_passport_number_check_digit(self, valid_td3_mrz_sample):
        l1 = valid_td3_mrz_sample["line1"]
        # Modify passport check digit from 6 to 9
        l2_tampered = "L898902C39UTO7408122F1204159ZE184226B<<<<<10"
        
        res = parse_td3_mrz(l1, l2_tampered)
        assert res["valid_format"] is True
        assert res["check_digits"]["passport_number"] is False
        assert res["overall_valid"] is False

    def test_tampered_date_of_birth_check_digit(self, valid_td3_mrz_sample):
        l1 = valid_td3_mrz_sample["line1"]
        # Modify DOB check digit from 2 to 7
        l2_tampered = "L898902C36UTO7408127F1204159ZE184226B<<<<<10"
        
        res = parse_td3_mrz(l1, l2_tampered)
        assert res["valid_format"] is True
        assert res["check_digits"]["date_of_birth"] is False
        assert res["overall_valid"] is False

    def test_tampered_expiry_date(self, valid_td3_mrz_sample):
        l1 = valid_td3_mrz_sample["line1"]
        # Change expiry date from 120415 to 190415 while keeping old check digit 9
        l2_tampered = "L898902C36UTO7408122F1904159ZE184226B<<<<<10"
        
        res = parse_td3_mrz(l1, l2_tampered)
        assert res["valid_format"] is True
        assert res["check_digits"]["date_of_expiry"] is False
        assert res["overall_valid"] is False


class TestMRZService:
    """Tests for the high-level MRZService."""

    def test_extract_and_validate_mrz_found(self, valid_td3_mrz_sample):
        raw_ocr = (
            "PASSPORT\n"
            f"{valid_td3_mrz_sample['line1']}\n"
            f"{valid_td3_mrz_sample['line2']}\n"
        )
        mrz_result, fields = MRZService.extract_and_validate_mrz(raw_ocr)
        assert mrz_result.detected is True
        assert mrz_result.valid_format is True
        assert mrz_result.overall_valid is True
        assert mrz_result.check_digits.passport_number is True
        assert fields["surname"] == "ERIKSSON"

    def test_extract_and_validate_mrz_missing(self):
        raw_ocr = "NATIONAL IDENTITY CARD\nNAME: JOHN DOE\nID: 123456789\nDOB: 1990-01-01"
        mrz_result, fields = MRZService.extract_and_validate_mrz(raw_ocr)
        assert mrz_result.detected is False
        assert mrz_result.valid_format is False
        assert mrz_result.overall_valid is False
        assert mrz_result.check_digits is None
        assert fields == {}
