"""Unit tests for ICAO 9303 MRZ parsing and check digit validation."""

import pytest
from app.services.mrz_service import MRZService
from app.utils.mrz_utils import (
    calculate_check_digit,
    char_to_mrz_value,
    normalize_mrz_line,
    normalize_ocr_alpha,
    normalize_ocr_digits,
    parse_mrva_mrz,
    parse_mrvb_mrz,
    parse_td1_mrz,
    parse_td2_mrz,
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
        mrz_result, fields, _ = MRZService.extract_and_validate_mrz(raw_ocr)
        assert mrz_result.detected is True
        assert mrz_result.valid_format is True
        assert mrz_result.overall_valid is True
        assert mrz_result.check_digits.passport_number is True
        assert fields["surname"] == "ERIKSSON"

    def test_extract_and_validate_mrz_missing(self):
        raw_ocr = "NATIONAL IDENTITY CARD\nNAME: JOHN DOE\nID: 123456789\nDOB: 1990-01-01"
        mrz_result, fields, _ = MRZService.extract_and_validate_mrz(raw_ocr)
        assert mrz_result.detected is False
        assert mrz_result.valid_format is False
        assert mrz_result.overall_valid is False
        assert mrz_result.check_digits is None
        assert fields == {}

    def test_clear_td3_mrz(self, valid_td3_mrz_sample):
        """Test with crystal clear 2-line TD3 MRZ input."""
        l1 = valid_td3_mrz_sample["line1"]
        l2 = valid_td3_mrz_sample["line2"]
        raw_text = f"PASSPORT REPUBLIC\n{l1}\n{l2}"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=raw_text)
        assert mrz_res.detected is True
        assert mrz_res.valid_format is True
        assert mrz_res.overall_valid is True
        assert fields["passport_number"] == valid_td3_mrz_sample["expected_passport_number"]
        assert fields["surname"] == valid_td3_mrz_sample["expected_surname"]

    def test_low_contrast_td3_mrz(self, valid_td3_mrz_sample):
        """Test where general OCR missed the MRZ due to low contrast, but dedicated MRZ pass captured it."""
        general_ocr = "PASSPORT\nNAME ERIKSSON ANNA MARIA\n(Unreadable bottom line)"
        mrz_pass_text = f"{valid_td3_mrz_sample['line1']}\n{valid_td3_mrz_sample['line2']}"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=general_ocr,
            mrz_candidate_texts=[mrz_pass_text]
        )
        assert mrz_res.detected is True
        assert mrz_res.valid_format is True
        assert mrz_res.overall_valid is True
        assert fields["passport_number"] == "L898902C3"

    def test_slightly_blurred_mrz_with_whitespace(self, valid_td3_mrz_sample):
        """Test where OCR produced whitespace and minor symbol artifacts in MRZ lines."""
        # Simulated OCR output with spaces between filler characters
        l1_noisy = "P<UTO  ERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<< "
        l2_noisy = "L898902C36 UTO 7408122 F 1204159 ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text="PASSPORT",
            mrz_candidate_texts=[f"{l1_noisy}\n{l2_noisy}"]
        )
        assert mrz_res.detected is True
        assert mrz_res.valid_format is True
        assert mrz_res.overall_valid is True
        assert fields["surname"] == "ERIKSSON"

    def test_malformed_mrz_rejection(self):
        """Test that malformed/garbled lines are rejected safely without crashing."""
        malformed_ocr = (
            "PASSPORT\n"
            "P<UTOERIKSSON<<GARBLED_TOO_SHORT\n"
            "L898902C36UTO7408122F1204159ZE"
        )
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=malformed_ocr)
        assert mrz_res.detected is False
        assert mrz_res.valid_format is False
        assert mrz_res.overall_valid is False
        assert fields == {}

    def test_no_mrz_document(self):
        """Test document without any MRZ returns clean empty results."""
        text = "DRIVER LICENSE\nSTATE OF CALIFORNIA\nNAME: JOHN SMITH\nEXPIRES: 12/31/2028"
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=text)
        assert mrz_res.detected is False
        assert mrz_res.valid_format is False
        assert mrz_res.overall_valid is False
        assert fields == {}

    def test_noisy_ocr_resembling_passport_data(self):
        """Test random OCR noise with alphanumeric tokens and dates that resemble passport data."""
        noisy_text = (
            "UNITED STATES PASSPORT OFFICE\n"
            "RANDOM SERIAL A8923489012\n"
            "DATE 2024-05-12 TIME 14:30\n"
            "CODE X89B721A09 NOT AN MRZ LINE AT ALL\n"
            "NOISE 12903481230498120394"
        )
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=noisy_text)
        assert mrz_res.detected is False
        assert mrz_res.valid_format is False
        assert mrz_res.overall_valid is False
        assert fields == {}

    def test_candidate_scoring_and_selection(self, valid_td3_mrz_sample):
        """Test selecting the highest-scoring candidate among multiple plausible/tampered sources."""
        tampered_candidate_1 = (
            "otsu_r20_psm6",
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C39UTO7408122F1204159ZE184226B<<<<<10"
        )
        noisy_padding_candidate_2 = (
            "adaptive_r25_psm6",
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<        \nL898902C36UTO7408122F1204159ZE184226B<<<<<10"
        )
        clean_candidate_3 = (
            "clahe_r35_psm6",
            f"{valid_td3_mrz_sample['line1']}\n{valid_td3_mrz_sample['line2']}"
        )
        
        mrz_res, fields, debug = MRZService.extract_and_validate_mrz(
            raw_ocr_text="PASSPORT",
            mrz_candidate_texts=[tampered_candidate_1, noisy_padding_candidate_2, clean_candidate_3],
            include_debug=True
        )
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is True
        assert fields["passport_number"] == valid_td3_mrz_sample["expected_passport_number"]
        assert debug is not None
        assert debug["best_candidate"]["source"] == "clahe_r35_psm6"
        assert debug["candidate_count"] == 3
        assert len(debug["candidate_scores"]) == 3


class TestDeterministicMRZCorrection:
    """Tests for position-aware, check-digit verified deterministic MRZ corrections."""

    def test_o_vs_0_confusion_in_numeric_fields(self, valid_td3_mrz_sample):
        """Test 'O' instead of '0' in DOB field is corrected when verified by check digit."""
        l1 = valid_td3_mrz_sample["line1"]
        # In Line 2: change DOB 740812 to 74O812 (contains letter O)
        l2_raw = "L898902C36UTO74O8122F1204159ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is True
        assert fields["date_of_birth"] == "740812"
        assert mrz_res.line2 == "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        assert mrz_res.raw_line2 == l2_raw
        
        # Verify correction was logged
        dob_corrections = [c for c in mrz_res.corrections if c.field == "date_of_birth"]
        assert len(dob_corrections) >= 1
        assert dob_corrections[0].from_char == "O"
        assert dob_corrections[0].to_char == "0"
        assert dob_corrections[0].position == 15

    def test_i_vs_1_confusion_in_expiry_date(self, valid_td3_mrz_sample):
        """Test 'I' instead of '1' in Expiry field is corrected when verified by check digit."""
        l1 = valid_td3_mrz_sample["line1"]
        # In Line 2: change Expiry 120415 to I204I5
        l2_raw = "L898902C36UTO7408122FI204I59ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is True
        assert fields["date_of_expiry"] == "120415"
        
        exp_corrections = [c for c in mrz_res.corrections if c.field == "date_of_expiry"]
        assert len(exp_corrections) >= 1
        assert exp_corrections[0].to_char == "1"

    def test_j_vs_9_confusion_in_composite_check_digit(self, valid_td3_mrz_sample):
        """Test 'O' or 'I' or confusions in check digit positions."""
        l1 = valid_td3_mrz_sample["line1"]
        # In Line 2: composite check digit at index 43 is 'O' instead of '0'
        l2_raw = "L898902C36UTO7408122F1204159ZE184226B<<<<<1O"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is True
        assert mrz_res.line2[43] == "0"
        
        cd_corrections = [c for c in mrz_res.corrections if c.field == "composite_check_digit"]
        assert len(cd_corrections) == 1
        assert cd_corrections[0].from_char == "O"
        assert cd_corrections[0].to_char == "0"

    def test_trailing_filler_confusion_in_line1(self, valid_td3_mrz_sample):
        """Test trailing filler characters 'K' or 'E' in Line 1 are normalized to '<'."""
        # Line 1 with trailing K and E instead of <
        l1_raw = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<KK<EE"
        l2 = valid_td3_mrz_sample["line2"]
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1_raw}\n{l2}"
        )
        assert mrz_res.detected is True
        assert fields["surname"] == "ERIKSSON"
        assert fields["given_names"] == "ANNA MARIA"
        assert mrz_res.line1.endswith("<<<<")
        assert mrz_res.raw_line1 == l1_raw
        
        filler_corrections = [c for c in mrz_res.corrections if c.field == "filler"]
        assert len(filler_corrections) >= 2

    def test_valid_ind_nationality_recovery(self, valid_td3_mrz_sample):
        """Test '1ND' in nationality field is safely corrected to valid ICAO country code 'IND'."""
        l1 = valid_td3_mrz_sample["line1"]
        # In Line 2: replace UTO with 1ND
        l2_raw = "L898902C361ND7408122F1204159ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        assert fields["nationality"] == "IND"
        assert mrz_res.raw_line2 == l2_raw
        
        nat_corrections = [c for c in mrz_res.corrections if c.field == "nationality"]
        assert len(nat_corrections) == 1
        assert nat_corrections[0].from_char == "1"
        assert nat_corrections[0].to_char == "I"
        assert "IND" in nat_corrections[0].reason

    def test_invalid_ioo_nationality_rejected(self, valid_td3_mrz_sample):
        """Test '100' in nationality field is NOT blindly transformed to 'IOO' because IOO is invalid."""
        l1 = valid_td3_mrz_sample["line1"]
        # In Line 2: nationality is '100'
        l2_raw = "L898902C361007408122F1204159ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        # Must NOT be 'IOO'
        assert fields["nationality"] != "IOO"
        assert fields["nationality"] == "100"
        # No false nationality correction logged
        assert not any(c.field == "nationality" and c.to_char == "O" for c in mrz_res.corrections)

    def test_ambiguous_o_0_stays_unchanged_without_check_digit(self, valid_td3_mrz_sample):
        """Test ambiguous O/0 characters are preserved when no check digit confirms the replacement."""
        l1 = valid_td3_mrz_sample["line1"]
        # In Line 2: corrupted DOB with invalid check digit that cannot be resolved by single substitution
        l2_raw = "L898902C36UTO8888888F1204159ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is False
        assert fields["date_of_birth"] == "888888"

    def test_ambiguous_i_1_stays_unchanged(self, valid_td3_mrz_sample):
        """Test '1QQ' in nationality is not transformed to 'IQQ' because IQQ is not a valid country code."""
        l1 = valid_td3_mrz_sample["line1"]
        l2_raw = "L898902C361QQ7408122F1204159ZE184226B<<<<<10"
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1}\n{l2_raw}"
        )
        assert mrz_res.detected is True
        assert fields["nationality"] == "1QQ"
        assert not any(c.field == "nationality" for c in mrz_res.corrections)

    def test_name_characters_never_modified_as_fillers(self, valid_td3_mrz_sample):
        """Test letters 'K', 'E', 'C' within names are strictly preserved and never converted to '<'."""
        l1_raw = "P<UTOJACKSON<<ERIC<<<<<<<<<<<<<<<<<<<<<<<<<<"
        l2 = valid_td3_mrz_sample["line2"]
        
        mrz_res, fields, _ = MRZService.extract_and_validate_mrz(
            raw_ocr_text=f"{l1_raw}\n{l2}"
        )
        assert mrz_res.detected is True
        assert fields["surname"] == "JACKSON"
        assert "K" in fields["surname"]
        assert fields["given_names"] == "ERIC"
        assert "C" in fields["given_names"]
        assert "E" in fields["given_names"]
        # Ensure 'K', 'C', 'E' inside names were NOT altered to '<'
        assert not any(c.position < 25 and c.to_char == "<" for c in mrz_res.corrections)


class TestFieldLevelValidation:
    """Tests for explicit field-level validation and diagnostics."""

    def test_valid_ind_nationality_field_validation(self, valid_td3_mrz_sample):
        """Test valid IND nationality passes field-level validation."""
        l1 = valid_td3_mrz_sample["line1"]
        l2_raw = "L898902C36IND7408122F1204159ZE184226B<<<<<10"
        
        mrz_res, _, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=f"{l1}\n{l2_raw}")
        assert mrz_res.detected is True
        assert mrz_res.field_validation is not None
        nat_val = mrz_res.field_validation["nationality"]
        assert nat_val.valid is True
        assert nat_val.value == "IND"
        assert nat_val.reason is None

    def test_numeric_nationality_marked_invalid(self, valid_td3_mrz_sample):
        """Test numeric nationality '100' is marked invalid with clear reason and not converted to IND."""
        l1 = valid_td3_mrz_sample["line1"]
        l2_raw = "L898902C361007408122F1204159ZE184226B<<<<<10"
        
        mrz_res, _, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=f"{l1}\n{l2_raw}")
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is False
        assert mrz_res.field_validation is not None
        
        nat_val = mrz_res.field_validation["nationality"]
        assert nat_val.valid is False
        assert nat_val.value == "100"
        assert "Invalid nationality code" in nat_val.reason
        assert "100" in nat_val.reason

    def test_missing_sex_marked_invalid(self, valid_td3_mrz_sample):
        """Test missing sex character ('<') is flagged in field_validation."""
        l1 = valid_td3_mrz_sample["line1"]
        # Replace sex 'F' with '<' filler
        l2_raw = "L898902C36UTO7408122<1204159ZE184226B<<<<<10"
        
        mrz_res, _, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=f"{l1}\n{l2_raw}")
        assert mrz_res.detected is True
        assert mrz_res.field_validation is not None
        
        sex_val = mrz_res.field_validation["sex"]
        assert sex_val.valid is False
        assert "Missing or invalid MRZ sex value" in sex_val.reason

    def test_invalid_sex_character_marked_invalid(self, valid_td3_mrz_sample):
        """Test invalid sex character (e.g. '0' or '9' or 'Z') is flagged in field_validation."""
        l1 = valid_td3_mrz_sample["line1"]
        # Replace sex 'F' with invalid 'Z'
        l2_raw = "L898902C36UTO7408122Z1204159ZE184226B<<<<<10"
        
        mrz_res, _, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=f"{l1}\n{l2_raw}")
        assert mrz_res.detected is True
        assert mrz_res.field_validation is not None
        
        sex_val = mrz_res.field_validation["sex"]
        assert sex_val.valid is False
        assert "Missing or invalid MRZ sex value" in sex_val.reason

    def test_valid_passport_number_field_validation(self, valid_td3_mrz_sample):
        """Test valid passport number with matching check digit passes field validation."""
        l1 = valid_td3_mrz_sample["line1"]
        l2 = valid_td3_mrz_sample["line2"]
        
        mrz_res, _, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=f"{l1}\n{l2}")
        assert mrz_res.detected is True
        assert mrz_res.field_validation is not None
        
        pass_val = mrz_res.field_validation["passport_number"]
        assert pass_val.valid is True
        assert pass_val.value == valid_td3_mrz_sample["expected_passport_number"]
        assert pass_val.reason is None

    def test_failed_passport_number_check_digit_marked_invalid(self, valid_td3_mrz_sample):
        """Test tampered/failed passport number check digit is marked invalid with reason."""
        l1 = valid_td3_mrz_sample["line1"]
        # Tamper passport number check digit from 6 to 3
        l2_tampered = "L898902C33UTO7408122F1204159ZE184226B<<<<<10"
        
        mrz_res, _, _ = MRZService.extract_and_validate_mrz(raw_ocr_text=f"{l1}\n{l2_tampered}")
        assert mrz_res.detected is True
        assert mrz_res.overall_valid is False
        assert mrz_res.field_validation is not None
        
        pass_val = mrz_res.field_validation["passport_number"]
        assert pass_val.valid is False
        assert "check digit validation failed" in pass_val.reason.lower()


class TestVisaMRZParsing:
    """Tests for ICAO Doc 9303 MRV-A and MRV-B Visa MRZ parsing and validation."""

    def test_valid_mrva_parsing_and_check_digits(self):
        """Test parsing of valid 44-character MRV-A visa format."""
        l1 = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        l2 = "L898902C36UTO7408122F1204159<<<<<<<<<<<<<<<<"

        parsed = parse_mrva_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["format"] == "MRVA"
        assert parsed["overall_valid"] is True
        assert parsed["fields"]["surname"] == "ERIKSSON"
        assert parsed["fields"]["given_names"] == "ANNA MARIA"
        assert parsed["fields"]["document_number"] == "L898902C3"
        assert parsed["fields"]["nationality"] == "UTO"
        assert parsed["fields"]["date_of_birth"] == "740812"
        assert parsed["fields"]["sex"] == "F"
        assert parsed["fields"]["date_of_expiry"] == "120415"
        assert parsed["check_digits"]["document_number"] is True
        assert parsed["check_digits"]["date_of_birth"] is True
        assert parsed["check_digits"]["date_of_expiry"] is True
        assert parsed["field_validation"]["document_number"]["valid"] is True
        assert parsed["field_validation"]["nationality"]["valid"] is True
        assert parsed["field_validation"]["date_of_birth"]["valid"] is True
        assert parsed["field_validation"]["date_of_expiry"]["valid"] is True

    def test_valid_mrvb_parsing_and_check_digits(self):
        """Test parsing of valid 36-character MRV-B visa format."""
        l1 = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<"
        l2 = "L898902C36UTO7408122F1204159<<<<<<<<"

        parsed = parse_mrvb_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["format"] == "MRVB"
        assert parsed["overall_valid"] is True
        assert parsed["fields"]["surname"] == "ERIKSSON"
        assert parsed["fields"]["given_names"] == "ANNA MARIA"
        assert parsed["fields"]["document_number"] == "L898902C3"
        assert parsed["fields"]["nationality"] == "UTO"
        assert parsed["fields"]["date_of_birth"] == "740812"
        assert parsed["fields"]["sex"] == "F"
        assert parsed["fields"]["date_of_expiry"] == "120415"
        assert parsed["check_digits"]["document_number"] is True
        assert parsed["check_digits"]["date_of_birth"] is True
        assert parsed["check_digits"]["date_of_expiry"] is True

    def test_wrong_line_length_rejected(self):
        """Test rejection of MRV-A and MRV-B with incorrect line lengths."""
        # MRV-A wrong length (38 chars)
        l1_short = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<"
        l2_long = "L898902C36UTO7408122F1204159<<<<<<<<<<<<<<<<"
        res_a = parse_mrva_mrz(l1_short, l2_long)
        assert res_a["valid_format"] is False
        assert res_a["overall_valid"] is False
        assert "Invalid MRV-A line lengths" in res_a["error"]

        # MRV-B wrong length (40 chars)
        l1_mrvb_long = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<"
        l2_mrvb_good = "L898902C36UTO7408122F1204159<<<<<<<<"
        res_b = parse_mrvb_mrz(l1_mrvb_long, l2_mrvb_good)
        assert res_b["valid_format"] is False
        assert res_b["overall_valid"] is False
        assert "Invalid MRV-B line lengths" in res_b["error"]

    def test_invalid_nationality_lau_marked_invalid(self):
        """Test invalid nationality 'LAU' is marked invalid and not blindly forced into a fake code."""
        l1 = "V<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        l2 = "VJDEHE5CK1LAU7408122F1204159<<<<<<<<<<<<<<<<"  # Nationality is 'LAU'

        parsed = parse_mrva_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["fields"]["nationality"] == "LAU"
        assert parsed["field_validation"]["nationality"]["valid"] is False
        assert "Invalid nationality code: 'LAU'" in parsed["field_validation"]["nationality"]["reason"]
        assert parsed["overall_valid"] is False

    def test_dob_containing_ocr_letters_marked_invalid(self):
        """Test non-numeric date of birth such as 'S93101' is marked invalid without guessing."""
        l1 = "V<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        l2 = "VJDEHE5CK1USA S93101 4<1204159<<<<<<<<<<<<<<<<"  # DOB is 'S93101'

        parsed = parse_mrva_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["field_validation"]["date_of_birth"]["valid"] is False
        assert "contains non-numeric characters" in parsed["field_validation"]["date_of_birth"]["reason"]
        assert parsed["overall_valid"] is False

    def test_invalid_expiry_date_marked_invalid(self):
        """Test invalid expiry date values such as month 70 ('117062') are marked invalid."""
        l1 = "V<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        l2 = "VJDEHE5CK1USA7408122F1170626<<<<<<<<<<<<<<<<"  # Expiry is '117062' (month 70)

        parsed = parse_mrva_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["field_validation"]["date_of_expiry"]["valid"] is False
        assert "month '70' out of range" in parsed["field_validation"]["date_of_expiry"]["reason"]
        assert parsed["overall_valid"] is False

    def test_invalid_check_digit_marked_invalid(self):
        """Test detection of invalid/tampered check digits in MRV-A."""
        l1 = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        # Tampered visa number check digit 6 -> 0
        l2 = "L898902C30UTO7408122F1204159<<<<<<<<<<<<<<<<"
        parsed = parse_mrva_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["overall_valid"] is False
        assert parsed["check_digits"]["document_number"] is False
        assert parsed["field_validation"]["document_number"]["valid"] is False
        assert "check digit validation failed" in parsed["field_validation"]["document_number"]["reason"]

    def test_malformed_mrz_that_must_remain_invalid(self):
        """Test real-world noisy fragment with multiple invalid fields remains invalid."""
        l1 = "V<USAJOHNSON<<ROBERT<<<<<<<<<<<<<<<<<<<<<<<<"
        l2 = "VJDEHE5CK1LAUS931014<11706262AUS<<<<<<<<<<<<"

        parsed = parse_mrva_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["overall_valid"] is False
        # Nationality invalid
        assert parsed["field_validation"]["nationality"]["valid"] is False
        assert parsed["field_validation"]["nationality"]["value"] == "LAU"
        # DOB invalid
        assert parsed["field_validation"]["date_of_birth"]["valid"] is False
        assert parsed["field_validation"]["date_of_birth"]["value"] == "S93101"
        # Expiry invalid
        assert parsed["field_validation"]["date_of_expiry"]["valid"] is False
        assert parsed["field_validation"]["date_of_expiry"]["value"] == "117062"



class TestNationalIDMRZParsing:
    """Tests for ICAO Doc 9303 TD1 (3x30) and TD2 (2x36) National ID MRZ parsing."""

    def test_valid_td1_parsing_and_check_digits(self):
        """Test parsing of standard 3-line TD1 ID card MRZ."""
        l1 = "I<UTOD231458907<<<<<<<<<<<<<<<"
        l2 = "7408122F1204159UTO<<<<<<<<<<<6"
        l3 = "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"

        parsed = parse_td1_mrz(l1, l2, l3)
        assert parsed["valid_format"] is True
        assert parsed["format"] == "TD1"
        assert parsed["overall_valid"] is True
        assert parsed["fields"]["surname"] == "ERIKSSON"
        assert parsed["fields"]["given_names"] == "ANNA MARIA"
        assert parsed["fields"]["document_number"] == "D23145890"
        assert parsed["fields"]["nationality"] == "UTO"
        assert parsed["fields"]["date_of_birth"] == "740812"
        assert parsed["fields"]["sex"] == "F"
        assert parsed["fields"]["date_of_expiry"] == "120415"
        assert parsed["check_digits"]["document_number"] is True
        assert parsed["check_digits"]["date_of_birth"] is True
        assert parsed["check_digits"]["date_of_expiry"] is True
        assert parsed["check_digits"]["composite"] is True

    def test_tampered_td1_composite_check_digit(self):
        """Test detection of tampered composite check digit in TD1 format."""
        l1 = "I<UTOD231458907<<<<<<<<<<<<<<<"
        # Tampered composite CD 6 -> 1
        l2 = "7408122F1204159UTO<<<<<<<<<<<1"
        l3 = "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"

        parsed = parse_td1_mrz(l1, l2, l3)
        assert parsed["valid_format"] is True
        assert parsed["overall_valid"] is False
        assert parsed["check_digits"]["composite"] is False

    def test_malformed_td1_line_length(self):
        """Test rejection of TD1 with invalid line length."""
        l1 = "I<UTOD231458907<<<<<<<<<<<<"  # 28 chars
        l2 = "7408122F1204159UTO<<<<<<<<<<<6"
        l3 = "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"
        parsed = parse_td1_mrz(l1, l2, l3)
        assert parsed["valid_format"] is False
        assert parsed["overall_valid"] is False

    def test_valid_td2_parsing_and_check_digits(self):
        """Test parsing of standard 2-line TD2 ID card MRZ."""
        l1 = "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<"
        l2 = "D231458907UTO7408122F1204159<<<<<<<6"

        parsed = parse_td2_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["format"] == "TD2"
        assert parsed["overall_valid"] is True
        assert parsed["fields"]["surname"] == "ERIKSSON"
        assert parsed["fields"]["given_names"] == "ANNA MARIA"
        assert parsed["fields"]["document_number"] == "D23145890"
        assert parsed["fields"]["nationality"] == "UTO"
        assert parsed["fields"]["date_of_birth"] == "740812"
        assert parsed["fields"]["sex"] == "F"
        assert parsed["fields"]["date_of_expiry"] == "120415"
        assert parsed["check_digits"]["document_number"] is True
        assert parsed["check_digits"]["composite"] is True

    def test_tampered_td2_dob_check_digit(self):
        """Test detection of tampered date of birth check digit in TD2."""
        l1 = "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<"
        # Tampered DOB check digit 2 -> 8
        l2 = "D231458907UTO7408128F1204159<<<<<<<6"

        parsed = parse_td2_mrz(l1, l2)
        assert parsed["valid_format"] is True
        assert parsed["overall_valid"] is False
        assert parsed["check_digits"]["date_of_birth"] is False







