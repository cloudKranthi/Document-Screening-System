"""ICAO Doc 9303 MRZ calculation, normalization, and parsing utilities."""

from typing import Dict, Optional, Tuple


ICAO_WEIGHTS = [7, 3, 1]


def char_to_mrz_value(char: str) -> int:
    """Converts an MRZ character to its ICAO Doc 9303 numeric weight value.
    
    Digits 0-9 -> 0-9
    Letters A-Z -> 10-35
    Filler '<' -> 0
    Others -> 0
    """
    char = char.upper()
    if '0' <= char <= '9':
        return int(char)
    if 'A' <= char <= 'Z':
        return ord(char) - ord('A') + 10
    return 0


def calculate_check_digit(value: str) -> int:
    """Calculates ICAO Doc 9303 check digit using repeating weights [7, 3, 1].
    
    Args:
        value: Input string containing digits, uppercase letters, or '<'.
        
    Returns:
        Modulo 10 check digit (0-9).
    """
    total = 0
    for idx, char in enumerate(value):
        weight = ICAO_WEIGHTS[idx % 3]
        val = char_to_mrz_value(char)
        total += val * weight
    return total % 10


def validate_check_digit(value: str, expected_digit: str | int) -> bool:
    """Validates if the computed check digit matches the expected check digit.
    
    Args:
        value: The string value over which the check digit was computed.
        expected_digit: The check digit character or int from the MRZ.
        
    Returns:
        True if check digit matches, False otherwise.
    """
    if expected_digit is None:
        return False
    expected_str = str(expected_digit).strip()
    if not expected_str:
        return False
    # If the check digit position in MRZ is '<', in some formats it means unused or zero
    if expected_str == '<':
        # If value is all '<' (empty field), check digit '<' or '0' is valid
        if set(value) == {'<'}:
            return True
        expected_str = '0'
    try:
        expected_int = int(expected_str)
    except ValueError:
        return False
        
    calculated = calculate_check_digit(value)
    return calculated == expected_int


def normalize_mrz_line(raw_line: str) -> str:
    """Cleans up and normalizes raw OCR line to standard MRZ character set.
    
    Converts spaces and common OCR artifacts into '<', uppercase letters, or numbers.
    """
    line = raw_line.strip().upper()
    # Replace whitespace and common OCR filler misrecognitions
    line = line.replace(' ', '<').replace('«', '<').replace('‹', '<').replace('{', '<').replace('(', '<')
    line = line.replace('[', '<').replace(']', '<').replace('|', '<')
    # Filter to only allowed MRZ characters: A-Z, 0-9, <
    cleaned = ''.join(c for c in line if ('A' <= c <= 'Z') or ('0' <= c <= '9') or c == '<')
    return cleaned


def normalize_ocr_digits(val: str) -> str:
    """Context-aware OCR character substitution for numeric fields."""
    replacements = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1', 'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
    }
    res = list(val)
    for i, c in enumerate(res):
        if c in replacements:
            res[i] = replacements[c]
    return ''.join(res)


def normalize_ocr_alpha(val: str) -> str:
    """Context-aware OCR character substitution for alphabetic fields."""
    replacements = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '5': 'S',
        '8': 'B',
    }
    res = list(val)
    for i, c in enumerate(res):
        if c in replacements:
            res[i] = replacements[c]
    return ''.join(res)


def parse_td3_mrz(line1: str, line2: str) -> Dict:
    """Parses standard ICAO Doc 9303 TD3 Passport MRZ (2 lines of 44 chars).
    
    Structure:
    Line 1 (44 characters):
      - 00..01 (2 chars): Document code (P, P<, etc.)
      - 02..04 (3 chars): Issuing state (3-letter ICAO code)
      - 05..43 (39 chars): Name (Primary identifier / Surname << Secondary / Given names)
    
    Line 2 (44 characters):
      - 00..08 (9 chars): Passport / Document number
      - 09 (1 char): Passport number check digit
      - 10..12 (3 chars): Nationality (3-letter ICAO code)
      - 13..18 (6 chars): Date of birth (YYMMDD)
      - 19 (1 char): Date of birth check digit
      - 20 (1 char): Sex (M, F, X, <)
      - 21..26 (6 chars): Expiry date (YYMMDD)
      - 27 (1 char): Expiry date check digit
      - 28..41 (14 chars): Personal / Optional number
      - 42 (1 char): Optional number check digit
      - 43 (1 char): Composite check digit over (0-9, 13-19, 21-42)
    """
    line1 = normalize_mrz_line(line1)
    line2 = normalize_mrz_line(line2)

    if len(line1) != 44 or len(line2) != 44:
        return {
            "valid_format": False,
            "error": f"Invalid TD3 line lengths: line1={len(line1)}, line2={len(line2)} (expected 44 each)",
            "line1": line1,
            "line2": line2,
            "fields": {},
            "check_digits": None,
            "overall_valid": False,
        }

    # Extract Line 1 fields
    doc_code = line1[0:2].replace('<', '')
    issuing_state = normalize_ocr_alpha(line1[2:5]).replace('<', '')
    name_raw = line1[5:44]
    
    # Parse Names (Surname << Given names)
    if '<<' in name_raw:
        parts = name_raw.split('<<', 1)
        surname = parts[0].replace('<', ' ').strip()
        given_names = parts[1].replace('<', ' ').strip()
    else:
        surname = name_raw.replace('<', ' ').strip()
        given_names = ""

    # Extract Line 2 fields
    passport_num = line2[0:9].replace('<', '')
    passport_cd = normalize_ocr_digits(line2[9:10])
    
    nationality = normalize_ocr_alpha(line2[10:13]).replace('<', '')
    
    dob_raw = normalize_ocr_digits(line2[13:19])
    dob_cd = normalize_ocr_digits(line2[19:20])
    
    sex = line2[20:21]
    if sex not in ('M', 'F', 'X'):
        sex = '<'
        
    expiry_raw = normalize_ocr_digits(line2[21:27])
    expiry_cd = normalize_ocr_digits(line2[27:28])
    
    personal_num_raw = line2[28:42]
    personal_num = personal_num_raw.replace('<', '')
    personal_cd = line2[42:43]
    
    composite_cd = normalize_ocr_digits(line2[43:44])

    # Validate individual check digits
    # 1. Passport number check digit (chars 0..8)
    valid_passport_cd = validate_check_digit(line2[0:9], passport_cd)
    
    # 2. Date of birth check digit (chars 13..18)
    valid_dob_cd = validate_check_digit(dob_raw, dob_cd)
    
    # 3. Expiry date check digit (chars 21..26)
    valid_expiry_cd = validate_check_digit(expiry_raw, expiry_cd)
    
    # 4. Optional / personal number check digit (chars 28..41)
    if set(personal_num_raw) == {'<'} or personal_cd == '<':
        valid_personal_cd = True
    else:
        valid_personal_cd = validate_check_digit(personal_num_raw, personal_cd)
        
    # 5. Composite check digit: Line 2 positions: [0:10] + [13:20] + [21:43]
    composite_payload = line2[0:10] + line2[13:20] + line2[21:43]
    valid_composite_cd = validate_check_digit(composite_payload, composite_cd)

    check_digit_results = {
        "passport_number": valid_passport_cd,
        "date_of_birth": valid_dob_cd,
        "date_of_expiry": valid_expiry_cd,
        "personal_number": valid_personal_cd,
        "composite": valid_composite_cd,
    }

    # Overall validity requires all primary check digits to match
    overall_valid = (
        valid_passport_cd and
        valid_dob_cd and
        valid_expiry_cd and
        valid_personal_cd and
        valid_composite_cd
    )

    return {
        "valid_format": True,
        "line1": line1,
        "line2": line2,
        "document_code": doc_code,
        "issuing_state": issuing_state,
        "fields": {
            "surname": surname,
            "given_names": given_names,
            "passport_number": passport_num,
            "nationality": nationality,
            "date_of_birth": dob_raw,
            "sex": sex if sex != '<' else "",
            "date_of_expiry": expiry_raw,
            "personal_number": personal_num,
            "issuing_state": issuing_state,
            "document_code": doc_code,
        },
        "check_digits": check_digit_results,
        "overall_valid": overall_valid,
    }
