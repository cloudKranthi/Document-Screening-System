"""Passport MRZ detection, line normalization, and ICAO 9303 TD3 parsing service."""

import re
from typing import List, Optional, Tuple
from app.models.schemas import MRZCheckDigits, MRZResult
from app.utils.logger import get_logger
from app.utils.mrz_utils import normalize_mrz_line, parse_td3_mrz

logger = get_logger(__name__)


class MRZService:
    """Service to detect candidate MRZ lines, parse ICAO TD3 fields, and validate check digits."""

    @classmethod
    def extract_and_validate_mrz(cls, raw_ocr_text: str, ocr_lines: Optional[List[str]] = None) -> Tuple[MRZResult, dict]:
        """Identifies MRZ candidate lines from OCR text, parses TD3 format, and validates check digits.
        
        Returns:
            Tuple of (MRZResult, parsed_fields_dict).
        """
        all_lines = []
        if ocr_lines:
            all_lines.extend(ocr_lines)
        if raw_ocr_text:
            all_lines.extend(raw_ocr_text.splitlines())
            
        candidate_pair = cls._find_td3_mrz_lines(all_lines)
        
        if not candidate_pair:
            logger.info("No MRZ detected in document.")
            empty_result = MRZResult(
                detected=False,
                line1=None,
                line2=None,
                valid_format=False,
                check_digits=None,
                overall_valid=False,
                document_code=None,
                issuing_state=None,
            )
            return empty_result, {}
            
        line1, line2 = candidate_pair
        parsed_data = parse_td3_mrz(line1, line2)
        
        if not parsed_data["valid_format"]:
            result = MRZResult(
                detected=True,
                line1=line1,
                line2=line2,
                valid_format=False,
                check_digits=None,
                overall_valid=False,
                document_code=None,
                issuing_state=None,
            )
            return result, {}
            
        check_digits_model = MRZCheckDigits(
            passport_number=parsed_data["check_digits"]["passport_number"],
            date_of_birth=parsed_data["check_digits"]["date_of_birth"],
            date_of_expiry=parsed_data["check_digits"]["date_of_expiry"],
            personal_number=parsed_data["check_digits"].get("personal_number"),
            composite=parsed_data["check_digits"]["composite"],
        )
        
        result = MRZResult(
            detected=True,
            line1=parsed_data["line1"],
            line2=parsed_data["line2"],
            valid_format=True,
            check_digits=check_digits_model,
            overall_valid=parsed_data["overall_valid"],
            document_code=parsed_data.get("document_code"),
            issuing_state=parsed_data.get("issuing_state"),
        )
        
        return result, parsed_data.get("fields", {})

    @staticmethod
    def _find_td3_mrz_lines(lines: List[str]) -> Optional[Tuple[str, str]]:
        """Scans lines to detect standard 2-line TD3 MRZ candidates."""
        cleaned_lines = [normalize_mrz_line(l) for l in lines]
        cleaned_lines = [l for l in cleaned_lines if len(l) >= 30]
        
        # Strategy 1: Look for line starting with 'P<' or 'P[A-Z]' followed by 44-char candidate
        for i in range(len(cleaned_lines) - 1):
            l1 = cleaned_lines[i]
            l2 = cleaned_lines[i + 1]
            
            # Passport type indicator check
            if l1.startswith('P') and ('<' in l1):
                # Attempt to pad/trim to 44 characters if minor OCR drift
                if 40 <= len(l1) <= 46 and 40 <= len(l2) <= 46:
                    l1_padded = l1.ljust(44, '<')[:44]
                    l2_padded = l2.ljust(44, '<')[:44]
                    return l1_padded, l2_padded
                    
        # Strategy 2: Look for any two consecutive lines of length ~44 with lots of '<'
        for i in range(len(cleaned_lines) - 1):
            l1 = cleaned_lines[i]
            l2 = cleaned_lines[i + 1]
            if 40 <= len(l1) <= 46 and 40 <= len(l2) <= 46:
                if l1.count('<') >= 2 or l2.count('<') >= 2:
                    l1_padded = l1.ljust(44, '<')[:44]
                    l2_padded = l2.ljust(44, '<')[:44]
                    return l1_padded, l2_padded
                    
        return None
