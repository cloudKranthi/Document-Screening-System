"""Passport MRZ detection, multi-candidate scoring, line normalization, and ICAO 9303 TD3 parsing service."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from app.models.schemas import FieldValidationItem, MRZCheckDigits, MRZCorrection, MRZResult
from app.utils.logger import get_logger
from app.utils.mrz_utils import (
    normalize_mrz_line,
    parse_mrva_mrz,
    parse_mrvb_mrz,
    parse_td1_mrz,
    parse_td2_mrz,
    parse_td3_mrz,
)

logger = get_logger(__name__)


class MRZService:
    """Service to detect candidate MRZ lines, score preprocessing outputs, parse ICAO TD3/MRV/TD1/TD2 fields, and validate check digits."""

    @classmethod
    def extract_and_validate_mrz(
        cls,
        raw_ocr_text: str,
        ocr_lines: Optional[List[str]] = None,
        mrz_candidate_texts: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
        include_debug: bool = False
    ) -> Tuple[MRZResult, dict, Optional[Dict[str, Any]]]:
        """Identifies, scores, and parses candidate MRZ lines across TD3, MRV-A, MRV-B, TD1, and TD2 formats."""
        # 1. Normalize candidate source inputs
        sources: List[Tuple[str, str]] = []
        
        if mrz_candidate_texts:
            for item in mrz_candidate_texts:
                if isinstance(item, tuple):
                    src_name, text = item
                    if text and text.strip():
                        sources.append((src_name, text.strip()))
                elif isinstance(item, str) and item.strip():
                    sources.append((f"mrz_pass_{len(sources)+1}", item.strip()))
                    
        if raw_ocr_text and raw_ocr_text.strip():
            sources.append(("general_ocr", raw_ocr_text.strip()))
            
        if ocr_lines:
            sources.append(("general_ocr_lines", "\n".join(ocr_lines)))

        # 2. Extract and score candidate line sets (2-line pairs or 3-line triplets) from all sources
        evaluated_candidates: List[Dict[str, Any]] = []
        
        for source_name, text in sources:
            lines = text.splitlines()
            candidate_groups = cls._find_all_candidate_groups(lines)
            
            for group in candidate_groups:
                parsed = cls._parse_candidate_group(group)
                if not parsed:
                    continue
                    
                score = cls._score_candidate(group, parsed)
                
                # Only keep candidates with meaningful positive structural score
                if score >= 40.0:
                    cand_entry = {
                        "source": source_name,
                        "raw_group": group,
                        "raw_line1": group[0] if len(group) > 0 else None,
                        "raw_line2": group[1] if len(group) > 1 else None,
                        "raw_line3": group[2] if len(group) > 2 else None,
                        "line1": parsed.get("line1"),
                        "line2": parsed.get("line2"),
                        "line3": parsed.get("line3"),
                        "format": parsed.get("format"),
                        "score": score,
                        "valid_format": parsed.get("valid_format", False),
                        "overall_valid": parsed.get("overall_valid", False),
                        "parsed": parsed
                    }
                    evaluated_candidates.append(cand_entry)

        # 3. Sort candidates by score descending
        evaluated_candidates.sort(key=lambda c: c["score"], reverse=True)
        
        # Build debug dictionary
        mrz_debug: Optional[Dict[str, Any]] = None
        if include_debug:
            top_scores = [
                {
                    "source": c["source"],
                    "score": round(c["score"], 2),
                    "format": c.get("format"),
                    "line1": (c["line1"][:22] + "...") if c.get("line1") else "",
                    "line2": (c["line2"][:22] + "...") if c.get("line2") else "",
                    "valid_format": c["valid_format"],
                    "overall_valid": c["overall_valid"]
                }
                for c in evaluated_candidates[:8]
            ]
            mrz_debug = {
                "candidate_count": len(evaluated_candidates),
                "best_candidate": {
                    "source": evaluated_candidates[0]["source"],
                    "score": round(evaluated_candidates[0]["score"], 2),
                    "format": evaluated_candidates[0].get("format"),
                    "line1": evaluated_candidates[0]["line1"],
                    "line2": evaluated_candidates[0]["line2"],
                    "valid_format": evaluated_candidates[0]["valid_format"],
                    "overall_valid": evaluated_candidates[0]["overall_valid"]
                } if evaluated_candidates else None,
                "candidate_scores": top_scores
            }

        # 4. Handle no candidate found or best candidate failing structural thresholds
        if not evaluated_candidates or evaluated_candidates[0]["score"] < 80.0:
            logger.info("No plausible MRZ detected in document.")
            empty_result = MRZResult(
                detected=False,
                format=None,
                line1=None,
                line2=None,
                line3=None,
                raw_line1=None,
                raw_line2=None,
                raw_line3=None,
                valid_format=False,
                check_digits=None,
                field_validation=None,
                overall_valid=False,
                document_code=None,
                issuing_state=None,
                corrections=[],
            )
            return empty_result, {}, mrz_debug

        best = evaluated_candidates[0]
        best_parsed = best["parsed"]

        corrections_list = [
            MRZCorrection(
                line=c["line"],
                position=c["position"],
                from_char=c["from_char"],
                to_char=c["to_char"],
                field=c["field"],
                reason=c["reason"]
            )
            for c in best_parsed.get("corrections", [])
        ]

        if not best_parsed or not best_parsed.get("valid_format"):
            result = MRZResult(
                detected=True,
                format=best_parsed.get("format", best.get("format")),
                line1=best.get("line1"),
                line2=best.get("line2"),
                line3=best.get("line3"),
                raw_line1=best.get("raw_line1"),
                raw_line2=best.get("raw_line2"),
                raw_line3=best.get("raw_line3"),
                valid_format=False,
                check_digits=None,
                field_validation=None,
                overall_valid=False,
                document_code=None,
                issuing_state=None,
                corrections=[],
            )
            return result, {}, mrz_debug

        raw_cd = best_parsed.get("check_digits", {})
        check_digits_model = MRZCheckDigits(
            passport_number=raw_cd.get("passport_number"),
            document_number=raw_cd.get("document_number", raw_cd.get("passport_number")),
            date_of_birth=raw_cd.get("date_of_birth"),
            date_of_expiry=raw_cd.get("date_of_expiry"),
            personal_number=raw_cd.get("personal_number"),
            composite=raw_cd.get("composite"),
        )

        field_validation_dict: Optional[Dict[str, FieldValidationItem]] = None
        if "field_validation" in best_parsed and best_parsed["field_validation"]:
            field_validation_dict = {
                k: FieldValidationItem(
                    valid=v["valid"],
                    value=v.get("value"),
                    reason=v.get("reason")
                )
                for k, v in best_parsed["field_validation"].items()
            }

        result = MRZResult(
            detected=True,
            format=best_parsed.get("format"),
            line1=best_parsed.get("line1"),
            line2=best_parsed.get("line2"),
            line3=best_parsed.get("line3"),
            raw_line1=best_parsed.get("raw_line1", best.get("raw_line1")),
            raw_line2=best_parsed.get("raw_line2", best.get("raw_line2")),
            raw_line3=best_parsed.get("raw_line3", best.get("raw_line3")),
            valid_format=True,
            check_digits=check_digits_model,
            field_validation=field_validation_dict,
            overall_valid=best_parsed.get("overall_valid", False),
            document_code=best_parsed.get("document_code"),
            issuing_state=best_parsed.get("issuing_state"),
            corrections=corrections_list,
        )

        return result, best_parsed.get("fields", {}), mrz_debug

    @classmethod
    def _find_all_candidate_groups(cls, lines: List[str]) -> List[Tuple[str, ...]]:
        """Scans lines to extract potential MRZ candidate line sets (2-line pairs or 3-line triplets)."""
        cleaned = [l.strip() for l in lines if l and len(l.strip()) >= 12]
        groups: List[Tuple[str, ...]] = []
        
        # 1. Look for 3-line triplets (TD1 National ID)
        for i in range(len(cleaned) - 2):
            g3 = (cleaned[i], cleaned[i + 1], cleaned[i + 2])
            # Check if line lengths are near 30 chars (TD1 format)
            if any(24 <= len(l) <= 34 for l in g3):
                groups.append(g3)
                
        # 2. Look for adjacent pairs (TD3, MRVA, MRVB, TD2)
        for i in range(len(cleaned) - 1):
            groups.append((cleaned[i], cleaned[i + 1]))
            
        # 3. Also look for pairs separated by an empty/noise artifact line
        for i in range(len(cleaned) - 2):
            groups.append((cleaned[i], cleaned[i + 2]))
            
        return groups

    @classmethod
    def _parse_candidate_group(cls, group: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """Dispatches a candidate line group to the appropriate format parser with strict plausibility gates."""
        if len(group) == 3:
            l1, l2, l3 = group
            l1_n = normalize_mrz_line(l1)
            l2_n = normalize_mrz_line(l2)
            l3_n = normalize_mrz_line(l3)
            
            # TD1 plausibility checks:
            # 1. Line 1 must start with I, A, or C
            # 2. Line 2 must contain at least 8 digits (DOB + Expiry)
            # 3. All lines between 25 and 35 chars
            # 4. Must contain '<' fillers
            is_td1_l1 = (l1_n.startswith(('I', 'A', 'C')) and 25 <= len(l1_n) <= 35)
            is_td1_l2 = (sum(1 for c in l2_n if c.isdigit()) >= 8 and 25 <= len(l2_n) <= 35)
            is_td1_l3 = (25 <= len(l3_n) <= 35)
            has_fillers = (l1_n.count('<') + l2_n.count('<') + l3_n.count('<')) >= 3
            
            if not (is_td1_l1 and is_td1_l2 and is_td1_l3 and has_fillers):
                return None
                
            l1_pad = l1_n.ljust(30, '<')[:30]
            l2_pad = l2_n.ljust(30, '<')[:30]
            l3_pad = l3_n.ljust(30, '<')[:30]
            return parse_td1_mrz(l1_pad, l2_pad, l3_pad)
            
        elif len(group) == 2:
            l1, l2 = group
            l1_n = normalize_mrz_line(l1)
            l2_n = normalize_mrz_line(l2)
            
            # Visa MRV-A (44 chars) vs MRV-B (36 chars)
            if l1_n.startswith('V') and ('<' in l1_n or any(c.isalpha() for c in l1_n[:5])):
                l2_digits = sum(1 for c in l2_n if c.isdigit())
                if l2_digits >= 4:
                    len_avg = (len(l1_n) + len(l2_n)) / 2.0
                    if len_avg >= 40:
                        l1_pad = l1_n.ljust(44, '<')[:44]
                        l2_pad = l2_n.ljust(44, '<')[:44]
                        return parse_mrva_mrz(l1_pad, l2_pad)
                    elif 30 <= len_avg <= 39:
                        l1_pad = l1_n.ljust(36, '<')[:36]
                        l2_pad = l2_n.ljust(36, '<')[:36]
                        return parse_mrvb_mrz(l1_pad, l2_pad)
                    
            # National ID TD2 (36 chars)

            if l1_n.startswith(('I', 'A', 'C')) and ('<' in l1_n) and (30 <= len(l1_n) <= 40):
                l2_digits = sum(1 for c in l2_n if c.isdigit())
                if l2_digits >= 8 and (30 <= len(l2_n) <= 40):
                    l1_pad = l1_n.ljust(36, '<')[:36]
                    l2_pad = l2_n.ljust(36, '<')[:36]
                    return parse_td2_mrz(l1_pad, l2_pad)
                
            # Default TD3 Passport (44 chars)
            is_plausible_l1 = (l1_n.startswith('P') and '<' in l1_n and 36 <= len(l1_n) <= 52)
            is_plausible_l2 = (36 <= len(l2_n) <= 52 and sum(1 for c in l2_n if c.isdigit()) >= 4)
            if not (is_plausible_l1 and is_plausible_l2):
                if not (36 <= len(l1_n) <= 52 and 36 <= len(l2_n) <= 52 and (l1_n.count('<') + l2_n.count('<')) >= 6):
                    return None
                    
            l1_pad = l1_n.ljust(44, '<')[:44]
            l2_pad = l2_n.ljust(44, '<')[:44]
            return parse_td3_mrz(l1_pad, l2_pad)
            
        return None


    @classmethod
    def _score_candidate(
        cls,
        raw_l1_or_group: Union[str, Tuple[str, ...]],
        raw_l2_or_parsed: Union[str, Dict[str, Any]],
        l1: Optional[str] = None,
        l2: Optional[str] = None,
        parsed: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculates a composite suitability score for an MRZ candidate (pair or triplet).
        
        Scoring components:
          1. Allowed MRZ character ratio ([A-Z0-9<]) (up to 30 pts)
          2. Raw line length proximity to expected format lengths (up to 40 pts)
          3. Structural features and valid format parsing (50 pts)
          4. Check digits validated (up to 75 pts)
          5. Valid country codes (up to 30 pts)
          6. Overall valid composite checksum bonus (100 pts)
        """
        if isinstance(raw_l1_or_group, tuple):
            group = raw_l1_or_group
            parsed_dict = raw_l2_or_parsed if isinstance(raw_l2_or_parsed, dict) else {}
        else:
            group = (str(raw_l1_or_group), str(raw_l2_or_parsed))
            parsed_dict = parsed if parsed else {}
            
        score = 0.0
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        all_raw = "".join(group)
        valid_count = sum(1 for c in all_raw if c.upper() in allowed)
        char_ratio = valid_count / max(1, len(all_raw))
        score += char_ratio * 30.0
        
        expected_len = 30 if len(group) == 3 else (36 if parsed_dict.get("format") in ("MRVB", "TD2") else 44)
        for line in group:
            diff = abs(len(line.strip()) - expected_len)
            score += max(0.0, 15.0 - diff * 2.5)
            
        # Format and Check Digits
        if parsed_dict.get("valid_format"):
            score += 50.0
            cd = parsed_dict.get("check_digits", {})
            if cd:
                for k, v in cd.items():
                    if v is True:
                        score += 15.0
            if parsed_dict.get("overall_valid"):
                score += 100.0
                
            fv = parsed_dict.get("field_validation", {})
            if fv:
                if fv.get("nationality", {}).get("valid"):
                    score += 15.0
                if fv.get("issuing_state", {}).get("valid"):
                    score += 15.0
                    
        return score



