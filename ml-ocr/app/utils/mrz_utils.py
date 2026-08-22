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
    # Replace common OCR filler misrecognitions
    line = line.replace('«', '<').replace('‹', '<').replace('{', '<').replace('(', '<')
    line = line.replace('[', '<').replace(']', '<').replace('|', '<')
    
    # Check if line looks like TD3 Line 1 (starts with P and has names/country) vs Line 2
    if line.startswith('P') and ('<' in line or not any(c.isdigit() for c in line[:5])):
        # Line 1: spaces inside name/country fields represent '<' fillers
        line = line.replace(' ', '<')
    else:
        # Line 2: spaces between fields are OCR word splits and should be stripped
        line = line.replace(' ', '')
        
    # Filter to only allowed MRZ characters: A-Z, 0-9, <
    cleaned = ''.join(c for c in line if ('A' <= c <= 'Z') or ('0' <= c <= '9') or c == '<')
    return cleaned


def _clean_mrz_name_component(raw_component: str) -> str:
    """Cleans an MRZ name component (surname or given names) by removing OCR filler artifact noise (e.g. SRSSSESSESSSS, K, E)."""
    if not raw_component:
        return ""
    spaced = raw_component.replace('<', ' ')
    words = [w.strip() for w in spaced.split() if w.strip()]
    if not words:
        return ""
    
    clean_words = []
    for i, word in enumerate(words):
        w_up = word.upper()
        # 1. Repetitive filler noise strings (e.g. 'SRSSSESSESSSS', 'KKKKKK', 'EEEE', 'XXXX', 'RRRR')
        if len(w_up) >= 3 and all(c in 'SREKCX<' for c in w_up) and len(set(w_up)) <= 3:
            continue
        # 2. Trailing single-letter OCR artifacts at the end of the name when preceding words exist
        if i > 0 and i == len(words) - 1 and len(w_up) == 1 and w_up in 'SREKCX<':
            continue
        clean_words.append(word)
        
    return " ".join(clean_words).strip()



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


def _find_field_digit_correction(
    raw_val: str,
    raw_cd: str,
    field_name: str,
    base_pos: int,
    line_num: int,
    validate_fn=None
) -> Tuple[str, str, list[dict]]:
    """Attempts to find a single, mathematically verified digit substitution that satisfies validate_check_digit.
    
    Guarantees:
      - Only applies substitutions if they yield a mathematically valid ICAO 9303 check digit.
      - If no substitution works or multiple ambiguous substitutions work, leaves characters unchanged.
    """
    corrections = []
    
    # Common OCR digit confusion mapping
    CONFUSIONS = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1', 'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'J': '9', 'g': '9',
    }
    
    cand_cd = raw_cd
    cd_corrected = False
    if cand_cd in CONFUSIONS:
        cand_cd = CONFUSIONS[cand_cd]
        cd_corrected = True
        
    curr_val = raw_val
    # 1. Check if already valid as-is
    if validate_check_digit(curr_val, cand_cd) and (validate_fn is None or validate_fn(curr_val)):
        if cd_corrected:
            corrections.append({
                "line": line_num,
                "position": base_pos + len(raw_val),
                "from_char": raw_cd,
                "to_char": cand_cd,
                "field": f"{field_name}_check_digit",
                "reason": f"Corrected check digit '{raw_cd}' -> '{cand_cd}'"
            })
        return curr_val, cand_cd, corrections
        
    # 2. Search for candidate single-character substitutions in raw_val
    valid_candidates = []
    for i, c in enumerate(raw_val):
        if c in CONFUSIONS:
            sub = CONFUSIONS[c]
            cand = curr_val[:i] + sub + curr_val[i+1:]
            if validate_check_digit(cand, cand_cd) and (validate_fn is None or validate_fn(cand)):
                valid_candidates.append((cand, [(i, c, sub)]))
                
    # 3. If no single substitution worked, try full substitution across all confusable positions
    if not valid_candidates:
        full_sub = "".join(CONFUSIONS.get(c, c) for c in raw_val)
        if full_sub != curr_val:
            if validate_check_digit(full_sub, cand_cd) and (validate_fn is None or validate_fn(full_sub)):
                diffs = [(i, c, full_sub[i]) for i, c in enumerate(raw_val) if c != full_sub[i]]
                valid_candidates.append((full_sub, diffs))
                
    # 4. If exactly ONE candidate restores mathematical validity, apply it
    if len(valid_candidates) == 1:
        best_val, diffs = valid_candidates[0]
        if cd_corrected:
            corrections.append({
                "line": line_num,
                "position": base_pos + len(raw_val),
                "from_char": raw_cd,
                "to_char": cand_cd,
                "field": f"{field_name}_check_digit",
                "reason": f"Corrected check digit '{raw_cd}' -> '{cand_cd}'"
            })
        for idx, orig_c, new_c in diffs:
            corrections.append({
                "line": line_num,
                "position": base_pos + idx,
                "from_char": orig_c,
                "to_char": new_c,
                "field": field_name,
                "reason": f"Check-digit verified substitution '{orig_c}' -> '{new_c}'"
            })
        return best_val, cand_cd, corrections
        
    # 5. Otherwise leave unchanged
    return raw_val, raw_cd, []


ICAO_COUNTRY_CODES: frozenset[str] = frozenset({
    # ISO 3166-1 alpha-3 official country codes
    "ABW", "AFG", "AGO", "AIA", "ALA", "ALB", "AND", "ARE", "ARG", "ARM", "ASM", "ATA", "ATF", "ATG", "AUS", "AUT", "AZE",
    "BDI", "BEL", "BEN", "BES", "BFA", "BGD", "BGR", "BHR", "BHS", "BIH", "BLM", "BLR", "BLZ", "BMU", "BOL", "BRA", "BRB", "BRN", "BTN", "BVT", "BWA",
    "CAF", "CAN", "CCK", "CHE", "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COK", "COL", "COM", "CPV", "CRI", "CUB", "CUW", "CXR", "CYM", "CYP", "CZE",
    "DEU", "DJI", "DMA", "DNK", "DOM", "DZA",
    "ECU", "EGY", "ERI", "ESH", "ESP", "EST", "ETH",
    "FIN", "FJI", "FLK", "FRA", "FRO", "FSM",
    "GAB", "GBR", "GEO", "GGY", "GHA", "GIB", "GIN", "GLP", "GMB", "GNB", "GNQ", "GRC", "GRD", "GRL", "GTM", "GUF", "GUM", "GUY",
    "HKG", "HMD", "HND", "HRV", "HTI", "HUN",
    "IDN", "IMN", "IND", "IOT", "IRL", "IRN", "IRQ", "ISL", "ISR", "ITA",
    "JAM", "JEY", "JOR", "JPN",
    "KAZ", "KEN", "KGZ", "KHM", "KIR", "KNA", "KOR", "KWT",
    "LAO", "LBN", "LBR", "LBY", "LCA", "LIE", "LKA", "LSO", "LTU", "LUX", "LVA",
    "MAC", "MAF", "MAR", "MCO", "MDA", "MDG", "MDV", "MEX", "MHL", "MKD", "MLI", "MLT", "MMR", "MNE", "MNG", "MNP", "MOZ", "MRT", "MSR", "MTQ", "MUS", "MWI", "MYS", "MYT",
    "NAM", "NCL", "NER", "NFK", "NGA", "NIC", "NIU", "NLD", "NOR", "NPL", "NRU", "NZL",
    "OMN",
    "PAK", "PAN", "PCN", "PER", "PHL", "PLW", "PNG", "POL", "PRI", "PRK", "PRT", "PRY", "PSE", "PYF",
    "QAT",
    "REU", "ROU", "RUS", "RWA",
    "SAU", "SDN", "SEN", "SGP", "SGS", "SHN", "SJM", "SLB", "SLE", "SLV", "SMR", "SOM", "SPM", "SRB", "SSD", "STP", "SUR", "SVK", "SVN", "SWE", "SWZ", "SXM", "SYC", "SYR",
    "TCA", "TCD", "TGO", "THA", "TJK", "TKL", "TKM", "TLS", "TON", "TTO", "TUN", "TUR", "TUV", "TWN", "TZA",
    "UGA", "UKR", "UMI", "URY", "USA", "UZB",
    "VAT", "VCT", "VEN", "VGB", "VIR", "VNM", "VUT",
    "WLF", "WSM",
    "YEM",
    "ZAF", "ZMB", "ZWE",
    # ICAO Doc 9303 Designated Special Organization & Refugee Codes
    "UTO", "UNA", "UNK", "XOM", "XXA", "XXB", "XXC", "XXX", "XPO", "GBD", "GBN", "GBO", "GBS", "RKS"
})


def _correct_country_code(
    raw_code: str,
    line_num: int,
    base_pos: int,
    field_name: str
) -> Tuple[str, list[dict]]:
    """Validates and corrects a 3-letter ICAO/ISO country code against known country codes.
    
    Guarantees:
      - Never normalizes blindly (e.g. '100' does NOT become 'IOO' because 'IOO' is invalid).
      - Only accepts a candidate substitution if the resulting code is a genuine ICAO/ISO country code (e.g. '1ND' -> 'IND', 'UT0' -> 'UTO').
      - If no valid code is found or if ambiguous, preserves the raw characters.
    """
    code = raw_code.strip().upper()
    if len(code) != 3:
        return raw_code, []
        
    # 1. If already a valid ICAO country code, accept as-is with no corrections
    if code in ICAO_COUNTRY_CODES:
        return code, []
        
    # 2. Candidate substitutions for OCR confusions in country codes
    CONFUSIONS = {
        '0': ['O'],
        '1': ['I'],
        '2': ['Z'],
        '5': ['S'],
        '8': ['B'],
        '6': ['G', 'B'],
        'V': ['U'],
        'U': ['V'],
    }
    
    char_options = []
    for c in code:
        opts = [c]
        if c in CONFUSIONS:
            opts.extend(CONFUSIONS[c])
        char_options.append(opts)
        
    valid_matches = []
    for c0 in char_options[0]:
        for c1 in char_options[1]:
            for c2 in char_options[2]:
                cand = f"{c0}{c1}{c2}"
                if cand in ICAO_COUNTRY_CODES:
                    diffs = [(i, code[i], cand[i]) for i in range(3) if code[i] != cand[i]]
                    valid_matches.append((cand, diffs))
                    
    # Sort by fewest differences (1-char diff prioritized over 2-char diff)
    valid_matches.sort(key=lambda m: len(m[1]))
    
    # If exactly ONE unique candidate (or single best with lowest distance) matches a valid country code
    if valid_matches:
        min_diff = len(valid_matches[0][1])
        best_matches = [m for m in valid_matches if len(m[1]) == min_diff]
        
        if len(best_matches) == 1 and min_diff > 0:
            best_code, diffs = best_matches[0]
            corrections = []
            for idx, orig_c, new_c in diffs:
                corrections.append({
                    "line": line_num,
                    "position": base_pos + idx,
                    "from_char": orig_c,
                    "to_char": new_c,
                    "field": field_name,
                    "reason": f"Corrected '{orig_c}' -> '{new_c}' to restore valid ICAO/ISO country code '{best_code}'"
                })
            return best_code, corrections
            
    # If no valid country code found or ambiguous, leave unchanged
    return raw_code, []


def correct_td3_mrz(raw_line1: str, raw_line2: str) -> Tuple[str, str, list[dict]]:
    """Performs deterministic, position-aware character corrections on TD3 lines.
    
    Guarantees:
      - Raw MRZ strings are not modified in-place.
      - Fields with check digits are only corrected if mathematically verified.
      - Name fields are never altered aggressively.
      - Country codes are verified against official ICAO/ISO 3166-1 alpha-3 codes.
      - Every change is logged with exact line, position, from, to, field, and reason.
    """
    corrections: list[dict] = []
    l1 = list(raw_line1)
    l2 = list(raw_line2)

    # ------------------ LINE 1 CORRECTIONS ------------------
    # 1. Document Code: Position 0 must be 'P' for TD3 passports
    if len(l1) > 0 and l1[0] != 'P':
        orig = l1[0]
        l1[0] = 'P'
        corrections.append({
            "line": 1,
            "position": 0,
            "from_char": orig,
            "to_char": "P",
            "field": "document_code",
            "reason": "Standard ICAO Doc 9303 TD3 passport prefix 'P'"
        })
        
    # 2. Issuing State (positions 2..4): Verified against ICAO/ISO country codes
    if len(l1) >= 5:
        raw_state = "".join(l1[2:5])
        corr_state, state_corrs = _correct_country_code(
            raw_code=raw_state,
            line_num=1,
            base_pos=2,
            field_name="issuing_state"
        )
        for i, c in enumerate(corr_state):
            l1[2 + i] = c
        corrections.extend(state_corrs)
            
    # 3. Trailing Fillers in Line 1 (strictly after name section): normalize misrecognized '<'
    l1_str = "".join(l1)
    if '<<' in l1_str[5:]:
        # Find last alphabetic character before trailing fillers
        last_alpha_idx = -1
        for i in range(len(l1) - 1, 4, -1):
            if l1[i].isalpha() and l1[i] not in ('K', 'E', 'C', '«', '‹'):
                last_alpha_idx = i
                break
        start_filler_idx = max(30, last_alpha_idx + 2 if last_alpha_idx > 0 else 30)
        for pos in range(start_filler_idx, len(l1)):
            if l1[pos] in ('K', 'E', 'C', '«', '‹', '{', '(', '[', ']', '|'):
                orig = l1[pos]
                l1[pos] = '<'
                corrections.append({
                    "line": 1,
                    "position": pos,
                    "from_char": orig,
                    "to_char": "<",
                    "field": "filler",
                    "reason": f"Normalized trailing filler character '{orig}' -> '<' in permitted filler zone"
                })

    # ------------------ LINE 2 CORRECTIONS ------------------
    if len(l2) == 44:
        # 1. Passport Number [0:9] and Check Digit [9]
        raw_p_num = "".join(l2[0:9])
        raw_p_cd = l2[9]
        corr_p_num, corr_p_cd, p_corrs = _find_field_digit_correction(
            raw_val=raw_p_num,
            raw_cd=raw_p_cd,
            field_name="passport_number",
            base_pos=0,
            line_num=2
        )
        for i, c in enumerate(corr_p_num):
            l2[i] = c
        l2[9] = corr_p_cd
        corrections.extend(p_corrs)

        # 2. Nationality [10:13]: Verified against ICAO/ISO country codes
        raw_nat = "".join(l2[10:13])
        corr_nat, nat_corrs = _correct_country_code(
            raw_code=raw_nat,
            line_num=2,
            base_pos=10,
            field_name="nationality"
        )
        for i, c in enumerate(corr_nat):
            l2[10 + i] = c
        corrections.extend(nat_corrs)

        # 3. Date of Birth [13:19] and Check Digit [19]
        raw_dob = "".join(l2[13:19])
        raw_dob_cd = l2[19]
        def validate_dob_dates(d: str) -> bool:
            try:
                mm = int(d[2:4])
                dd = int(d[4:6])
                return 1 <= mm <= 12 and 1 <= dd <= 31
            except ValueError:
                return False

        corr_dob, corr_dob_cd, dob_corrs = _find_field_digit_correction(
            raw_val=raw_dob,
            raw_cd=raw_dob_cd,
            field_name="date_of_birth",
            base_pos=13,
            line_num=2,
            validate_fn=validate_dob_dates
        )
        for i, c in enumerate(corr_dob):
            l2[13 + i] = c
        l2[19] = corr_dob_cd
        corrections.extend(dob_corrs)

        # 4. Sex [20]: 'M', 'F', 'X', '<'
        if l2[20] not in ('M', 'F', 'X', '<'):
            orig_sex = l2[20]
            if orig_sex in ('0', '1', 'I', 'L'):
                l2[20] = '<'
                corrections.append({
                    "line": 2,
                    "position": 20,
                    "from_char": orig_sex,
                    "to_char": "<",
                    "field": "sex",
                    "reason": f"Normalized unrecognized sex character '{orig_sex}' to filler '<'"
                })
            elif orig_sex == 'E':
                l2[20] = 'F'
                corrections.append({
                    "line": 2,
                    "position": 20,
                    "from_char": "E",
                    "to_char": "F",
                    "field": "sex",
                    "reason": "Corrected OCR confusion 'E' -> 'F' at sex field"
                })

        # 5. Date of Expiry [21:27] and Check Digit [27]
        raw_exp = "".join(l2[21:27])
        raw_exp_cd = l2[27]
        def validate_exp_dates(d: str) -> bool:
            try:
                mm = int(d[2:4])
                dd = int(d[4:6])
                return 1 <= mm <= 12 and 1 <= dd <= 31
            except ValueError:
                return False

        corr_exp, corr_exp_cd, exp_corrs = _find_field_digit_correction(
            raw_val=raw_exp,
            raw_cd=raw_exp_cd,
            field_name="date_of_expiry",
            base_pos=21,
            line_num=2,
            validate_fn=validate_exp_dates
        )
        for i, c in enumerate(corr_exp):
            l2[21 + i] = c
        l2[27] = corr_exp_cd
        corrections.extend(exp_corrs)

        # 6. Composite Check Digit [43]
        DIGIT_CONFUSIONS = {
            'O': '0', 'D': '0', 'Q': '0',
            'I': '1', 'L': '1', 'l': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'J': '9', 'g': '9',
        }
        if l2[43] in DIGIT_CONFUSIONS:
            orig_comp_cd = l2[43]
            cand_comp_cd = DIGIT_CONFUSIONS[orig_comp_cd]
            comp_payload = "".join(l2[0:10]) + "".join(l2[13:20]) + "".join(l2[21:43])
            if validate_check_digit(comp_payload, cand_comp_cd):
                l2[43] = cand_comp_cd
                corrections.append({
                    "line": 2,
                    "position": 43,
                    "from_char": orig_comp_cd,
                    "to_char": cand_comp_cd,
                    "field": "composite_check_digit",
                    "reason": f"Corrected composite check digit '{orig_comp_cd}' -> '{cand_comp_cd}'"
                })

    return "".join(l1), "".join(l2), corrections



def parse_td3_mrz(line1: str, line2: str) -> Dict:
    """Parses standard ICAO Doc 9303 TD3 Passport MRZ (2 lines of 44 chars) with check-digit verified correction.
    
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
    raw_l1 = normalize_mrz_line(line1)
    raw_l2 = normalize_mrz_line(line2)

    if len(raw_l1) != 44 or len(raw_l2) != 44:
        return {
            "valid_format": False,
            "error": f"Invalid TD3 line lengths: line1={len(raw_l1)}, line2={len(raw_l2)} (expected 44 each)",
            "raw_line1": raw_l1,
            "raw_line2": raw_l2,
            "line1": raw_l1,
            "line2": raw_l2,
            "corrections": [],
            "fields": {},
            "check_digits": None,
            "overall_valid": False,
        }

    # Apply deterministic, position-aware, check-digit verified corrections
    corr_l1, corr_l2, corrections = correct_td3_mrz(raw_l1, raw_l2)

    # Extract Line 1 fields from corrected representation
    doc_code = corr_l1[0:2].replace('<', '')
    issuing_state = corr_l1[2:5].replace('<', '')
    name_raw = corr_l1[5:44]
    
    # Parse Names (Surname << Given names)
    name_clean = name_raw.lstrip('<')
    if '<<' in name_clean:
        parts = name_clean.split('<<', 1)
        surname = _clean_mrz_name_component(parts[0])
        given_names = _clean_mrz_name_component(parts[1])
    else:
        surname = _clean_mrz_name_component(name_clean)
        given_names = ""


    # Extract Line 2 fields from corrected representation
    passport_num = corr_l2[0:9].replace('<', '')
    passport_cd = corr_l2[9:10]
    
    nationality = corr_l2[10:13].replace('<', '')
    
    dob_raw = corr_l2[13:19]
    dob_cd = corr_l2[19:20]
    
    sex = corr_l2[20:21]
    if sex not in ('M', 'F', 'X'):
        sex = '<'
        
    expiry_raw = corr_l2[21:27]
    expiry_cd = corr_l2[27:28]
    
    personal_num_raw = corr_l2[28:42]
    personal_num = personal_num_raw.replace('<', '')
    personal_cd = corr_l2[42:43]
    
    composite_cd = corr_l2[43:44]

    # Validate individual check digits strictly against ICAO 9303 modulo-10 algorithm
    # 1. Passport number check digit (chars 0..8)
    valid_passport_cd = validate_check_digit(corr_l2[0:9], passport_cd)
    
    # 2. Date of birth check digit (chars 13..18)
    valid_dob_cd = validate_check_digit(dob_raw, dob_cd)
    valid_dob_dates = False
    try:
        mm = int(dob_raw[2:4])
        dd = int(dob_raw[4:6])
        valid_dob_dates = (1 <= mm <= 12 and 1 <= dd <= 31)
    except (ValueError, IndexError):
        valid_dob_dates = False
    
    # 3. Expiry date check digit (chars 21..26)
    valid_expiry_cd = validate_check_digit(expiry_raw, expiry_cd)
    valid_exp_dates = False
    try:
        mm = int(expiry_raw[2:4])
        dd = int(expiry_raw[4:6])
        valid_exp_dates = (1 <= mm <= 12 and 1 <= dd <= 31)
    except (ValueError, IndexError):
        valid_exp_dates = False
    
    # 4. Optional / personal number check digit (chars 28..41)
    if set(personal_num_raw) == {'<'} or personal_cd == '<':
        valid_personal_cd = True
    else:
        valid_personal_cd = validate_check_digit(personal_num_raw, personal_cd)
        
    # 5. Composite check digit: Line 2 positions: [0:10] + [13:20] + [21:43]
    composite_payload = corr_l2[0:10] + corr_l2[13:20] + corr_l2[21:43]
    valid_composite_cd = validate_check_digit(composite_payload, composite_cd)

    check_digit_results = {
        "passport_number": valid_passport_cd,
        "date_of_birth": valid_dob_cd,
        "date_of_expiry": valid_expiry_cd,
        "personal_number": valid_personal_cd,
        "composite": valid_composite_cd,
    }

    # ------------------ FIELD-LEVEL VALIDATION ------------------
    valid_nationality = (len(nationality) == 3 and nationality in ICAO_COUNTRY_CODES)
    valid_issuing_state = (len(issuing_state) == 3 and issuing_state in ICAO_COUNTRY_CODES)
    valid_sex = (sex in ('M', 'F', 'X'))

    field_validation = {
        "passport_number": {
            "valid": valid_passport_cd,
            "value": passport_num,
            "reason": None if valid_passport_cd else "Passport number check digit validation failed"
        },
        "issuing_state": {
            "valid": valid_issuing_state,
            "value": issuing_state,
            "reason": None if valid_issuing_state else f"Invalid issuing state code: '{issuing_state}' is not a valid 3-letter ICAO/ISO country code"
        },
        "nationality": {
            "valid": valid_nationality,
            "value": nationality,
            "reason": None if valid_nationality else f"Invalid nationality code: '{nationality}' is not a valid 3-letter ICAO/ISO country code"
        },
        "date_of_birth": {
            "valid": (valid_dob_cd and valid_dob_dates),
            "value": dob_raw,
            "reason": None if (valid_dob_cd and valid_dob_dates) else (
                "DOB check digit validation failed" if not valid_dob_cd else "Invalid date of birth format"
            )
        },
        "sex": {
            "valid": valid_sex,
            "value": sex if sex != '<' else "",
            "reason": None if valid_sex else "Missing or invalid MRZ sex value"
        },
        "date_of_expiry": {
            "valid": (valid_expiry_cd and valid_exp_dates),
            "value": expiry_raw,
            "reason": None if (valid_expiry_cd and valid_exp_dates) else (
                "Expiry date check digit validation failed" if not valid_expiry_cd else "Invalid date of expiry format"
            )
        },
        "personal_number": {
            "valid": valid_personal_cd,
            "value": personal_num,
            "reason": None if valid_personal_cd else "Personal / optional number check digit validation failed"
        },
        "composite": {
            "valid": valid_composite_cd,
            "value": composite_cd,
            "reason": None if valid_composite_cd else "Composite check digit validation failed"
        },
    }

    # Overall validity requires all primary check digits AND structural fields to match
    overall_valid = (
        valid_passport_cd and
        valid_dob_cd and
        valid_expiry_cd and
        valid_personal_cd and
        valid_composite_cd and
        valid_nationality and
        valid_issuing_state
    )

    return {
        "valid_format": True,
        "format": "TD3",
        "raw_line1": raw_l1,
        "raw_line2": raw_l2,
        "line1": corr_l1,
        "line2": corr_l2,
        "corrections": corrections,
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
        "field_validation": field_validation,
        "overall_valid": overall_valid,
    }


def correct_mrv_mrz(raw_l1: str, raw_l2: str, format_type: str = "MRVA") -> Tuple[str, str, list[dict]]:
    """Context-aware, position-specific, and check-digit verified correction for MRV-A and MRV-B Visa MRZs.
    
    Guarantees:
      - Raw characters are preserved unless structurally and mathematically verified.
      - Country codes are validated against ICAO_COUNTRY_CODES.
      - Dates of birth and expiry are verified for numeric YYMMDD and calendar bounds.
      - Never applies TD3 composite check digit or personal number logic.
    """
    expected_len = 44 if format_type == "MRVA" else 36
    if len(raw_l1) != expected_len or len(raw_l2) != expected_len:
        return raw_l1, raw_l2, []

    l1 = list(raw_l1)
    l2 = list(raw_l2)
    corrections: list[dict] = []

    def is_valid_calendar_date(d: str) -> bool:
        if len(d) != 6 or not d.isdigit():
            return False
        try:
            mm = int(d[2:4])
            dd = int(d[4:6])
            return 1 <= mm <= 12 and 1 <= dd <= 31
        except ValueError:
            return False

    # 1. Line 1: Issuing State (chars 2..4)
    raw_state = "".join(l1[2:5])
    corr_state, state_corrs = _correct_country_code(raw_state, line_num=1, base_pos=2, field_name="issuing_state")
    if corr_state != raw_state:
        for idx, orig_c, new_c in [(i, raw_state[i], corr_state[i]) for i in range(3) if raw_state[i] != corr_state[i]]:
            l1[2 + idx] = new_c
        corrections.extend(state_corrs)

    # 2. Line 2: Visa Document Number (chars 0..8) & Check Digit (char 9)
    raw_num = "".join(l2[0:9])
    raw_num_cd = l2[9]
    corr_num, corr_num_cd, num_corrs = _find_field_digit_correction(
        raw_num, raw_num_cd, field_name="document_number", base_pos=0, line_num=2
    )
    if corr_num != raw_num:
        for idx in range(9):
            l2[idx] = corr_num[idx]
    if corr_num_cd != raw_num_cd:
        l2[9] = corr_num_cd
    corrections.extend(num_corrs)

    # 3. Line 2: Nationality (chars 10..12)
    raw_nat = "".join(l2[10:13])
    corr_nat, nat_corrs = _correct_country_code(raw_nat, line_num=2, base_pos=10, field_name="nationality")
    if corr_nat != raw_nat:
        for idx, orig_c, new_c in [(i, raw_nat[i], corr_nat[i]) for i in range(3) if raw_nat[i] != corr_nat[i]]:
            l2[10 + idx] = new_c
        corrections.extend(nat_corrs)

    # 4. Line 2: Date of Birth (chars 13..18) & Check Digit (char 19)
    raw_dob = "".join(l2[13:19])
    raw_dob_cd = l2[19]
    corr_dob, corr_dob_cd, dob_corrs = _find_field_digit_correction(
        raw_dob, raw_dob_cd, field_name="date_of_birth", base_pos=13, line_num=2, validate_fn=is_valid_calendar_date
    )
    if corr_dob != raw_dob:
        for idx in range(6):
            l2[13 + idx] = corr_dob[idx]
    if corr_dob_cd != raw_dob_cd:
        l2[19] = corr_dob_cd
    corrections.extend(dob_corrs)

    # 5. Line 2: Date of Expiry (chars 21..26) & Check Digit (char 27)
    raw_exp = "".join(l2[21:27])
    raw_exp_cd = l2[27]
    corr_exp, corr_exp_cd, exp_corrs = _find_field_digit_correction(
        raw_exp, raw_exp_cd, field_name="date_of_expiry", base_pos=21, line_num=2, validate_fn=is_valid_calendar_date
    )
    if corr_exp != raw_exp:
        for idx in range(6):
            l2[21 + idx] = corr_exp[idx]
    if corr_exp_cd != raw_exp_cd:
        l2[27] = corr_exp_cd
    corrections.extend(exp_corrs)

    return "".join(l1), "".join(l2), corrections


def parse_mrva_mrz(line1: str, line2: str) -> Dict:
    """Parses ICAO Doc 9303 MRV-A Visa MRZ (2 lines of 44 characters).
    
    Structure:
    Line 1 (44 characters):
      - 00..01 (2 chars): Document code (V, V<, VA, etc.)
      - 02..04 (3 chars): Issuing state (3-letter ICAO code)
      - 05..43 (39 chars): Name (Surname << Given names)
      
    Line 2 (44 characters):
      - 00..08 (9 chars): Visa / Document number
      - 09 (1 char): Visa number check digit
      - 10..12 (3 chars): Nationality (3-letter ICAO code)
      - 13..18 (6 chars): Date of birth (YYMMDD)
      - 19 (1 char): Date of birth check digit
      - 20 (1 char): Sex (M, F, <)
      - 21..26 (6 chars): Expiry date (YYMMDD)
      - 27 (1 char): Expiry date check digit
      - 28..43 (16 chars): Optional data elements
    """
    raw_l1 = normalize_mrz_line(line1)
    raw_l2 = normalize_mrz_line(line2)

    if len(raw_l1) != 44 or len(raw_l2) != 44:
        return {
            "valid_format": False,
            "format": "MRVA",
            "error": f"Invalid MRV-A line lengths: line1={len(raw_l1)}, line2={len(raw_l2)} (expected 44 each)",
            "raw_line1": raw_l1,
            "raw_line2": raw_l2,
            "line1": raw_l1,
            "line2": raw_l2,
            "corrections": [],
            "fields": {},
            "check_digits": None,
            "field_validation": None,
            "overall_valid": False,
        }

    corr_l1, corr_l2, corrections = correct_mrv_mrz(raw_l1, raw_l2, format_type="MRVA")

    # Line 1 fields
    doc_code = corr_l1[0:2].replace('<', '')
    issuing_state = corr_l1[2:5].replace('<', '')
    name_raw = corr_l1[5:44].lstrip('<')
    if '<<' in name_raw:
        parts = name_raw.split('<<', 1)
        surname = _clean_mrz_name_component(parts[0])
        given_names = _clean_mrz_name_component(parts[1])
    else:
        surname = _clean_mrz_name_component(name_raw)
        given_names = ""


    # Line 2 fields
    visa_num = corr_l2[0:9].replace('<', '')
    visa_cd = corr_l2[9:10]
    nationality = corr_l2[10:13].replace('<', '')
    dob_raw = corr_l2[13:19]
    dob_cd = corr_l2[19:20]
    raw_sex = corr_l2[20:21]
    sex = raw_sex if raw_sex in ('M', 'F', 'X') else '<'
    expiry_raw = corr_l2[21:27]
    expiry_cd = corr_l2[27:28]
    optional_data = corr_l2[28:44].replace('<', '')

    # Validations
    valid_visa_cd = validate_check_digit(corr_l2[0:9], visa_cd)

    # DOB validation
    valid_dob_cd = validate_check_digit(dob_raw, dob_cd)
    dob_has_non_digits = (len(dob_raw) != 6 or not dob_raw.isdigit())
    valid_dob_dates = False
    dob_reason = None
    if dob_has_non_digits:
        dob_reason = f"Date of birth contains non-numeric characters: '{dob_raw}'"
    else:
        try:
            mm = int(dob_raw[2:4])
            dd = int(dob_raw[4:6])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                valid_dob_dates = True
            else:
                dob_reason = f"Invalid DOB date values: month={dob_raw[2:4]}, day={dob_raw[4:6]}"
        except ValueError:
            dob_reason = f"Invalid DOB date format: '{dob_raw}'"
    if not dob_reason and not valid_dob_cd:
        dob_reason = "DOB check digit validation failed"

    # Expiry validation
    valid_expiry_cd = validate_check_digit(expiry_raw, expiry_cd)
    exp_has_non_digits = (len(expiry_raw) != 6 or not expiry_raw.isdigit())
    valid_exp_dates = False
    exp_reason = None
    if exp_has_non_digits:
        exp_reason = f"Expiry date contains non-numeric characters: '{expiry_raw}'"
    else:
        try:
            mm = int(expiry_raw[2:4])
            dd = int(expiry_raw[4:6])
            if not (1 <= mm <= 12):
                exp_reason = f"Invalid expiry date: month '{expiry_raw[2:4]}' out of range (01-12)"
            elif not (1 <= dd <= 31):
                exp_reason = f"Invalid expiry date: day '{expiry_raw[4:6]}' out of range (01-31)"
            else:
                valid_exp_dates = True
        except ValueError:
            exp_reason = f"Invalid expiry date format: '{expiry_raw}'"
    if not exp_reason and not valid_expiry_cd:
        exp_reason = "Expiry date check digit validation failed"

    # Nationality & Issuing State
    valid_nationality = (len(nationality) == 3 and nationality.isalpha() and nationality in ICAO_COUNTRY_CODES)
    valid_issuing_state = (len(issuing_state) == 3 and issuing_state.isalpha() and issuing_state in ICAO_COUNTRY_CODES)
    valid_sex = (raw_sex in ('M', 'F', 'X', '<'))

    check_digit_results = {
        "document_number": valid_visa_cd,
        "passport_number": valid_visa_cd,
        "date_of_birth": valid_dob_cd,
        "date_of_expiry": valid_expiry_cd,
        "composite": None,
    }

    field_validation = {
        "document_number": {
            "valid": valid_visa_cd,
            "value": visa_num,
            "reason": None if valid_visa_cd else "Visa document number check digit validation failed"
        },
        "issuing_state": {
            "valid": valid_issuing_state,
            "value": issuing_state,
            "reason": None if valid_issuing_state else f"Invalid issuing state code: '{issuing_state}'"
        },
        "nationality": {
            "valid": valid_nationality,
            "value": nationality,
            "reason": None if valid_nationality else f"Invalid nationality code: '{nationality}'"
        },
        "date_of_birth": {
            "valid": (valid_dob_cd and valid_dob_dates and not dob_has_non_digits),
            "value": dob_raw,
            "reason": dob_reason
        },
        "sex": {
            "valid": valid_sex,
            "value": sex if sex != '<' else "",
            "reason": None if valid_sex else f"Invalid sex character: '{raw_sex}' (expected M, F, or <)"
        },
        "date_of_expiry": {
            "valid": (valid_expiry_cd and valid_exp_dates and not exp_has_non_digits),
            "value": expiry_raw,
            "reason": exp_reason
        }
    }

    overall_valid = bool(
        valid_visa_cd and
        valid_dob_cd and
        valid_dob_dates and
        not dob_has_non_digits and
        valid_expiry_cd and
        valid_exp_dates and
        not exp_has_non_digits and
        valid_nationality and
        valid_issuing_state and
        valid_sex
    )

    return {
        "valid_format": True,
        "format": "MRVA",
        "raw_line1": raw_l1,
        "raw_line2": raw_l2,
        "line1": corr_l1,
        "line2": corr_l2,
        "corrections": corrections,
        "document_code": doc_code,
        "issuing_state": issuing_state,
        "fields": {
            "surname": surname,
            "given_names": given_names,
            "document_number": visa_num,
            "visa_number": visa_num,
            "nationality": nationality,
            "date_of_birth": dob_raw,
            "sex": sex if sex != '<' else "",
            "date_of_expiry": expiry_raw,
            "expiry_date": expiry_raw,
            "optional_data": optional_data,
            "issuing_state": issuing_state,
            "document_code": doc_code,
        },
        "check_digits": check_digit_results,
        "field_validation": field_validation,
        "overall_valid": overall_valid,
    }


def parse_mrvb_mrz(line1: str, line2: str) -> Dict:
    """Parses ICAO Doc 9303 MRV-B Short Visa MRZ (2 lines of 36 characters).
    
    Structure:
    Line 1 (36 characters):
      - 00..01 (2 chars): Document code (V, V<, VB, etc.)
      - 02..04 (3 chars): Issuing state (3-letter ICAO code)
      - 05..35 (31 chars): Name (Surname << Given names)
      
    Line 2 (36 characters):
      - 00..08 (9 chars): Visa / Document number
      - 09 (1 char): Visa number check digit
      - 10..12 (3 chars): Nationality (3-letter ICAO code)
      - 13..18 (6 chars): Date of birth (YYMMDD)
      - 19 (1 char): Date of birth check digit
      - 20 (1 char): Sex (M, F, <)
      - 21..26 (6 chars): Expiry date (YYMMDD)
      - 27 (1 char): Expiry date check digit
      - 28..35 (8 chars): Optional data elements
    """
    raw_l1 = normalize_mrz_line(line1)
    raw_l2 = normalize_mrz_line(line2)

    if len(raw_l1) != 36 or len(raw_l2) != 36:
        return {
            "valid_format": False,
            "format": "MRVB",
            "error": f"Invalid MRV-B line lengths: line1={len(raw_l1)}, line2={len(raw_l2)} (expected 36 each)",
            "raw_line1": raw_l1,
            "raw_line2": raw_l2,
            "line1": raw_l1,
            "line2": raw_l2,
            "corrections": [],
            "fields": {},
            "check_digits": None,
            "field_validation": None,
            "overall_valid": False,
        }

    corr_l1, corr_l2, corrections = correct_mrv_mrz(raw_l1, raw_l2, format_type="MRVB")

    # Line 1 fields
    doc_code = corr_l1[0:2].replace('<', '')
    issuing_state = corr_l1[2:5].replace('<', '')
    name_raw = corr_l1[5:36].lstrip('<')
    if '<<' in name_raw:
        parts = name_raw.split('<<', 1)
        surname = _clean_mrz_name_component(parts[0])
        given_names = _clean_mrz_name_component(parts[1])
    else:
        surname = _clean_mrz_name_component(name_raw)
        given_names = ""


    # Line 2 fields
    visa_num = corr_l2[0:9].replace('<', '')
    visa_cd = corr_l2[9:10]
    nationality = corr_l2[10:13].replace('<', '')
    dob_raw = corr_l2[13:19]
    dob_cd = corr_l2[19:20]
    raw_sex = corr_l2[20:21]
    sex = raw_sex if raw_sex in ('M', 'F', 'X') else '<'
    expiry_raw = corr_l2[21:27]
    expiry_cd = corr_l2[27:28]
    optional_data = corr_l2[28:36].replace('<', '')

    # Validations
    valid_visa_cd = validate_check_digit(corr_l2[0:9], visa_cd)

    # DOB validation
    valid_dob_cd = validate_check_digit(dob_raw, dob_cd)
    dob_has_non_digits = (len(dob_raw) != 6 or not dob_raw.isdigit())
    valid_dob_dates = False
    dob_reason = None
    if dob_has_non_digits:
        dob_reason = f"Date of birth contains non-numeric characters: '{dob_raw}'"
    else:
        try:
            mm = int(dob_raw[2:4])
            dd = int(dob_raw[4:6])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                valid_dob_dates = True
            else:
                dob_reason = f"Invalid DOB date values: month={dob_raw[2:4]}, day={dob_raw[4:6]}"
        except ValueError:
            dob_reason = f"Invalid DOB date format: '{dob_raw}'"
    if not dob_reason and not valid_dob_cd:
        dob_reason = "DOB check digit validation failed"

    # Expiry validation
    valid_expiry_cd = validate_check_digit(expiry_raw, expiry_cd)
    exp_has_non_digits = (len(expiry_raw) != 6 or not expiry_raw.isdigit())
    valid_exp_dates = False
    exp_reason = None
    if exp_has_non_digits:
        exp_reason = f"Expiry date contains non-numeric characters: '{expiry_raw}'"
    else:
        try:
            mm = int(expiry_raw[2:4])
            dd = int(expiry_raw[4:6])
            if not (1 <= mm <= 12):
                exp_reason = f"Invalid expiry date: month '{expiry_raw[2:4]}' out of range (01-12)"
            elif not (1 <= dd <= 31):
                exp_reason = f"Invalid expiry date: day '{expiry_raw[4:6]}' out of range (01-31)"
            else:
                valid_exp_dates = True
        except ValueError:
            exp_reason = f"Invalid expiry date format: '{expiry_raw}'"
    if not exp_reason and not valid_expiry_cd:
        exp_reason = "Expiry date check digit validation failed"

    # Nationality & Issuing State
    valid_nationality = (len(nationality) == 3 and nationality.isalpha() and nationality in ICAO_COUNTRY_CODES)
    valid_issuing_state = (len(issuing_state) == 3 and issuing_state.isalpha() and issuing_state in ICAO_COUNTRY_CODES)
    valid_sex = (raw_sex in ('M', 'F', 'X', '<'))

    check_digit_results = {
        "document_number": valid_visa_cd,
        "passport_number": valid_visa_cd,
        "date_of_birth": valid_dob_cd,
        "date_of_expiry": valid_expiry_cd,
        "composite": None,
    }

    field_validation = {
        "document_number": {
            "valid": valid_visa_cd,
            "value": visa_num,
            "reason": None if valid_visa_cd else "Visa document number check digit validation failed"
        },
        "issuing_state": {
            "valid": valid_issuing_state,
            "value": issuing_state,
            "reason": None if valid_issuing_state else f"Invalid issuing state code: '{issuing_state}'"
        },
        "nationality": {
            "valid": valid_nationality,
            "value": nationality,
            "reason": None if valid_nationality else f"Invalid nationality code: '{nationality}'"
        },
        "date_of_birth": {
            "valid": (valid_dob_cd and valid_dob_dates and not dob_has_non_digits),
            "value": dob_raw,
            "reason": dob_reason
        },
        "sex": {
            "valid": valid_sex,
            "value": sex if sex != '<' else "",
            "reason": None if valid_sex else f"Invalid sex character: '{raw_sex}' (expected M, F, or <)"
        },
        "date_of_expiry": {
            "valid": (valid_expiry_cd and valid_exp_dates and not exp_has_non_digits),
            "value": expiry_raw,
            "reason": exp_reason
        }
    }

    overall_valid = bool(
        valid_visa_cd and
        valid_dob_cd and
        valid_dob_dates and
        not dob_has_non_digits and
        valid_expiry_cd and
        valid_exp_dates and
        not exp_has_non_digits and
        valid_nationality and
        valid_issuing_state and
        valid_sex
    )

    return {
        "valid_format": True,
        "format": "MRVB",
        "raw_line1": raw_l1,
        "raw_line2": raw_l2,
        "line1": corr_l1,
        "line2": corr_l2,
        "corrections": corrections,
        "document_code": doc_code,
        "issuing_state": issuing_state,
        "fields": {
            "surname": surname,
            "given_names": given_names,
            "document_number": visa_num,
            "visa_number": visa_num,
            "nationality": nationality,
            "date_of_birth": dob_raw,
            "sex": sex if sex != '<' else "",
            "date_of_expiry": expiry_raw,
            "expiry_date": expiry_raw,
            "optional_data": optional_data,
            "issuing_state": issuing_state,
            "document_code": doc_code,
        },
        "check_digits": check_digit_results,
        "field_validation": field_validation,
        "overall_valid": overall_valid,
    }



def parse_td1_mrz(line1: str, line2: str, line3: str) -> Dict:
    """Parses ICAO Doc 9303 TD1 National ID MRZ (3 lines of 30 characters)."""
    raw_l1 = normalize_mrz_line(line1)
    raw_l2 = normalize_mrz_line(line2)
    raw_l3 = normalize_mrz_line(line3)

    if len(raw_l1) != 30 or len(raw_l2) != 30 or len(raw_l3) != 30:
        return {
            "valid_format": False,
            "format": "TD1",
            "error": f"Invalid TD1 line lengths: line1={len(raw_l1)}, line2={len(raw_l2)}, line3={len(raw_l3)} (expected 30 each)",
            "raw_line1": raw_l1,
            "raw_line2": raw_l2,
            "raw_line3": raw_l3,
            "line1": raw_l1,
            "line2": raw_l2,
            "line3": raw_l3,
            "corrections": [],
            "fields": {},
            "check_digits": None,
            "overall_valid": False,
        }

    # Line 1: Doc Code (0..1), Issuing State (2..4), Doc Number (5..13), Check Digit (14), Optional 1 (15..29)
    doc_code = raw_l1[0:2].replace('<', '')
    raw_state = raw_l1[2:5]
    corr_state, state_corrs = _correct_country_code(raw_state, line_num=1, base_pos=2, field_name="issuing_state")
    doc_num = raw_l1[5:14].replace('<', '')
    doc_cd = raw_l1[14:15]
    optional_1 = raw_l1[15:30].replace('<', '')

    # Line 2: DOB (0..5), DOB CD (6), Sex (7), Expiry (8..13), Expiry CD (14), Nationality (15..17), Optional 2 (18..28), Composite CD (29)
    dob_raw = raw_l2[0:6]
    dob_cd = raw_l2[6:7]
    sex = raw_l2[7:8] if raw_l2[7:8] in ('M', 'F', 'X') else '<'
    expiry_raw = raw_l2[8:14]
    expiry_cd = raw_l2[14:15]
    raw_nat = raw_l2[15:18]
    corr_nat, nat_corrs = _correct_country_code(raw_nat, line_num=2, base_pos=15, field_name="nationality")
    optional_2 = raw_l2[18:29].replace('<', '')
    composite_cd = raw_l2[29:30]

    # Line 3: Surname << Given names
    name_raw = raw_l3.lstrip('<')
    if '<<' in name_raw:
        parts = name_raw.split('<<', 1)
        surname = parts[0].replace('<', ' ').strip()
        given_names = parts[1].replace('<', ' ').strip()
    else:
        surname = name_raw.replace('<', ' ').strip()
        given_names = ""

    # Check Digits
    valid_doc_cd = validate_check_digit(raw_l1[5:14], doc_cd)
    valid_dob_cd = validate_check_digit(dob_raw, dob_cd)
    valid_dob_dates = False
    try:
        valid_dob_dates = (1 <= int(dob_raw[2:4]) <= 12 and 1 <= int(dob_raw[4:6]) <= 31)
    except (ValueError, IndexError):
        pass

    valid_expiry_cd = validate_check_digit(expiry_raw, expiry_cd)
    valid_exp_dates = False
    try:
        valid_exp_dates = (1 <= int(expiry_raw[2:4]) <= 12 and 1 <= int(expiry_raw[4:6]) <= 31)
    except (ValueError, IndexError):
        pass

    # TD1 composite: L1[5:30] + L2[0:7] + L2[8:15] + L2[18:29]
    composite_payload = raw_l1[5:30] + raw_l2[0:7] + raw_l2[8:15] + raw_l2[18:29]
    valid_composite_cd = validate_check_digit(composite_payload, composite_cd)

    valid_nationality = (len(corr_nat) == 3 and corr_nat in ICAO_COUNTRY_CODES)
    valid_issuing_state = (len(corr_state) == 3 and corr_state in ICAO_COUNTRY_CODES)
    valid_sex = (sex in ('M', 'F', 'X'))

    check_digit_results = {
        "document_number": valid_doc_cd,
        "passport_number": valid_doc_cd,
        "date_of_birth": valid_dob_cd,
        "date_of_expiry": valid_expiry_cd,
        "composite": valid_composite_cd,
    }

    field_validation = {
        "document_number": {
            "valid": valid_doc_cd,
            "value": doc_num,
            "reason": None if valid_doc_cd else "National ID document number check digit validation failed"
        },
        "issuing_state": {
            "valid": valid_issuing_state,
            "value": corr_state,
            "reason": None if valid_issuing_state else f"Invalid issuing state code: '{corr_state}'"
        },
        "nationality": {
            "valid": valid_nationality,
            "value": corr_nat,
            "reason": None if valid_nationality else f"Invalid nationality code: '{corr_nat}'"
        },
        "date_of_birth": {
            "valid": (valid_dob_cd and valid_dob_dates),
            "value": dob_raw,
            "reason": None if (valid_dob_cd and valid_dob_dates) else "DOB check digit validation failed"
        },
        "sex": {
            "valid": valid_sex,
            "value": sex if sex != '<' else "",
            "reason": None if valid_sex else "Missing or invalid MRZ sex value"
        },
        "date_of_expiry": {
            "valid": (valid_expiry_cd and valid_exp_dates),
            "value": expiry_raw,
            "reason": None if (valid_expiry_cd and valid_exp_dates) else "Expiry date check digit validation failed"
        },
        "composite": {
            "valid": valid_composite_cd,
            "value": composite_cd,
            "reason": None if valid_composite_cd else "Composite check digit validation failed"
        }
    }

    overall_valid = (
        valid_doc_cd and
        valid_dob_cd and
        valid_expiry_cd and
        valid_composite_cd and
        valid_nationality and
        valid_issuing_state
    )

    return {
        "valid_format": True,
        "format": "TD1",
        "raw_line1": raw_l1,
        "raw_line2": raw_l2,
        "raw_line3": raw_l3,
        "line1": raw_l1,
        "line2": raw_l2,
        "line3": raw_l3,
        "corrections": state_corrs + nat_corrs,
        "document_code": doc_code,
        "issuing_state": corr_state,
        "fields": {
            "surname": surname,
            "given_names": given_names,
            "document_number": doc_num,
            "id_number": doc_num,
            "nationality": corr_nat,
            "date_of_birth": dob_raw,
            "sex": sex if sex != '<' else "",
            "gender": sex if sex != '<' else "",
            "date_of_expiry": expiry_raw,
            "optional_1": optional_1,
            "optional_2": optional_2,
            "issuing_state": corr_state,
            "document_code": doc_code,
        },
        "check_digits": check_digit_results,
        "field_validation": field_validation,
        "overall_valid": overall_valid,
    }


def parse_td2_mrz(line1: str, line2: str) -> Dict:
    """Parses ICAO Doc 9303 TD2 National ID MRZ (2 lines of 36 characters)."""
    raw_l1 = normalize_mrz_line(line1)
    raw_l2 = normalize_mrz_line(line2)

    if len(raw_l1) != 36 or len(raw_l2) != 36:
        return {
            "valid_format": False,
            "format": "TD2",
            "error": f"Invalid TD2 line lengths: line1={len(raw_l1)}, line2={len(raw_l2)} (expected 36 each)",
            "raw_line1": raw_l1,
            "raw_line2": raw_l2,
            "line1": raw_l1,
            "line2": raw_l2,
            "corrections": [],
            "fields": {},
            "check_digits": None,
            "overall_valid": False,
        }

    corr_l1, corr_l2, corrections = correct_td3_mrz(raw_l1.ljust(44, '<'), raw_l2.ljust(44, '<'))
    corr_l1 = corr_l1[:36]
    corr_l2 = corr_l2[:36]

    doc_code = corr_l1[0:2].replace('<', '')
    issuing_state = corr_l1[2:5].replace('<', '')
    name_raw = corr_l1[5:36].lstrip('<')
    if '<<' in name_raw:
        parts = name_raw.split('<<', 1)
        surname = parts[0].replace('<', ' ').strip()
        given_names = parts[1].replace('<', ' ').strip()
    else:
        surname = name_raw.replace('<', ' ').strip()
        given_names = ""

    doc_num = corr_l2[0:9].replace('<', '')
    doc_cd = corr_l2[9:10]
    nationality = corr_l2[10:13].replace('<', '')
    dob_raw = corr_l2[13:19]
    dob_cd = corr_l2[19:20]
    sex = corr_l2[20:21] if corr_l2[20:21] in ('M', 'F', 'X') else '<'
    expiry_raw = corr_l2[21:27]
    expiry_cd = corr_l2[27:28]
    optional_data = corr_l2[28:35].replace('<', '')
    composite_cd = corr_l2[35:36]

    valid_doc_cd = validate_check_digit(corr_l2[0:9], doc_cd)
    valid_dob_cd = validate_check_digit(dob_raw, dob_cd)
    valid_dob_dates = False
    try:
        valid_dob_dates = (1 <= int(dob_raw[2:4]) <= 12 and 1 <= int(dob_raw[4:6]) <= 31)
    except (ValueError, IndexError):
        pass

    valid_expiry_cd = validate_check_digit(expiry_raw, expiry_cd)
    valid_exp_dates = False
    try:
        valid_exp_dates = (1 <= int(expiry_raw[2:4]) <= 12 and 1 <= int(expiry_raw[4:6]) <= 31)
    except (ValueError, IndexError):
        pass

    # TD2 composite: Line 2 positions [0:10] + [13:20] + [21:35]
    composite_payload = corr_l2[0:10] + corr_l2[13:20] + corr_l2[21:35]
    valid_composite_cd = validate_check_digit(composite_payload, composite_cd)

    valid_nationality = (len(nationality) == 3 and nationality in ICAO_COUNTRY_CODES)
    valid_issuing_state = (len(issuing_state) == 3 and issuing_state in ICAO_COUNTRY_CODES)
    valid_sex = (sex in ('M', 'F', 'X'))

    check_digit_results = {
        "document_number": valid_doc_cd,
        "passport_number": valid_doc_cd,
        "date_of_birth": valid_dob_cd,
        "date_of_expiry": valid_expiry_cd,
        "composite": valid_composite_cd,
    }

    field_validation = {
        "document_number": {
            "valid": valid_doc_cd,
            "value": doc_num,
            "reason": None if valid_doc_cd else "TD2 document number check digit validation failed"
        },
        "issuing_state": {
            "valid": valid_issuing_state,
            "value": issuing_state,
            "reason": None if valid_issuing_state else f"Invalid issuing state code: '{issuing_state}'"
        },
        "nationality": {
            "valid": valid_nationality,
            "value": nationality,
            "reason": None if valid_nationality else f"Invalid nationality code: '{nationality}'"
        },
        "date_of_birth": {
            "valid": (valid_dob_cd and valid_dob_dates),
            "value": dob_raw,
            "reason": None if (valid_dob_cd and valid_dob_dates) else "DOB check digit validation failed"
        },
        "sex": {
            "valid": valid_sex,
            "value": sex if sex != '<' else "",
            "reason": None if valid_sex else "Missing or invalid MRZ sex value"
        },
        "date_of_expiry": {
            "valid": (valid_expiry_cd and valid_exp_dates),
            "value": expiry_raw,
            "reason": None if (valid_expiry_cd and valid_exp_dates) else "Expiry date check digit validation failed"
        },
        "composite": {
            "valid": valid_composite_cd,
            "value": composite_cd,
            "reason": None if valid_composite_cd else "Composite check digit validation failed"
        }
    }

    overall_valid = (
        valid_doc_cd and
        valid_dob_cd and
        valid_expiry_cd and
        valid_composite_cd and
        valid_nationality and
        valid_issuing_state
    )

    return {
        "valid_format": True,
        "format": "TD2",
        "raw_line1": raw_l1,
        "raw_line2": raw_l2,
        "line1": corr_l1,
        "line2": corr_l2,
        "corrections": corrections,
        "document_code": doc_code,
        "issuing_state": issuing_state,
        "fields": {
            "surname": surname,
            "given_names": given_names,
            "document_number": doc_num,
            "id_number": doc_num,
            "nationality": nationality,
            "date_of_birth": dob_raw,
            "sex": sex if sex != '<' else "",
            "gender": sex if sex != '<' else "",
            "date_of_expiry": expiry_raw,
            "optional_data": optional_data,
            "issuing_state": issuing_state,
            "document_code": doc_code,
        },
        "check_digits": check_digit_results,
        "field_validation": field_validation,
        "overall_valid": overall_valid,
    }



