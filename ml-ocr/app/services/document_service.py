"""Document classification and extensible field extraction service for Passports, Visas, and National IDs."""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.config import settings
from app.models.schemas import DocumentTypeEnum, MRZResult, OCRRegion
from app.services.image_service import ImageService
from app.utils.logger import get_logger




logger = get_logger(__name__)

# Common government & document header tokens that must never be treated as personal names
HEADER_BLACKLIST = {
    "GOVERNMENT", "GOVT", "INDIA", "BHARAT", "REPUBLIC", "STATE", "UNION",
    "AUTHORITY", "UNIQUE", "IDENTIFICATION", "ELECTION", "COMMISSION",
    "INCOME", "TAX", "DEPARTMENT", "TRANSPORT", "MINISTRY", "DIRECTORATE",
    "IDENTITY", "CARD", "NATIONAL", "AADHAAR", "AADHAR", "MERA",
    "PEHCHAN", "ENROLMENT", "ENROLLMENT", "RESIDENT", "CITIZEN",
    "DRIVING", "LICENCE", "LICENSE", "PERMANENT", "ACCOUNT",
    "NUMBER", "VOTER", "ELECTOR", "PHOTO", "SIGNATURE", "OFFICIAL",
    "HELP", "LINE", "WWW", "GOV", "IN", "NIC", "DOWNLOAD", "DATE",
    "MALE", "FEMALE", "GENDER", "SEX", "DOB", "YEAR", "BIRTH", "ADDRESS",
    "ISSUE", "EXPIRY", "VALID", "FATHER", "MOTHER", "HUSBAND", "WIFE"
}


class BaseDocumentExtractor(ABC):
    """Abstract base extractor for specific document types."""

    @property
    @abstractmethod
    def document_type(self) -> str:
        pass

    @abstractmethod
    def extract_fields(
        self,
        ocr_text: str,
        mrz_fields: Dict[str, Any],
        mrz_format: Optional[str] = None,
        ocr_regions: Optional[List[OCRRegion]] = None,
        document_image: Optional[np.ndarray] = None,
        ocr_service: Optional[Any] = None,
        include_debug: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extracts structured key-value fields and per-field confidences from OCR text, regions, and MRZ data."""
        pass

    @staticmethod
    def _find_region_confidence(val: str, regions: List[OCRRegion]) -> float:
        """Finds confidence score for matching text substring across OCR regions."""
        if not val or not regions:
            return 0.85
        val_clean = val.strip().lower()
        for r in regions:
            if val_clean in r.text.lower():
                return r.confidence
        return 0.85

    @classmethod
    def _extract_spatial_value(
        cls,
        regions: List[OCRRegion],
        label_patterns: List[str],
        val_pattern: str
    ) -> Tuple[Optional[str], float]:
        """Searches for a field value spatially located to the right or immediately below a detected label region."""
        if not regions:
            return None, 0.0

        for i, label_reg in enumerate(regions):
            label_text = label_reg.text.strip()
            is_label = any(re.search(pat, label_text, re.IGNORECASE) for pat in label_patterns)
            if not is_label:
                continue

            # 1. Check if the value is already inside the same region after the label
            inline_match = re.search(val_pattern, label_text)
            if inline_match and inline_match.group(1).strip() != label_text:
                return inline_match.group(1).strip(), label_reg.confidence

            lx1, ly1, lx2, ly2 = label_reg.bbox
            l_height = ly2 - ly1
            l_center_y = (ly1 + ly2) / 2.0

            # 2. Check candidates on the same line to the right
            same_line_cands: List[Tuple[float, str, float]] = []
            for other_reg in regions:
                if other_reg == label_reg:
                    continue
                ox1, oy1, ox2, oy2 = other_reg.bbox
                o_center_y = (oy1 + oy2) / 2.0
                if abs(o_center_y - l_center_y) <= max(18.0, l_height * 0.9):
                    if ox1 >= lx1 + 5:
                        cand_match = re.search(val_pattern, other_reg.text.strip())
                        if cand_match:
                            h_dist = ox1 - lx2
                            same_line_cands.append((h_dist, cand_match.group(1).strip(), other_reg.confidence))

            if same_line_cands:
                same_line_cands.sort(key=lambda item: item[0])
                return same_line_cands[0][1], same_line_cands[0][2]

            # 3. Check candidates in the region immediately below the label
            below_cands: List[Tuple[float, str, float]] = []
            for other_reg in regions:
                if other_reg == label_reg:
                    continue
                ox1, oy1, ox2, oy2 = other_reg.bbox
                if oy1 >= ly1 and (oy1 - ly2) <= max(35.0, l_height * 2.0):
                    if ox1 <= lx2 + 80 and ox2 >= lx1 - 40:
                        cand_match = re.search(val_pattern, other_reg.text.strip())
                        if cand_match:
                            v_dist = oy1 - ly2
                            below_cands.append((v_dist, cand_match.group(1).strip(), other_reg.confidence))

            if below_cands:
                below_cands.sort(key=lambda item: item[0])
                return below_cands[0][1], below_cands[0][2]

        return None, 0.0


class PassportExtractor(BaseDocumentExtractor):
    """Structured field extractor for Passports."""

    def __init__(self):
        self.last_field_sources: Dict[str, Dict[str, Any]] = {}
        self.last_field_debug: Optional[Dict[str, Any]] = None
        self.last_warnings: List[str] = []

    @property
    def document_type(self) -> str:
        return DocumentTypeEnum.PASSPORT.value

    def extract_fields(
        self,
        ocr_text: str,
        mrz_fields: Dict[str, Any],
        mrz_format: Optional[str] = None,
        ocr_regions: Optional[List[OCRRegion]] = None,
        document_image: Optional[np.ndarray] = None,
        ocr_service: Optional[Any] = None,
        include_debug: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.last_field_sources = {}
        self.last_warnings = []
        self.last_field_debug = None

        fields: Dict[str, Any] = {}
        confidences: Dict[str, float] = {}
        
        # Primary source: MRZ fields
        if mrz_fields:
            fields["surname"] = mrz_fields.get("surname", "")
            fields["given_names"] = mrz_fields.get("given_names", "")
            fields["passport_number"] = mrz_fields.get("passport_number", "")
            fields["nationality"] = mrz_fields.get("nationality", "")
            fields["date_of_birth"] = mrz_fields.get("date_of_birth", "")
            fields["sex"] = mrz_fields.get("sex", "")
            fields["date_of_expiry"] = mrz_fields.get("date_of_expiry", "")
            if mrz_fields.get("issuing_state"):
                fields["issuing_state"] = mrz_fields.get("issuing_state")
            if mrz_fields.get("personal_number"):
                fields["personal_number"] = mrz_fields.get("personal_number")
            if mrz_format:
                fields["mrz_format"] = mrz_format

            for k, v in fields.items():
                if v:
                    confidences[k] = 0.98
                    self.last_field_sources[k] = {"value": v, "source": "mrz", "confidence": 0.98}

            return fields, confidences

        # Fallback regex extraction from visual text if MRZ was absent/damaged
        passport_num_match = re.search(
            r'(?:Passport No|Passport Number|Document No|Passport/Passeport|Passeport No|Passport\s*#)[:\s]+([A-Z0-9]{8,9})',
            ocr_text,
            re.IGNORECASE
        )
        fields["passport_number"] = passport_num_match.group(1) if passport_num_match else ""
            
        dob_match = re.search(r'(?:DOB|Date of Birth|Birth Date)[:\s]+([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{4}[/-][0-9]{2}[/-][0-9]{2})', ocr_text, re.IGNORECASE)
        if dob_match:
            fields["date_of_birth"] = dob_match.group(1)

        expiry_match = re.search(r'(?:Date of Expiry|Expiry Date|Expires)[:\s]+([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{4}[/-][0-9]{2}[/-][0-9]{2})', ocr_text, re.IGNORECASE)
        if expiry_match:
            fields["date_of_expiry"] = expiry_match.group(1)

        sex_match = re.search(r'(?:Sex|Gender)[:\s]+([MFX]|MALE|FEMALE)', ocr_text, re.IGNORECASE)
        if sex_match:
            val = sex_match.group(1).upper()
            fields["sex"] = "M" if "M" in val else ("F" if "F" in val else "X")

        for k, v in fields.items():
            if v:
                confidences[k] = 0.85
                self.last_field_sources[k] = {"value": v, "source": "visual_ocr", "confidence": 0.85}

        return fields, confidences


class VisaExtractor(BaseDocumentExtractor):
    """Structured dual-source field extractor for Visas (Visual Zone OCR + MRV MRZ OCR)."""

    def __init__(self, min_confidence: float = 0.50):
        self.min_confidence = min_confidence
        self.last_field_sources: Dict[str, Dict[str, Any]] = {}
        self.last_field_debug: Optional[Dict[str, Any]] = None
        self.last_warnings: List[str] = []

    @property
    def document_type(self) -> str:
        return DocumentTypeEnum.VISA.value

    @classmethod
    def _normalize_entries(cls, raw: str) -> Optional[str]:
        """Normalizes visa entries token into standard controlled values: SINGLE, DOUBLE, MULTIPLE, 1, 2, M, S, D."""
        if not raw:
            return None
        clean = raw.strip().upper()
        if clean in ("MULTIPLE", "MULT", "M", "MULTIPLE/MULTIPLE", "M/M"):
            return "MULTIPLE"
        elif clean in ("SINGLE", "S", "ONE", "1"):
            return "SINGLE"
        elif clean in ("DOUBLE", "D", "TWO", "2"):
            return "DOUBLE"
        elif clean.isdigit():
            return clean
        elif len(clean) <= 10:
            return clean
        return None

    @classmethod
    def _is_valid_date(cls, date_str: str) -> bool:
        """Validates that a date string is a plausible calendar date."""
        if not date_str:
            return False
        clean = date_str.strip()
        m = re.match(r'^([0-9]{2})[/.-]([0-9]{2})[/.-]([0-9]{4})$', clean)
        if m:
            d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return 1 <= d <= 31 and 1 <= mth <= 12 and 1900 <= y <= 2060
        m_iso = re.match(r'^([0-9]{4})[/.-]([0-9]{2})[/.-]([0-9]{2})$', clean)
        if m_iso:
            y, mth, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            return 1 <= d <= 31 and 1 <= mth <= 12 and 1900 <= y <= 2060
        m_txt = re.match(r'^([0-9]{1,2})\s+([A-Za-z]{3,9})\s+([0-9]{4})$', clean)
        if m_txt:
            d, y = int(m_txt.group(1)), int(m_txt.group(3))
            return 1 <= d <= 31 and 1900 <= y <= 2060
        return False

    @classmethod
    def _try_second_pass_crop_date(
        cls,
        document_image: np.ndarray,
        ocr_service: Any,
        sorted_regions: List[OCRRegion],
        label_keywords: List[str],
        target_field: str,
        vis_fields: Dict[str, Any],
        vis_confs: Dict[str, float]
    ):
        """Generates dynamic ROI crops around a date label and runs targeted numeric OCR to recover missed dates."""
        label_regs = []
        for r in sorted_regions:
            if any(re.search(kw, r.text, re.IGNORECASE) for kw in label_keywords):
                label_regs.append(r)

        for l_reg in label_regs:
            variants = ImageService.get_field_crop_variants(
                document_image,
                label_bbox=l_reg.bbox,
                field_type="dob",
                scale_factor=3.0
            )
            for crop_name, prep_name, prep_img, crop_bbox in variants:
                for psm in [7, 8, 6]:
                    try:
                        crop_res = ocr_service.extract_field(prep_img, field_type="dob", psm=psm)
                        cand_text = crop_res.raw_text.strip()
                        if not cand_text:
                            continue
                        date_matches = re.findall(
                            r'([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})',
                            cand_text
                        )
                        for dm in date_matches:
                            dm_clean = dm.strip()
                            if cls._is_valid_date(dm_clean):
                                vis_fields[target_field] = dm_clean
                                vis_confs[target_field] = crop_res.average_confidence
                                return
                    except Exception as e:
                        logger.debug(f"Targeted date crop failed for {target_field}: {str(e)}")

    def extract_fields(
        self,
        ocr_text: str,
        mrz_fields: Dict[str, Any],
        mrz_format: Optional[str] = None,
        ocr_regions: Optional[List[OCRRegion]] = None,
        document_image: Optional[np.ndarray] = None,
        ocr_service: Optional[Any] = None,
        include_debug: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.last_field_sources = {}
        self.last_field_debug = {"visual_candidates": [], "discrepancies": []}
        self.last_warnings = []

        fields: Dict[str, Any] = {
            "visa_number": "",
            "document_number": "",
            "name": "",
            "surname": "",
            "given_names": "",
            "nationality": "",
            "date_of_birth": "",
            "sex": "",
            "passport_number": "",
            "visa_type": "",
            "issue_date": "",
            "expiry_date": "",
            "entries": "",
            "issuing_authority": "",
            "issuing_state": ""
        }
        confidences: Dict[str, float] = {}

        sorted_regions = sorted(ocr_regions, key=lambda r: (r.bbox[1], r.bbox[0])) if ocr_regions else []

        # =========================================================
        # PHASE 1: Visual-Zone OCR Extraction (Main Visa Body)
        # =========================================================
        vis_fields: Dict[str, Any] = {}
        vis_confs: Dict[str, float] = {}

        # 1. Visa Type / Class
        vt_patterns = [
            r'(?:Visa\s*Type\s*/\s*Class|Type\s*/\s*Class|Visa\s*Type|Type\s*de\s*visa|Visa\s*Category|Category|Class)\s*[:\s]\s*([A-Za-z0-9/_-]+)',
            r'\b(?:Type|Class)\s*:\s*([A-Za-z0-9/_-]+)'
        ]
        for pat in vt_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                cand_vt = m.group(1).strip().upper()
                if cand_vt and cand_vt not in ("/", "-", "_", "CLASS", "CATEGORY", "PASSPORT", "VISA", "NUMBER", "DATE", "CONTROL"):
                    vis_fields["visa_type"] = cand_vt
                    vis_confs["visa_type"] = self._find_region_confidence(cand_vt, sorted_regions)
                    break
        if "visa_type" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Visa\s*Type\s*/\s*Class|Type\s*/\s*Class|Visa\s*Type|Class|Category)\b'],
                val_pattern=r'([A-Za-z0-9/_-]{1,15})'
            )
            if val and val.upper() not in ("/", "-", "_", "CLASS", "CATEGORY", "PASSPORT", "VISA", "NUMBER", "DATE", "CONTROL"):
                vis_fields["visa_type"] = val.upper()
                vis_confs["visa_type"] = conf


        # 2. Number of Entries
        ent_patterns = [
            r'(?:Entries|No\s*of\s*Entries|Number\s*of\s*Entries|Nombre\s*d\'\s*entrées|Entries\s*/\s*Entrées)[:\s]+(MULTIPLE|SINGLE|DOUBLE|MULT|\bM\b|\bS\b|\bD\b|\b[12]\b)',
            r'\bEntries[:\s]+([A-Za-z0-9]+)'
        ]
        for pat in ent_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                raw_ent = m.group(1).strip().upper()
                norm_ent = self._normalize_entries(raw_ent)
                if norm_ent:
                    vis_fields["entries"] = norm_ent
                    vis_confs["entries"] = self._find_region_confidence(m.group(1), sorted_regions)
                    break
        if "entries" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Entries|No\s*of\s*Entries|Number\s*of\s*Entries)\b'],
                val_pattern=r'([A-Za-z0-9]+)'
            )
            if val:
                norm_ent = self._normalize_entries(val)
                if norm_ent:
                    vis_fields["entries"] = norm_ent
                    vis_confs["entries"] = conf

        # 3. Date of Issue (Issue Date)
        iss_patterns = [
            r'(?:Issue\s*Date|Date\s*of\s*Issue|Issued|Date\s*d\'\s*émission|Issued\s*On)[:\s]+([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2})',
        ]
        for pat in iss_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                cand_iss = m.group(1).strip()
                if self._is_valid_date(cand_iss):
                    vis_fields["issue_date"] = cand_iss
                    vis_confs["issue_date"] = self._find_region_confidence(cand_iss, sorted_regions)
                    break
        if "issue_date" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Issue\s*Date|Date\s*of\s*Issue|Issued)\b'],
                val_pattern=r'([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2})'
            )
            if val and self._is_valid_date(val):
                vis_fields["issue_date"] = val.strip()
                vis_confs["issue_date"] = conf

        # 4. Date of Expiry (Visual Zone)
        exp_patterns = [
            r'(?:Expiry\s*Date|Date\s*of\s*Expiry|Expiration\s*Date|Expires\s*On|Valid\s*Until|Date\s*d\'\s*expiration|Expires)[:\s]+([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2})',
        ]
        for pat in exp_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                cand_exp = m.group(1).strip()
                if self._is_valid_date(cand_exp):
                    vis_fields["expiry_date"] = cand_exp
                    vis_confs["expiry_date"] = self._find_region_confidence(cand_exp, sorted_regions)
                    break
        if "expiry_date" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Expiry\s*Date|Expiration\s*Date|Date\s*of\s*Expiry|Expires\s*On|Valid\s*Until)\b'],
                val_pattern=r'([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2})'
            )
            if val and self._is_valid_date(val):
                vis_fields["expiry_date"] = val.strip()
                vis_confs["expiry_date"] = conf

        # 5. Passport Number (Visual Zone)
        pass_patterns = [
            r'(?:Passport\s*No|Passport\s*Number|Pass\s*No|Passport\s*/\s*Passeport|Passeport\s*No)[:\s]+([A-Z0-9]{7,12})',
        ]
        for pat in pass_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                cand_p = m.group(1).strip().upper()
                vis_fields["passport_number"] = cand_p
                vis_confs["passport_number"] = self._find_region_confidence(cand_p, sorted_regions)
                break
        if "passport_number" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Passport\s*No|Passport\s*Number|Passport)\b'],
                val_pattern=r'([A-Z0-9]{7,12})'
            )
            if val:
                vis_fields["passport_number"] = val.strip().upper()
                vis_confs["passport_number"] = conf

        # 6. Visa / Document / Control Number (Visual Zone)
        visa_num_patterns = [
            r'(?:Visa\s*No|Visa\s*Number|Control\s*Number|Document\s*No|Visa\s*#)[:\s]+([A-Z0-9]{7,14})',
            r'\bControl\s*Number[:\s]+([0-9]{8,14})',
        ]
        for pat in visa_num_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                cand_vn = m.group(1).strip().upper()
                vis_fields["visa_number"] = cand_vn
                vis_confs["visa_number"] = self._find_region_confidence(cand_vn, sorted_regions)
                break
        if "visa_number" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Visa\s*No|Control\s*Number|Document\s*No)\b'],
                val_pattern=r'([A-Z0-9]{7,14})'
            )
            if val:
                vis_fields["visa_number"] = val.strip().upper()
                vis_confs["visa_number"] = conf

        # 7. Name (Visual Zone)
        name_patterns = [
            r'(?:Name|Full\s*Name|Given\s*Name|Surname|Bearer|Holder)[:\s]+([A-Za-z\s.\'-]{2,50})(?:\n|$)',
        ]
        for pat in name_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                cand_name = m.group(1).strip()
                if cand_name.upper() not in ("VISA", "UNITED STATES", "CONTROL NUMBER"):
                    vis_fields["name"] = cand_name
                    vis_confs["name"] = self._find_region_confidence(cand_name, sorted_regions)
                    break
        if "name" not in vis_fields and sorted_regions:
            val, conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Name|Full\s*Name|Given\s*Name)\b'],
                val_pattern=r'([A-Za-z\s.\'-]{2,50})'
            )
            if val and val.upper() not in ("VISA", "UNITED STATES", "CONTROL NUMBER"):
                vis_fields["name"] = val.strip()
                vis_confs["name"] = conf

        # 8. Issuing Authority
        auth_patterns = [
            r'(?:Authority|Issuing\s*Post|Issued\s*By|Embassy|Consulate)[:\s]+([A-Za-z0-9\s,.-]+)(?:\n|$)',
        ]
        for pat in auth_patterns:
            m = re.search(pat, ocr_text, re.IGNORECASE)
            if m:
                vis_fields["issuing_authority"] = m.group(1).strip()
                vis_confs["issuing_authority"] = self._find_region_confidence(m.group(1), sorted_regions)
                break

        # 9. Second Pass Crop Fallback for Missing Visual Dates if image provided
        if document_image is not None and ocr_service is not None and sorted_regions:
            if "issue_date" not in vis_fields:
                self._try_second_pass_crop_date(
                    document_image, ocr_service, sorted_regions,
                    label_keywords=[r'\b(?:Issue\s*Date|Date\s*of\s*Issue|Issued)\b'],
                    target_field="issue_date",
                    vis_fields=vis_fields, vis_confs=vis_confs
                )
            if "expiry_date" not in vis_fields:
                self._try_second_pass_crop_date(
                    document_image, ocr_service, sorted_regions,
                    label_keywords=[r'\b(?:Expiry\s*Date|Date\s*of\s*Expiry|Expiration\s*Date|Expires\s*On|Valid\s*Until)\b'],
                    target_field="expiry_date",
                    vis_fields=vis_fields, vis_confs=vis_confs
                )

        # =========================================================
        # PHASE 2: Dual-Source Fusion (MRV OCR + Visual Zone OCR)
        # =========================================================
        has_valid_mrz = bool(mrz_fields and mrz_format in ("MRVA", "MRVB"))

        if has_valid_mrz:
            mrz_doc_num = mrz_fields.get("document_number", mrz_fields.get("visa_number", ""))
            fields["visa_number"] = mrz_doc_num
            fields["document_number"] = mrz_doc_num
            confidences["visa_number"] = 0.98
            confidences["document_number"] = 0.98
            self.last_field_sources["visa_number"] = {"value": mrz_doc_num, "source": "mrz", "confidence": 0.98}
            self.last_field_sources["document_number"] = {"value": mrz_doc_num, "source": "mrz", "confidence": 0.98}

            fields["surname"] = mrz_fields.get("surname", "")
            fields["given_names"] = mrz_fields.get("given_names", "")
            mrz_full_name = f"{fields['surname']} {fields['given_names']}".strip()
            fields["name"] = mrz_full_name if mrz_full_name else vis_fields.get("name", "")
            confidences["name"] = 0.98 if mrz_full_name else vis_confs.get("name", 0.85)
            self.last_field_sources["name"] = {"value": fields["name"], "source": "mrz" if mrz_full_name else "visual_ocr", "confidence": confidences["name"]}

            fields["nationality"] = mrz_fields.get("nationality", "")
            confidences["nationality"] = 0.98 if fields["nationality"] else 0.0
            self.last_field_sources["nationality"] = {"value": fields["nationality"], "source": "mrz", "confidence": confidences["nationality"]}

            fields["date_of_birth"] = mrz_fields.get("date_of_birth", "")
            confidences["date_of_birth"] = 0.98 if fields["date_of_birth"] else 0.0
            self.last_field_sources["date_of_birth"] = {"value": fields["date_of_birth"], "source": "mrz", "confidence": confidences["date_of_birth"]}

            fields["sex"] = mrz_fields.get("sex", "")
            confidences["sex"] = 0.98 if fields["sex"] else 0.0
            self.last_field_sources["sex"] = {"value": fields["sex"], "source": "mrz", "confidence": confidences["sex"]}

            fields["issuing_state"] = mrz_fields.get("issuing_state", "")
            confidences["issuing_state"] = 0.98 if fields["issuing_state"] else 0.0
            self.last_field_sources["issuing_state"] = {"value": fields["issuing_state"], "source": "mrz", "confidence": confidences["issuing_state"]}

            fields["mrz_format"] = mrz_format
        else:
            # Fall back to visual zone for identity fields
            if "visa_number" in vis_fields:
                fields["visa_number"] = vis_fields["visa_number"]
                fields["document_number"] = vis_fields["visa_number"]
                confidences["visa_number"] = vis_confs.get("visa_number", 0.85)
                confidences["document_number"] = vis_confs.get("visa_number", 0.85)
                self.last_field_sources["visa_number"] = {"value": fields["visa_number"], "source": "visual_ocr", "confidence": confidences["visa_number"]}
                self.last_field_sources["document_number"] = {"value": fields["document_number"], "source": "visual_ocr", "confidence": confidences["document_number"]}

            if "name" in vis_fields:
                fields["name"] = vis_fields["name"]
                confidences["name"] = vis_confs.get("name", 0.85)
                self.last_field_sources["name"] = {"value": fields["name"], "source": "visual_ocr", "confidence": confidences["name"]}

        # 2. Visa-Specific Printed Fields -> Prefer visual zone
        for k in ("visa_type", "entries", "issue_date", "passport_number", "issuing_authority"):
            if k in vis_fields:
                fields[k] = vis_fields[k]
                confidences[k] = vis_confs.get(k, 0.90)
                self.last_field_sources[k] = {"value": vis_fields[k], "source": "visual_ocr", "confidence": confidences[k]}

        # 3. Expiry Date Fusion (prefer full visual calendar format if present)
        if "expiry_date" in vis_fields:
            fields["expiry_date"] = vis_fields["expiry_date"]
            confidences["expiry_date"] = vis_confs.get("expiry_date", 0.90)
            self.last_field_sources["expiry_date"] = {"value": vis_fields["expiry_date"], "source": "visual_ocr", "confidence": confidences["expiry_date"]}
        elif has_valid_mrz and mrz_fields.get("date_of_expiry"):
            fields["expiry_date"] = mrz_fields["date_of_expiry"]
            confidences["expiry_date"] = 0.98
            self.last_field_sources["expiry_date"] = {"value": mrz_fields["date_of_expiry"], "source": "mrz", "confidence": 0.98}

        # 4. Discrepancy Detection & Warnings
        if has_valid_mrz:
            mrz_doc_num = mrz_fields.get("document_number", mrz_fields.get("visa_number", ""))
            vis_doc_num = vis_fields.get("visa_number", "")
            if vis_doc_num and mrz_doc_num:
                norm_vis = re.sub(r'[^A-Z0-9]', '', vis_doc_num.upper())
                norm_mrz = re.sub(r'[^A-Z0-9]', '', mrz_doc_num.upper())
                if norm_vis != norm_mrz and norm_vis not in norm_mrz and norm_mrz not in norm_vis:
                    disc_msg = f"Discrepancy: Visual document number '{vis_doc_num}' differs from MRV document number '{mrz_doc_num}'"
                    self.last_warnings.append(disc_msg)
                    if self.last_field_debug is not None:
                        self.last_field_debug["discrepancies"].append({"field": "document_number", "visual": vis_doc_num, "mrz": mrz_doc_num})

            vis_name = vis_fields.get("name", "")
            mrz_name = fields.get("name", "")
            if vis_name and mrz_name:
                vis_tokens = set(re.findall(r'[A-Z]{3,}', vis_name.upper()))
                mrz_tokens = set(re.findall(r'[A-Z]{3,}', mrz_name.upper()))
                if vis_tokens and mrz_tokens and not (vis_tokens & mrz_tokens):
                    disc_msg = f"Discrepancy: Visual name '{vis_name}' differs from MRV name '{mrz_name}'"
                    self.last_warnings.append(disc_msg)
                    if self.last_field_debug is not None:
                        self.last_field_debug["discrepancies"].append({"field": "name", "visual": vis_name, "mrz": mrz_name})

        return fields, confidences



class NationalIDExtractor(BaseDocumentExtractor):
    """Structured field extractor for National IDs with English-first spatial line grouping, candidate scoring, and confidence gating."""

    def __init__(self, min_confidence: float = 0.50, language_mode: str = "english_first"):
        self.min_confidence = min_confidence
        self.language_mode = language_mode
        self.last_field_debug: Optional[Dict[str, Any]] = None

    @property
    def document_type(self) -> str:
        return DocumentTypeEnum.NATIONAL_ID.value

    def extract_fields(
        self,
        ocr_text: str,
        mrz_fields: Dict[str, Any],
        mrz_format: Optional[str] = None,
        ocr_regions: Optional[List[OCRRegion]] = None,
        document_image: Optional[np.ndarray] = None,
        ocr_service: Optional[Any] = None,
        include_debug: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        fields: Dict[str, Any] = {
            "name": "",
            "id_number": "",
            "document_number": "",
            "date_of_birth": "",
            "gender": "",
            "sex": "",
            "nationality": "",
            "expiry_date": "",
            "issuing_state": "",
            "issuing_authority": "",
            "address": ""
        }
        confidences: Dict[str, float] = {}
        self.last_field_debug = {
            "dob_candidates": [],
            "gender_candidates": []
        }


        # 1. Populate from MRZ fields (TD1 / TD2) if available
        if mrz_fields:
            doc_num = mrz_fields.get("document_number", mrz_fields.get("id_number", ""))
            fields["id_number"] = doc_num
            fields["document_number"] = doc_num
            fields["surname"] = mrz_fields.get("surname", "")
            fields["given_names"] = mrz_fields.get("given_names", "")
            full_name = f"{fields['surname']} {fields['given_names']}".strip()
            if full_name:
                fields["name"] = full_name
            fields["date_of_birth"] = mrz_fields.get("date_of_birth", "")
            sex = mrz_fields.get("sex", mrz_fields.get("gender", ""))
            fields["sex"] = sex
            fields["gender"] = sex
            fields["nationality"] = mrz_fields.get("nationality", "")
            fields["expiry_date"] = mrz_fields.get("date_of_expiry", "")
            if mrz_fields.get("issuing_state"):
                fields["issuing_state"] = mrz_fields.get("issuing_state")
            if mrz_format:
                fields["mrz_format"] = mrz_format
            return fields, confidences

        # 2. Visual OCR & Spatial Line Grouping for Non-MRZ National IDs
        raw_regions = ocr_regions or []
        
        # Step 2a: English-First Line Grouping (merges adjacent word tokens on the same horizontal line)
        line_regions = self._group_tokens_by_line(raw_regions)
        all_candidate_regions = line_regions if line_regions else raw_regions
        sorted_regions = sorted(all_candidate_regions, key=lambda r: (r.bbox[1], r.bbox[0])) if all_candidate_regions else []

        # A. ID / Document Number Extraction
        # Priority 1: 12-digit grouped format (e.g. Aadhaar `8416 1590 3267`)
        id_12_match = re.search(r'\b([0-9]{4}\s[0-9]{4}\s[0-9]{4})\b', ocr_text)
        if not id_12_match and sorted_regions:
            # Check grouped lines for 12-digit pattern
            for lr in sorted_regions:
                m12 = re.search(r'\b([0-9]{4}\s[0-9]{4}\s[0-9]{4})\b', lr.text)
                if m12:
                    id_12_match = m12
                    break

        if id_12_match:
            val = id_12_match.group(1)
            fields["id_number"] = val
            fields["document_number"] = val
            confidences["id_number"] = self._find_region_confidence(val, sorted_regions)
        elif not fields["id_number"]:
            # Priority 2: Label-based ID search
            cand_id = re.search(
                r'(?:ID No|Identity No|National ID|Card No|Card Number|Document No|Document Number|ID Number|Aadhaar|Voter ID|PAN|SSN)[:\s#]+([A-Z0-9 -]{5,20})',
                ocr_text,
                re.IGNORECASE
            )
            if cand_id:
                val = cand_id.group(1).strip()
                fields["id_number"] = val
                fields["document_number"] = val
                confidences["id_number"] = self._find_region_confidence(val, sorted_regions)
            else:
                # Priority 3: Standard Indian PAN `ABCDE1234F`
                pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', ocr_text)
                if pan_match:
                    fields["id_number"] = pan_match.group(1)
                    fields["document_number"] = pan_match.group(1)
                    confidences["id_number"] = self._find_region_confidence(pan_match.group(1), sorted_regions)
                else:
                    # Generic alphanumeric ID (must contain digits)
                    generic_id = re.search(r'\b([A-Z]{1,3}[0-9]{6,10}|[0-9]{8,14}|(?=[A-Z0-9]*[0-9])[A-Z0-9]{8,14})\b', ocr_text)
                    if generic_id:
                        fields["id_number"] = generic_id.group(1)
                        fields["document_number"] = generic_id.group(1)
                        confidences["id_number"] = self._find_region_confidence(generic_id.group(1), sorted_regions)

        # B. Date of Birth (DOB) Extraction
        # Strategy 1: Search for DOB / Birth labels on grouped lines or spatial regions
        dob_cand = None
        dob_conf = 0.0

        # Pattern for standard calendar dates (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY, DD MMM YYYY)
        FULL_DATE_PATTERN = r'([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})'
        # Pattern including explicit YOB 4-digit years
        YOB_DATE_PATTERN = r'([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{4})'

        # 1a. Inline match on grouped line regions (prefer exact label matches)
        for lr in sorted_regions:
            m = re.search(
                r'(?:DOB|Date of Birth|Birth Date|Birth|D\.O\.B|Year of Birth|YOB|DOB/Date of Birth|DOB / Date of Birth)[:\s/]+' + YOB_DATE_PATTERN,
                lr.text,
                re.IGNORECASE
            )
            if m:
                cand_val = m.group(1).strip()
                if self._is_valid_date_candidate(cand_val, label_context=lr.text, id_number=fields["id_number"]):
                    dob_cand = cand_val
                    dob_conf = lr.confidence
                    break

        # 1b. Inline match on raw text if not found in grouped lines
        if not dob_cand:
            m_text = re.search(
                r'(?:DOB|Date of Birth|Birth Date|Birth|D\.O\.B|Year of Birth|YOB|DOB/Date of Birth|DOB / Date of Birth)[:\s/]+' + YOB_DATE_PATTERN,
                ocr_text,
                re.IGNORECASE
            )
            if m_text:
                cand_val = m_text.group(1).strip()
                if self._is_valid_date_candidate(cand_val, label_context=m_text.group(0), id_number=fields["id_number"]):
                    dob_cand = cand_val
                    dob_conf = self._find_region_confidence(cand_val, sorted_regions)

        # 1c. Spatial search: label in one box, date value to the right or immediately below
        if not dob_cand and sorted_regions:
            for i, l_reg in enumerate(sorted_regions):
                l_text = l_reg.text.strip()
                is_dob_lbl = bool(re.search(r'\b(?:DOB|Date of Birth|Birth Date|Year of Birth|YOB|Birth)\b', l_text, re.IGNORECASE))
                if not is_dob_lbl:
                    continue

                lx1, ly1, lx2, ly2 = l_reg.bbox
                l_cy = (ly1 + ly2) / 2.0
                l_h = max(10.0, ly2 - ly1)

                # Search candidate regions to the RIGHT on the SAME horizontal line
                for r in sorted_regions:
                    if r == l_reg:
                        continue
                    rx1, ry1, rx2, ry2 = r.bbox
                    r_cy = (ry1 + ry2) / 2.0
                    # Same horizontal band & to the right
                    if abs(r_cy - l_cy) <= max(15.0, l_h * 1.2) and rx1 >= lx1:
                        # Extract full date or YOB year
                        dm = re.search(YOB_DATE_PATTERN, r.text)
                        if dm:
                            cand = dm.group(1).strip()
                            if self._is_valid_date_candidate(cand, label_context=l_text, id_number=fields["id_number"]):
                                dob_cand = cand
                                dob_conf = r.confidence
                                break
                if dob_cand:
                    break

                # Search candidate regions immediately BELOW within 1.5x height
                for r in sorted_regions:
                    if r == l_reg:
                        continue
                    rx1, ry1, rx2, ry2 = r.bbox
                    if 0 <= (ry1 - ly2) <= max(25.0, l_h * 1.5):
                        dm = re.search(FULL_DATE_PATTERN, r.text)
                        if dm:
                            cand = dm.group(1).strip()
                            if self._is_valid_date_candidate(cand, label_context=l_text, id_number=fields["id_number"]):
                                dob_cand = cand
                                dob_conf = r.confidence
                                break
                if dob_cand:
                    break

        # 1d. Unlabeled standalone full calendar date fallback
        if not dob_cand:
            date_matches = re.findall(r'\b([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2})\b', ocr_text)
            for dm in date_matches:
                if self._is_valid_date_candidate(dm, label_context="standalone", id_number=fields["id_number"]):
                    dob_cand = dm
                    dob_conf = self._find_region_confidence(dm, sorted_regions)
                    break

        if dob_cand:
            fields["date_of_birth"] = dob_cand
            confidences["date_of_birth"] = dob_conf

        # C. Gender / Sex Extraction
        gender_cand = None
        gender_conf = 0.0

        # Strategy 1: Inline label match across grouped lines or raw text
        for lr in sorted_regions:
            m = re.search(
                r'(?:Gender|Sex|Gender\s*/\s*Sex|Sex\s*/\s*Gender)[:\s]+(MALE|FEMALE|TRANSGENDER|OTHER|\bM\b|\bF\b)',
                lr.text,
                re.IGNORECASE
            )
            if m:
                gender_cand = m.group(1).upper().strip()
                gender_conf = lr.confidence
                break

        if not gender_cand:
            m_text = re.search(
                r'(?:Gender|Sex|Gender\s*/\s*Sex|Sex\s*/\s*Gender)[:\s]+(MALE|FEMALE|TRANSGENDER|OTHER|\bM\b|\bF\b)',
                ocr_text,
                re.IGNORECASE
            )
            if m_text:
                gender_cand = m_text.group(1).upper().strip()
                gender_conf = self._find_region_confidence(gender_cand, sorted_regions)

        # Strategy 2: Spatial search for M/F or Gender words adjacent to Gender label
        if not gender_cand and sorted_regions:
            for l_reg in sorted_regions:
                l_text = l_reg.text.strip()
                is_g_lbl = bool(re.search(r'\b(?:Gender|Sex)\b', l_text, re.IGNORECASE))
                if not is_g_lbl:
                    continue

                lx1, ly1, lx2, ly2 = l_reg.bbox
                l_cy = (ly1 + ly2) / 2.0
                l_h = max(10.0, ly2 - ly1)

                # Search to the right
                for r in sorted_regions:
                    if r == l_reg:
                        continue
                    rx1, ry1, rx2, ry2 = r.bbox
                    r_cy = (ry1 + ry2) / 2.0
                    if abs(r_cy - l_cy) <= max(15.0, l_h * 1.2) and rx1 >= lx1:
                        gm = re.search(r'\b(MALE|FEMALE|TRANSGENDER|OTHER|M|F)\b', r.text, re.IGNORECASE)
                        if gm:
                            gender_cand = gm.group(1).upper().strip()
                            gender_conf = r.confidence
                            break
                if gender_cand:
                    break

        # Strategy 3: Standalone English gender words (MALE, FEMALE, TRANSGENDER) - NOT isolated single letters
        if not gender_cand:
            # Check FEMALE first to prevent MALE matching inside FEMALE
            if re.search(r'\bFEMALE\b', ocr_text, re.IGNORECASE):
                gender_cand = "FEMALE"
                gender_conf = self._find_region_confidence("FEMALE", sorted_regions)
            elif re.search(r'\bMALE\b', ocr_text, re.IGNORECASE):
                gender_cand = "MALE"
                gender_conf = self._find_region_confidence("MALE", sorted_regions)
            elif re.search(r'\bTRANSGENDER\b', ocr_text, re.IGNORECASE):
                gender_cand = "TRANSGENDER"
                gender_conf = self._find_region_confidence("TRANSGENDER", sorted_regions)

        if gender_cand:
            fields["gender"] = gender_cand
            fields["sex"] = "M" if ("M" in gender_cand and "FE" not in gender_cand) else ("F" if "F" in gender_cand else "X")
            confidences["gender"] = gender_conf


        # D. Name Extraction (English-First Candidate Ranking)
        # Strategy 1: Explicit labeled name search
        name_labeled = None
        for lr in sorted_regions:
            m = re.search(
                r'(?:Name|Full Name|Given Name|Holder|Cardholder|Name of Holder)[:\s]+([A-Za-z\s.\'-]+)(?:\n|$)',
                lr.text,
                re.IGNORECASE
            )
            if m:
                cand = m.group(1).strip()
                score = self._score_name_candidate(cand, lr.confidence, self.min_confidence)
                if score > 0:
                    name_labeled = (cand, lr.confidence)
                    break

        if not name_labeled:
            m_text = re.search(
                r'(?:Name|Full Name|Given Name|Holder|Cardholder|Name of Holder)[:\s]+([A-Za-z\s.\'-]+)(?:\n|$)',
                ocr_text,
                re.IGNORECASE
            )
            if m_text:
                cand = m_text.group(1).strip()
                score = self._score_name_candidate(cand, 0.85, self.min_confidence)
                if score > 0:
                    name_labeled = (cand, self._find_region_confidence(cand, sorted_regions))

        if name_labeled:
            fields["name"] = name_labeled[0]
            confidences["name"] = name_labeled[1]
        else:
            # Strategy 2: Spatial label search in OCR regions
            name_val, name_conf = self._extract_spatial_value(
                sorted_regions,
                label_patterns=[r'\b(?:Name|Full Name|Given Name|Cardholder)\b'],
                val_pattern=r'([A-Za-z\s.\'-]{3,40})'
            )
            if name_val and self._score_name_candidate(name_val, name_conf, self.min_confidence) > 0:
                fields["name"] = name_val.strip()
                confidences["name"] = name_conf

        # Strategy 3: English-First Candidate Ranking for unlabeled layouts (e.g. Aadhaar)
        if not fields["name"] and sorted_regions:
            scored_candidates: List[Tuple[float, str, float, float]] = []
            
            # Evaluate all grouped line regions
            for lr in sorted_regions:
                text_clean = lr.text.strip()
                score = self._score_name_candidate(text_clean, lr.confidence, self.min_confidence)
                if score > 0:
                    # Store (score, text, confidence, y_pos)
                    scored_candidates.append((score, text_clean, lr.confidence, lr.bbox[1]))

            # Also evaluate raw individual tokens if no grouped candidate qualified
            if not scored_candidates and raw_regions:
                for r in raw_regions:
                    text_clean = r.text.strip()
                    score = self._score_name_candidate(text_clean, r.confidence, self.min_confidence)
                    if score > 0:
                        scored_candidates.append((score, text_clean, r.confidence, r.bbox[1]))

            if scored_candidates:
                # Rank primarily by quality score (highest score first)
                scored_candidates.sort(key=lambda c: (-c[0], c[3]))
                best_score, best_name, best_conf, _ = scored_candidates[0]
                fields["name"] = best_name
                confidences["name"] = best_conf
        elif not fields["name"]:
            # Fallback line inspection on raw OCR text
            raw_cands = []
            for line in ocr_text.splitlines():
                line_clean = line.strip()
                score = self._score_name_candidate(line_clean, 0.70, self.min_confidence)
                if score > 0:
                    raw_cands.append((score, line_clean))
            if raw_cands:
                raw_cands.sort(key=lambda c: -c[0])
                fields["name"] = raw_cands[0][1]
                confidences["name"] = 0.75

        # E. Nationality Extraction
        nat_match = re.search(r'(?:Nationality|Citizen of)[:\s]+([A-Za-z]+)', ocr_text, re.IGNORECASE)
        if nat_match:
            fields["nationality"] = nat_match.group(1).upper()
            confidences["nationality"] = self._find_region_confidence(nat_match.group(1), sorted_regions)
        elif "INDIAN" in ocr_text.upper():
            fields["nationality"] = "INDIAN"
            confidences["nationality"] = 0.90

        # F. Address Extraction
        addr_match = re.search(r'(?:Address|Permanent Address|Residence)[:\s]+([^\n]+(?:\n[^\n]+)?)', ocr_text, re.IGNORECASE)
        if addr_match:
            fields["address"] = addr_match.group(1).strip().replace("\n", ", ")
            confidences["address"] = self._find_region_confidence(addr_match.group(1), sorted_regions)

        # G. Issuing Authority Extraction
        if "GOVERNMENT OF INDIA" in ocr_text.upper():
            fields["issuing_authority"] = "Government of India"
        elif "UNIQUE IDENTIFICATION AUTHORITY OF INDIA" in ocr_text.upper():
            fields["issuing_authority"] = "Unique Identification Authority of India"
        elif "ELECTION COMMISSION OF INDIA" in ocr_text.upper():
            fields["issuing_authority"] = "Election Commission of India"
        else:
            auth_match = re.search(r'(?:Authority|Issued By|Government of)[:\s]+([A-Za-z\s,]+)(?:\n|$)', ocr_text, re.IGNORECASE)
            if auth_match:
                fields["issuing_authority"] = auth_match.group(1).strip()

        # Step 2b: Second Pass - Targeted Dynamic Field ROI Crop Extraction for DOB and Gender
        if document_image is not None and ocr_service is not None and sorted_regions:
            # 1. Target DOB Label Regions
            dob_label_regions = []
            for r in sorted_regions:
                if re.search(r'\b(?:DOB|Date of Birth|Birth Date|Year of Birth|YOB|Birth)\b', r.text, re.IGNORECASE):
                    dob_label_regions.append(r)

            dob_candidates_scored = []
            for l_reg in dob_label_regions:
                variants = ImageService.get_field_crop_variants(
                    document_image,
                    label_bbox=l_reg.bbox,
                    field_type="dob",
                    scale_factor=3.0
                )
                for crop_name, prep_name, prep_img, crop_bbox in variants:
                    for psm in [7, 8, 6]:
                        try:
                            crop_res = ocr_service.extract_field(prep_img, field_type="dob", psm=psm)
                            cand_text = crop_res.raw_text.strip()
                            if not cand_text:
                                continue
                            
                            # Find date matches
                            date_matches = re.findall(
                                r'([0-9]{2}[/.-][0-9]{2}[/.-][0-9]{4}|[0-9]{4}[/.-][0-9]{2}[/.-][0-9]{2}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{4})',
                                cand_text
                            )
                            for dm in date_matches:
                                dm_clean = dm.strip()
                                is_valid = self._is_valid_date_candidate(
                                    dm_clean,
                                    label_context=l_reg.text,
                                    id_number=fields["id_number"]
                                )
                                if not is_valid:
                                    continue
                                
                                base_score = 1.0 if len(dm_clean) > 4 else 0.8
                                conf_bonus = crop_res.average_confidence * 1.5
                                dist = max(0, crop_bbox[0] - l_reg.bbox[2])
                                dist_penalty = dist * 0.001
                                same_line_bonus = 0.3 if "right" in crop_name else 0.0
                                
                                total_score = base_score + conf_bonus - dist_penalty + same_line_bonus
                                
                                candidate_record = {
                                    "value": dm_clean,
                                    "score": round(total_score, 4),
                                    "confidence": round(crop_res.average_confidence, 4),
                                    "crop": crop_name,
                                    "preprocessing": prep_name,
                                    "psm": psm,
                                    "bbox": crop_bbox
                                }
                                self.last_field_debug["dob_candidates"].append(candidate_record)
                                dob_candidates_scored.append((total_score, dm_clean, crop_res.average_confidence))
                        except Exception as e:
                            logger.debug(f"Targeted DOB OCR crop failed: {str(e)}")

            if dob_candidates_scored:
                dob_candidates_scored.sort(key=lambda c: -c[0])
                best_score, best_dob, best_conf = dob_candidates_scored[0]
                if best_score > 0:
                    fields["date_of_birth"] = best_dob
                    confidences["date_of_birth"] = best_conf

            # 2. Target Gender Label Regions
            gender_label_regions = []
            for r in sorted_regions:
                if re.search(r'\b(?:Gender|Sex)\b', r.text, re.IGNORECASE):
                    gender_label_regions.append(r)

            gender_candidates_scored = []
            for l_reg in gender_label_regions:
                variants = ImageService.get_field_crop_variants(
                    document_image,
                    label_bbox=l_reg.bbox,
                    field_type="gender",
                    scale_factor=3.0
                )
                for crop_name, prep_name, prep_img, crop_bbox in variants:
                    for psm in [7, 8, 6]:
                        try:
                            crop_res = ocr_service.extract_field(prep_img, field_type="gender", psm=psm)
                            cand_text = crop_res.raw_text.upper().strip()
                            if not cand_text:
                                continue

                            g_val = None
                            if "FEMALE" in cand_text:
                                g_val = "FEMALE"
                            elif "MALE" in cand_text:
                                g_val = "MALE"
                            elif "TRANSGENDER" in cand_text:
                                g_val = "TRANSGENDER"
                            elif re.search(r'\bM\b', cand_text):
                                g_val = "M"
                            elif re.search(r'\bF\b', cand_text):
                                g_val = "F"

                            if g_val:
                                word_bonus = 1.0 if len(g_val) > 1 else 0.6
                                total_score = word_bonus + crop_res.average_confidence * 1.5
                                candidate_record = {
                                    "value": g_val,
                                    "score": round(total_score, 4),
                                    "confidence": round(crop_res.average_confidence, 4),
                                    "crop": crop_name,
                                    "preprocessing": prep_name,
                                    "psm": psm,
                                    "bbox": crop_bbox
                                }
                                self.last_field_debug["gender_candidates"].append(candidate_record)
                                gender_candidates_scored.append((total_score, g_val, crop_res.average_confidence))
                        except Exception as e:
                            logger.debug(f"Targeted Gender OCR crop failed: {str(e)}")

            if gender_candidates_scored:
                gender_candidates_scored.sort(key=lambda c: -c[0])
                best_score, best_g, best_conf = gender_candidates_scored[0]
                if best_score > 0:
                    fields["gender"] = best_g
                    fields["sex"] = "M" if ("M" in best_g and "FE" not in best_g) else ("F" if "F" in best_g else "X")
                    confidences["gender"] = best_conf

        self.last_field_sources = {
            k: {"value": v, "source": "mrz" if k in mrz_fields and v else "visual_ocr", "confidence": confidences.get(k, 0.90)}
            for k, v in fields.items() if v
        }

        return fields, confidences



    @classmethod
    def _clean_english_token(cls, text: str) -> str:
        """Removes non-Latin characters and OCR garbage, retaining Latin letters, digits, and standard separators."""
        cleaned = re.sub(r'[^\x20-\x7E]', ' ', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    @classmethod
    def _is_valid_date_candidate(cls, date_str: str, label_context: str = "", id_number: str = "") -> bool:
        """Validates that a candidate string is a genuine calendar date or valid labeled birth year, not an ID fragment or arbitrary number."""
        if not date_str:
            return False
        clean = date_str.strip()

        # 1. Reject if candidate is part of the detected ID number
        if id_number:
            id_digits_only = re.sub(r'\D', '', id_number)
            cand_digits_only = re.sub(r'\D', '', clean)
            if cand_digits_only and len(cand_digits_only) >= 4 and cand_digits_only in id_digits_only:
                # E.g. "8416" is the first 4 digits of "8416 1590 3267"
                return False

        # 2. Check 4-digit Year only format (e.g. "1990", "1985")
        if re.fullmatch(r'\d{4}', clean):
            year_val = int(clean)
            # Year must be in a plausible human birth year range (1900 to 2026)
            if not (1900 <= year_val <= 2026):
                return False
            # ONLY accept 4-digit year if label explicitly indicates Year of Birth (YOB)
            lbl_upper = label_context.upper() if label_context else ""
            if any(k in lbl_upper for k in ["YOB", "YEAR OF BIRTH", "BIRTH YEAR", "YEAR"]):
                return True
            return False

        # 3. Check 3-part calendar date formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, YYYY/MM/DD
        d_clean = clean.replace('.', '/').replace('-', '/')
        parts = d_clean.split('/')
        if len(parts) == 3:
            try:
                p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                # DD/MM/YYYY or MM/DD/YYYY
                if 1 <= p1 <= 31 and 1 <= p2 <= 12 and 1900 <= p3 <= 2026:
                    return True
                # YYYY/MM/DD
                if 1900 <= p1 <= 2026 and 1 <= p2 <= 12 and 1 <= p3 <= 31:
                    return True
            except ValueError:
                pass

        # 4. Check DD MMM YYYY (e.g. "15 Aug 1990")
        month_words = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}
        m_match = re.fullmatch(r'(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})', clean)
        if m_match:
            try:
                d_day = int(m_match.group(1))
                m_txt = m_match.group(2)[:3].upper()
                d_year = int(m_match.group(3))
                if 1 <= d_day <= 31 and m_txt in month_words and 1900 <= d_year <= 2026:
                    return True
            except ValueError:
                pass

        return False


    @classmethod
    def _group_tokens_by_line(cls, regions: List[OCRRegion], line_threshold: float = 16.0) -> List[OCRRegion]:
        """Groups nearby same-line word tokens into coherent horizontal lines with length-weighted confidence."""
        if not regions:
            return []

        # Filter out obvious non-Latin garbage or very low confidence noise tokens
        valid_regions = []
        for r in regions:
            cleaned = cls._clean_english_token(r.text)
            if cleaned and len(cleaned) > 0:
                valid_regions.append(OCRRegion(
                    text=cleaned,
                    confidence=r.confidence,
                    bbox=r.bbox
                ))

        if not valid_regions:
            return []

        # Sort by vertical center
        sorted_by_y = sorted(valid_regions, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2.0)
        
        groups: List[List[OCRRegion]] = []
        for r in sorted_by_y:
            r_cy = (r.bbox[1] + r.bbox[3]) / 2.0
            placed = False
            for group in groups:
                group_cy = sum((m.bbox[1] + m.bbox[3]) / 2.0 for m in group) / len(group)
                if abs(r_cy - group_cy) <= line_threshold:
                    group.append(r)
                    placed = True
                    break
            if not placed:
                groups.append([r])

        line_regions: List[OCRRegion] = []
        for group in groups:
            # Sort tokens in the line from left to right
            group.sort(key=lambda r: r.bbox[0])
            merged_text = " ".join(r.text.strip() for r in group if r.text.strip())
            if not merged_text:
                continue
            total_len = sum(len(r.text.strip()) for r in group)
            weighted_conf = sum(r.confidence * len(r.text.strip()) for r in group) / max(1, total_len)
            merged_bbox = [
                min(r.bbox[0] for r in group),
                min(r.bbox[1] for r in group),
                max(r.bbox[2] for r in group),
                max(r.bbox[3] for r in group),
            ]
            line_regions.append(OCRRegion(
                text=merged_text,
                confidence=round(weighted_conf, 4),
                bbox=merged_bbox
            ))

        # Sort line regions top to bottom
        line_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return line_regions

    @classmethod
    def _score_name_candidate(cls, text: str, confidence: float, min_confidence: float = 0.50) -> float:
        """Computes a heuristic quality score for a personal name candidate based on English morphology, word count, confidence, and position."""
        clean = text.strip()
        if len(clean) < 3 or len(clean) > 50:
            return -100.0
        # Reject if digits are present
        if any(c.isdigit() for c in clean):
            return -100.0
        
        # Check Latin alphabet ratio
        latin_letters = sum(1 for c in clean if ('A' <= c <= 'Z' or 'a' <= c <= 'z'))
        if latin_letters / max(1, len(clean)) < 0.70:
            return -100.0

        words = [w for w in clean.split() if w]
        if not words:
            return -100.0

        # Reject if any significant word matches the header blacklist
        for w in words:
            w_upper = re.sub(r'[^A-Z]', '', w.upper())
            if w_upper in HEADER_BLACKLIST:
                return -100.0

        upper_text = clean.upper()
        if any(k in upper_text for k in ["GOVERNMENT", "AUTHORITY", "AADHAAR", "IDENTITY", "CARD", "E-AADHAAR", "REPUBLIC", "ELECTION", "COMMISSION", "DIRECTORATE"]):
            return -100.0

        # Base score from OCR confidence
        score = confidence * 1.5

        # Word count bonus (personal names are typically 2 to 4 words, e.g. "Sriram Mamundi")
        if len(words) == 2:
            score += 0.50
        elif len(words) == 3:
            score += 0.45
        elif len(words) == 4:
            score += 0.35
        elif len(words) == 1:
            if len(clean) >= 5:
                score += 0.10
            else:
                score -= 0.60  # heavily penalize short isolated noise tokens like "wher"

        # Capitalization quality bonus (Title Case or UPPER CASE)
        all_capitalized = all(w[0].isupper() for w in words if w)
        all_upper = clean.isupper()
        all_lower = clean.islower()
        if all_capitalized or all_upper:
            score += 0.30
        elif all_lower:
            score -= 0.50  # penalize all-lowercase garbage artifacts

        # Pure Latin letters & spaces bonus
        if re.fullmatch(r"[A-Za-z\s.'-]+", clean):
            score += 0.20
        else:
            score -= 0.30

        # Confidence threshold gating
        if confidence < min_confidence:
            score -= 0.80

        return score

    @staticmethod
    def _find_region_confidence(val: str, regions: List[OCRRegion]) -> float:
        """Finds confidence score for matching text substring across OCR regions."""
        if not val or not regions:
            return 0.85
        val_clean = val.strip().lower()
        for r in regions:
            if val_clean in r.text.lower():
                return r.confidence
        return 0.85

    @classmethod
    def _extract_spatial_value(
        cls,
        regions: List[OCRRegion],
        label_patterns: List[str],
        val_pattern: str
    ) -> Tuple[Optional[str], float]:
        """Searches for a field value spatially located to the right or immediately below a detected label region."""
        if not regions:
            return None, 0.0

        for i, label_reg in enumerate(regions):
            label_text = label_reg.text.strip()
            # Check if region matches any target label pattern
            is_label = any(re.search(pat, label_text, re.IGNORECASE) for pat in label_patterns)
            if not is_label:
                continue

            # 1. Check if the value is already inside the same region after the label
            inline_match = re.search(val_pattern, label_text)
            if inline_match and inline_match.group(1).strip() != label_text:
                return inline_match.group(1).strip(), label_reg.confidence

            lx1, ly1, lx2, ly2 = label_reg.bbox
            l_height = ly2 - ly1
            l_center_y = (ly1 + ly2) / 2.0

            # 2. Search candidate regions to the RIGHT of the label (similar Y vertical band)
            right_candidates = []
            for r in regions:
                if r == label_reg:
                    continue
                rx1, ry1, rx2, ry2 = r.bbox
                r_center_y = (ry1 + ry2) / 2.0
                if abs(r_center_y - l_center_y) <= max(15.0, l_height * 1.2) and rx1 >= lx1:
                    m = re.search(val_pattern, r.text)
                    if m:
                        dist = rx1 - lx2
                        right_candidates.append((dist, m.group(1).strip(), r.confidence))

            if right_candidates:
                right_candidates.sort(key=lambda c: c[0])
                return right_candidates[0][1], right_candidates[0][2]

            # 3. Search candidate regions immediately BELOW the label
            below_candidates = []
            for r in regions:
                if r == label_reg:
                    continue
                rx1, ry1, rx2, ry2 = r.bbox
                if 0 <= (ry1 - ly2) <= max(35.0, l_height * 2.5):
                    # Horizontal alignment check
                    if abs(rx1 - lx1) <= 250 or abs(rx2 - lx2) <= 250:
                        m = re.search(val_pattern, r.text)
                        if m:
                            v_dist = ry1 - ly2
                            below_candidates.append((v_dist, m.group(1).strip(), r.confidence))

            if below_candidates:
                below_candidates.sort(key=lambda c: c[0])
                return below_candidates[0][1], below_candidates[0][2]

        return None, 0.0



class DocumentService:
    """Orchestrates document classification and field extraction across Passports, Visas, and National IDs."""

    def __init__(self):
        self.extractors: Dict[str, BaseDocumentExtractor] = {
            DocumentTypeEnum.PASSPORT.value: PassportExtractor(),
            DocumentTypeEnum.VISA.value: VisaExtractor(),
            DocumentTypeEnum.NATIONAL_ID.value: NationalIDExtractor(
                min_confidence=settings.NATIONAL_ID_CONFIDENCE_THRESHOLD,
                language_mode=settings.DEFAULT_LANGUAGE_MODE
            ),
        }


    def detect_document_type(self, ocr_text: str, mrz_result: MRZResult) -> Tuple[str, float]:
        """Classifies document type based on MRZ format signatures and visual keyword cues."""
        text_upper = ocr_text.upper()
        
        # 1. MRZ Signature Classification (Highest Confidence)
        if mrz_result.detected and mrz_result.format:
            fmt = mrz_result.format.upper()
            if fmt in ("MRVA", "MRVB") or (mrz_result.document_code and mrz_result.document_code.startswith("V")):
                return DocumentTypeEnum.VISA.value, 0.98
            elif fmt in ("TD1", "TD2") or (mrz_result.document_code and mrz_result.document_code.startswith(("I", "A", "C"))):
                return DocumentTypeEnum.NATIONAL_ID.value, 0.98
            elif fmt == "TD3" or (mrz_result.document_code and mrz_result.document_code.startswith("P")):
                return DocumentTypeEnum.PASSPORT.value, 0.98
                
        # 2. Visual Keyword Classification
        visa_keywords = ["VISA", "ENTRY PERMIT", "CONTROL NUMBER", "NUMBER OF ENTRIES", "VISA TYPE", "VALID FOR"]
        if any(k in text_upper for k in visa_keywords):
            return DocumentTypeEnum.VISA.value, 0.92
            
        passport_keywords = ["PASSPORT", "PASSEPORT", "TRAVEL DOCUMENT", "REPUBLIC OF"]
        if any(k in text_upper for k in passport_keywords):
            return DocumentTypeEnum.PASSPORT.value, 0.92
            
        id_keywords = ["IDENTITY CARD", "NATIONAL ID", "AADHAAR", "CITIZEN", "DRIVING LICENCE", "IDENTITY NUMBER", "IDENTIFICATION", "ELECTION COMMISSION"]
        if any(k in text_upper for k in id_keywords):
            return DocumentTypeEnum.NATIONAL_ID.value, 0.92

        # 3. Default Fallback
        if mrz_result.detected:
            return DocumentTypeEnum.PASSPORT.value, 0.85

        return DocumentTypeEnum.NATIONAL_ID.value, 0.50


    def process_extraction(
        self,
        requested_type: str,
        ocr_text: str,
        mrz_result: MRZResult,
        mrz_fields: Dict[str, Any],
        ocr_regions: Optional[List[OCRRegion]] = None,
        document_image: Optional[np.ndarray] = None,
        ocr_service: Optional[Any] = None,
        include_debug: bool = False
    ) -> Tuple[str, Dict[str, Any], Dict[str, float], List[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Determines effective document type, extracts structured fields with per-field confidences, and provides diagnostic warnings."""
        warnings: List[str] = []
        
        if requested_type == DocumentTypeEnum.AUTO.value or not requested_type:
            effective_type, conf = self.detect_document_type(ocr_text, mrz_result)
        else:
            effective_type = requested_type.lower()
            
        extractor = self.extractors.get(effective_type, self.extractors[DocumentTypeEnum.NATIONAL_ID.value])
        fields, confidences = extractor.extract_fields(
            ocr_text,
            mrz_fields,
            mrz_format=mrz_result.format,
            ocr_regions=ocr_regions,
            document_image=document_image,
            ocr_service=ocr_service,
            include_debug=include_debug
        )
        
        # Attach informational warnings from extractor
        extractor_warnings = getattr(extractor, "last_warnings", [])
        if extractor_warnings:
            warnings.extend(extractor_warnings)

        if effective_type == DocumentTypeEnum.NATIONAL_ID.value and not mrz_result.detected:
            warnings.append("MRZ not detected on this National ID; fields extracted via visual OCR key-value analysis.")
        elif not mrz_result.detected:
            warnings.append(f"No MRZ detected for {effective_type}; extracted fields rely on visual OCR text.")

        field_debug_info = getattr(extractor, "last_field_debug", None)
        field_sources_info = getattr(extractor, "last_field_sources", None)
            
        return effective_type, fields, confidences, warnings, field_debug_info, field_sources_info




