"""Document classification and extensible field extraction service for Passports, Visas, and National IDs."""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional
from app.models.schemas import DocumentTypeEnum, MRZResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseDocumentExtractor(ABC):
    """Abstract base extractor for specific document types."""

    @property
    @abstractmethod
    def document_type(self) -> str:
        pass

    @abstractmethod
    def extract_fields(self, ocr_text: str, mrz_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts structured key-value fields from OCR text and MRZ data."""
        pass


class PassportExtractor(BaseDocumentExtractor):
    """Structured field extractor for Passports."""

    @property
    def document_type(self) -> str:
        return DocumentTypeEnum.PASSPORT.value

    def extract_fields(self, ocr_text: str, mrz_fields: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        
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
            return fields

        # Fallback regex extraction from visual text if MRZ was absent/damaged
        passport_num_match = re.search(r'\b([A-Z][0-9]{7,8}|[A-Z0-9]{8,9})\b', ocr_text)
        if passport_num_match:
            fields["passport_number"] = passport_num_match.group(1)
            
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

        return fields


class VisaExtractor(BaseDocumentExtractor):
    """Structured field extractor for Visas (pluggable template-ready)."""

    @property
    def document_type(self) -> str:
        return DocumentTypeEnum.VISA.value

    def extract_fields(self, ocr_text: str, mrz_fields: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "visa_number": "",
            "name": "",
            "nationality": "",
            "date_of_birth": "",
            "passport_number": "",
            "visa_type": "",
            "issue_date": "",
            "expiry_date": "",
            "entries": "",
            "issuing_authority": ""
        }

        # If MRZ exists on Visa (some visas have MRZ)
        if mrz_fields:
            fields["passport_number"] = mrz_fields.get("passport_number", "")
            fields["nationality"] = mrz_fields.get("nationality", "")
            fields["date_of_birth"] = mrz_fields.get("date_of_birth", "")
            full_name = f"{mrz_fields.get('surname', '')} {mrz_fields.get('given_names', '')}".strip()
            if full_name:
                fields["name"] = full_name
            if mrz_fields.get("date_of_expiry"):
                fields["expiry_date"] = mrz_fields.get("date_of_expiry")

        # Visa Number detection (e.g., Red control numbers or alphanumeric visa identifiers)
        visa_num = re.search(r'(?:Visa No|Control Number|Document No|Visa Number)[:\s]+([A-Z0-9]{7,12})', ocr_text, re.IGNORECASE)
        if visa_num:
            fields["visa_number"] = visa_num.group(1)
        elif not fields["visa_number"]:
            cand = re.search(r'\b([A-Z]\d{7,9}|\d{8,9})\b', ocr_text)
            if cand:
                fields["visa_number"] = cand.group(1)

        # Name extraction
        if not fields["name"]:
            name_match = re.search(r'(?:Name|Full Name|Given Name)[:\s]+([A-Za-z\s]+)(?:\n|$)', ocr_text, re.IGNORECASE)
            if name_match:
                fields["name"] = name_match.group(1).strip()

        # Passport Number
        if not fields["passport_number"]:
            p_num = re.search(r'(?:Passport No|Passport Number)[:\s]+([A-Z0-9]{7,10})', ocr_text, re.IGNORECASE)
            if p_num:
                fields["passport_number"] = p_num.group(1)

        # Visa Type / Class
        v_type = re.search(r'(?:Type / Class|Visa Type|Class|Category)[:\s]+([A-Z0-9/-]+)', ocr_text, re.IGNORECASE)
        if v_type:
            fields["visa_type"] = v_type.group(1)

        # Number of Entries
        entries_match = re.search(r'(?:Entries|Number of Entries)[:\s]+(MULTIPLE|SINGLE|DOUBLE|M|S|1|2|\b[0-9]\b)', ocr_text, re.IGNORECASE)
        if entries_match:
            fields["entries"] = entries_match.group(1).upper()

        # Issue Date
        issue_match = re.search(r'(?:Issue Date|Date of Issue|Issued)[:\s]+([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{2}[A-Za-z]{3}[0-9]{4})', ocr_text, re.IGNORECASE)
        if issue_match:
            fields["issue_date"] = issue_match.group(1)

        # Expiry Date
        if not fields["expiry_date"]:
            exp_match = re.search(r'(?:Expiry Date|Expiration Date|Expires On|Valid Until)[:\s]+([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{2}[A-Za-z]{3}[0-9]{4})', ocr_text, re.IGNORECASE)
            if exp_match:
                fields["expiry_date"] = exp_match.group(1)

        # Nationality
        if not fields["nationality"]:
            nat_match = re.search(r'(?:Nationality)[:\s]+([A-Za-z]+)', ocr_text, re.IGNORECASE)
            if nat_match:
                fields["nationality"] = nat_match.group(1).upper()

        # Issuing Authority
        auth_match = re.search(r'(?:Authority|Issuing Post|Issued By)[:\s]+([A-Za-z\s,]+)(?:\n|$)', ocr_text, re.IGNORECASE)
        if auth_match:
            fields["issuing_authority"] = auth_match.group(1).strip()

        return fields


class NationalIDExtractor(BaseDocumentExtractor):
    """Structured field extractor for National IDs (pluggable template-ready)."""

    @property
    def document_type(self) -> str:
        return DocumentTypeEnum.NATIONAL_ID.value

    def extract_fields(self, ocr_text: str, mrz_fields: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "name": "",
            "id_number": "",
            "date_of_birth": "",
            "gender": "",
            "nationality": "",
            "address": ""
        }

        # ID Number heuristics (e.g. 12-digit Aadhaar `XXXX XXXX XXXX`, alphanumeric ID `X12345678`, SSN `XXX-XX-XXXX`)
        id_match = re.search(r'\b([0-9]{4}\s[0-9]{4}\s[0-9]{4})\b', ocr_text)  # Aadhaar format
        if id_match:
            fields["id_number"] = id_match.group(1)
        else:
            cand_id = re.search(r'(?:ID No|Identity No|National ID|Card No|Document No)[:\s]+([A-Z0-9-]+)', ocr_text, re.IGNORECASE)
            if cand_id:
                fields["id_number"] = cand_id.group(1)
            else:
                generic_id = re.search(r'\b([A-Z]{1,3}[0-9]{6,10}|[0-9]{9,12})\b', ocr_text)
                if generic_id:
                    fields["id_number"] = generic_id.group(1)

        # Name Extraction
        name_match = re.search(r'(?:Name|Full Name|Holder)[:\s]+([A-Za-z\s]+)(?:\n|$)', ocr_text, re.IGNORECASE)
        if name_match:
            fields["name"] = name_match.group(1).strip()
        elif mrz_fields.get("surname") or mrz_fields.get("given_names"):
            fields["name"] = f"{mrz_fields.get('surname', '')} {mrz_fields.get('given_names', '')}".strip()

        # Date of Birth
        dob_match = re.search(r'(?:DOB|Date of Birth|Birth Date|Year of Birth)[:\s]+([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{4})', ocr_text, re.IGNORECASE)
        if dob_match:
            fields["date_of_birth"] = dob_match.group(1)
        elif mrz_fields.get("date_of_birth"):
            fields["date_of_birth"] = mrz_fields.get("date_of_birth")

        # Gender
        gender_match = re.search(r'(?:Gender|Sex)[:\s]+(MALE|FEMALE|M|F|OTHER|X)', ocr_text, re.IGNORECASE)
        if gender_match:
            fields["gender"] = gender_match.group(1).upper()
        elif mrz_fields.get("sex"):
            fields["gender"] = mrz_fields.get("sex")

        # Nationality
        nat_match = re.search(r'(?:Nationality|Citizen of)[:\s]+([A-Za-z]+)', ocr_text, re.IGNORECASE)
        if nat_match:
            fields["nationality"] = nat_match.group(1).upper()
        elif mrz_fields.get("nationality"):
            fields["nationality"] = mrz_fields.get("nationality")

        # Address where present
        addr_match = re.search(r'(?:Address|Permanent Address)[:\s]+([^\n]+(?:\n[^\n]+)?)', ocr_text, re.IGNORECASE)
        if addr_match:
            fields["address"] = addr_match.group(1).strip().replace("\n", ", ")

        return fields


class DocumentService:
    """Orchestrates document classification and field extraction."""

    def __init__(self):
        self.extractors: Dict[str, BaseDocumentExtractor] = {
            DocumentTypeEnum.PASSPORT.value: PassportExtractor(),
            DocumentTypeEnum.VISA.value: VisaExtractor(),
            DocumentTypeEnum.NATIONAL_ID.value: NationalIDExtractor(),
        }

    def detect_document_type(self, ocr_text: str, mrz_result: MRZResult) -> str:
        """Classifies document type based on visual text cues and MRZ characteristics."""
        text_upper = ocr_text.upper()
        
        # Passport check: MRZ detected with 'P<' or Passport keywords
        if mrz_result.detected and mrz_result.line1 and mrz_result.line1.startswith("P"):
            return DocumentTypeEnum.PASSPORT.value
            
        if "PASSPORT" in text_upper or "PASSEPORT" in text_upper or "REPUBLIC OF" in text_upper and "TRAVEL DOCUMENT" in text_upper:
            return DocumentTypeEnum.PASSPORT.value
            
        # Visa check
        visa_keywords = ["VISA", "ENTRY PERMIT", "CONTROL NUMBER", "NUMBER OF ENTRIES", "VISA TYPE", "VALID FOR"]
        if any(k in text_upper for k in visa_keywords):
            return DocumentTypeEnum.VISA.value
            
        # National ID check
        id_keywords = ["IDENTITY CARD", "NATIONAL ID", "AADHAAR", "CITIZEN", "DRIVING LICENCE", "IDENTITY NUMBER", "IDENTIFICATION"]
        if any(k in text_upper for k in id_keywords):
            return DocumentTypeEnum.NATIONAL_ID.value

        # If MRZ was detected regardless of document type
        if mrz_result.detected:
            return DocumentTypeEnum.PASSPORT.value

        # Default fallback
        return DocumentTypeEnum.NATIONAL_ID.value

    def process_extraction(
        self,
        requested_type: str,
        ocr_text: str,
        mrz_result: MRZResult,
        mrz_fields: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Determines effective document type and extracts structured fields."""
        if requested_type == DocumentTypeEnum.AUTO.value or not requested_type:
            effective_type = self.detect_document_type(ocr_text, mrz_result)
        else:
            effective_type = requested_type.lower()
            
        extractor = self.extractors.get(effective_type, self.extractors[DocumentTypeEnum.NATIONAL_ID.value])
        fields = extractor.extract_fields(ocr_text, mrz_fields)
        
        return effective_type, fields
