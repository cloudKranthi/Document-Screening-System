"""Modular OCR abstraction layer supporting Tesseract, PaddleOCR, and Mock engines with normalized confidence."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2

from app.config import settings
from app.models.schemas import OCRRegion
from app.services.confidence_service import ConfidenceService
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """Consolidated OCR extraction result."""
    raw_text: str
    regions: List[OCRRegion]
    average_confidence: float


class BaseOCREngine(ABC):
    """Abstract Base Class for pluggable OCR engines."""
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        pass

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extracts text, bounding boxes, and normalized confidence scores from an image.
        
        Args:
            image: Numpy array (Grayscale or BGR).
            
        Returns:
            OCRResult object with raw_text, regions, and average_confidence.
        """
        pass

    @abstractmethod
    def extract_mrz_text(self, image: np.ndarray, psm: int = 6) -> OCRResult:
        """Extracts text from an MRZ-dedicated image region using MRZ-specific segmentation and whitelist."""
        pass

    @abstractmethod
    def extract_field_text(self, image: np.ndarray, field_type: str, psm: int = 7) -> OCRResult:
        """Extracts text from a targeted small ROI crop (DOB or Gender) with field-specific whitelist and PSM."""
        pass


class TesseractOCREngine(BaseOCREngine):
    """Tesseract OCR Engine adapter using pytesseract."""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        import pytesseract
        self.pytesseract = pytesseract
        if tesseract_cmd:
            self.pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif settings.TESSERACT_CMD:
            self.pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

    @property
    def engine_name(self) -> str:
        return "tesseract"

    def extract_text(self, image: np.ndarray) -> OCRResult:
        try:
            # Use PSM 3 for general document layout
            custom_config = r'--oem 3 --psm 3'
            data = self.pytesseract.image_to_data(
                image,
                config=custom_config,
                output_type=self.pytesseract.Output.DICT
            )
            
            regions: List[OCRRegion] = []
            extracted_words = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = str(data['text'][i]).strip()
                conf_raw = data['conf'][i]
                
                # Filter empty text or unassigned boxes (conf == -1)
                if not text or conf_raw == -1 or conf_raw == '-1':
                    continue
                    
                try:
                    conf_float = float(conf_raw)
                except ValueError:
                    conf_float = 0.0
                    
                norm_conf = ConfidenceService.normalize_score(conf_float, raw_max=100.0)
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                
                regions.append(OCRRegion(text=text, confidence=norm_conf, bbox=bbox))
                extracted_words.append(text)
                
            raw_text = self.pytesseract.image_to_string(image, config=custom_config).strip()
            if not raw_text and extracted_words:
                raw_text = " ".join(extracted_words)
                
            avg_conf = ConfidenceService.calculate_average_confidence(regions)
            return OCRResult(raw_text=raw_text, regions=regions, average_confidence=avg_conf)
            
        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {str(e)}")
            raise RuntimeError(f"Tesseract execution error: {str(e)}")

    def extract_mrz_text(self, image: np.ndarray, psm: int = 6) -> OCRResult:
        """Dedicated MRZ OCR extraction with configurable PSM, strict ICAO whitelist, and dictionary lookup disabled."""
        try:
            # Disables dictionary word lookup to prevent Tesseract from turning '<<<<' into words like 'EERE' or 'KERR'.
            mrz_config = (
                f'--oem 3 --psm {psm} '
                '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< '
                '-c load_system_dawg=0 '
                '-c load_freq_dawg=0 '
                '-c load_punc_dawg=0 '
                '-c load_number_dawg=0 '
                '-c load_bigram_dawg=0'
            )
            data = self.pytesseract.image_to_data(
                image,
                config=mrz_config,
                output_type=self.pytesseract.Output.DICT
            )
            
            regions: List[OCRRegion] = []
            extracted_words = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = str(data['text'][i]).strip()
                conf_raw = data['conf'][i]
                
                if not text or conf_raw == -1 or conf_raw == '-1':
                    continue
                    
                try:
                    conf_float = float(conf_raw)
                except ValueError:
                    conf_float = 0.0
                    
                norm_conf = ConfidenceService.normalize_score(conf_float, raw_max=100.0)
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                
                regions.append(OCRRegion(text=text, confidence=norm_conf, bbox=bbox))
                extracted_words.append(text)
                
            raw_text = self.pytesseract.image_to_string(image, config=mrz_config).strip()
            if not raw_text and extracted_words:
                raw_text = "\n".join(extracted_words)
                
            avg_conf = ConfidenceService.calculate_average_confidence(regions)
            return OCRResult(raw_text=raw_text, regions=regions, average_confidence=avg_conf)
            
        except Exception as e:
            logger.error(f"Tesseract MRZ OCR extraction (PSM {psm}) failed: {str(e)}")
            raise RuntimeError(f"Tesseract MRZ execution error: {str(e)}")

    def extract_field_text(self, image: np.ndarray, field_type: str, psm: int = 7) -> OCRResult:
        """Targeted field OCR extraction for small crops (DOB, Gender) with whitelist and custom PSM."""
        try:
            if field_type.lower() == "dob":
                whitelist = "0123456789/-. "
            elif field_type.lower() == "gender":
                whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/ "
            else:
                whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-. "

            field_config = (
                f'--oem 3 --psm {psm} '
                f'-c tessedit_char_whitelist={whitelist} '
                '-c load_system_dawg=0 '
                '-c load_freq_dawg=0 '
                '-c load_punc_dawg=0 '
                '-c load_number_dawg=0 '
                '-c load_bigram_dawg=0'
            )
            data = self.pytesseract.image_to_data(
                image,
                config=field_config,
                output_type=self.pytesseract.Output.DICT
            )
            regions: List[OCRRegion] = []
            extracted_words = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = str(data['text'][i]).strip()
                conf_raw = data['conf'][i]
                if not text or conf_raw == -1 or conf_raw == '-1':
                    continue
                try:
                    conf_float = float(conf_raw)
                except ValueError:
                    conf_float = 0.0
                norm_conf = ConfidenceService.normalize_score(conf_float, raw_max=100.0)
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                regions.append(OCRRegion(text=text, confidence=norm_conf, bbox=bbox))
                extracted_words.append(text)

            raw_text = self.pytesseract.image_to_string(image, config=field_config).strip()
            if not raw_text and extracted_words:
                raw_text = " ".join(extracted_words)

            avg_conf = ConfidenceService.calculate_average_confidence(regions)
            return OCRResult(raw_text=raw_text, regions=regions, average_confidence=avg_conf)
        except Exception as e:
            logger.error(f"Tesseract field OCR extraction failed ({field_type}, PSM {psm}): {str(e)}")
            raise RuntimeError(f"Tesseract field execution error: {str(e)}")


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR Engine adapter."""
    
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except ImportError:
            raise ImportError("PaddleOCR is not installed. Please install 'paddlepaddle' and 'paddleocr'.")

    @property
    def engine_name(self) -> str:
        return "paddleocr"

    def extract_text(self, image: np.ndarray) -> OCRResult:
        try:
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            result = self.ocr.ocr(image_rgb, cls=True)
            regions: List[OCRRegion] = []
            extracted_lines = []
            
            if result and result[0]:
                for line in result[0]:
                    box_points, (text, conf_raw) = line
                    text = text.strip()
                    if not text:
                        continue
                    norm_conf = ConfidenceService.normalize_score(float(conf_raw), raw_max=1.0)
                    
                    # Compute [x1, y1, x2, y2] from 4 box points
                    xs = [p[0] for p in box_points]
                    ys = [p[1] for p in box_points]
                    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    
                    regions.append(OCRRegion(text=text, confidence=norm_conf, bbox=bbox))
                    extracted_lines.append(text)
                    
            raw_text = "\n".join(extracted_lines)
            avg_conf = ConfidenceService.calculate_average_confidence(regions)
            return OCRResult(raw_text=raw_text, regions=regions, average_confidence=avg_conf)
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {str(e)}")
            raise RuntimeError(f"PaddleOCR execution error: {str(e)}")

    def extract_mrz_text(self, image: np.ndarray, psm: int = 6) -> OCRResult:
        res = self.extract_text(image)
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        cleaned_regions = []
        for r in res.regions:
            clean_text = "".join(c for c in r.text.upper() if c in allowed or c in " \t")
            if clean_text:
                cleaned_regions.append(OCRRegion(text=clean_text, confidence=r.confidence, bbox=r.bbox))
        clean_raw = "\n".join(r.text for r in cleaned_regions)
        return OCRResult(raw_text=clean_raw, regions=cleaned_regions, average_confidence=res.average_confidence)

    def extract_field_text(self, image: np.ndarray, field_type: str, psm: int = 7) -> OCRResult:
        res = self.extract_text(image)
        if field_type.lower() == "dob":
            allowed = set("0123456789/-. ")
        elif field_type.lower() == "gender":
            allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/ ")
        else:
            allowed = None
        if allowed is not None:
            cleaned_regions = []
            for r in res.regions:
                clean_text = "".join(c for c in r.text if c in allowed)
                if clean_text:
                    cleaned_regions.append(OCRRegion(text=clean_text, confidence=r.confidence, bbox=r.bbox))
            clean_raw = " ".join(r.text for r in cleaned_regions)
            return OCRResult(raw_text=clean_raw, regions=cleaned_regions, average_confidence=res.average_confidence)
        return res


class MockOCREngine(BaseOCREngine):
    """Deterministic Mock OCR Engine for unit testing and standalone verification."""
    
    def __init__(self, predefined_result: Optional[OCRResult] = None):
        self.predefined_result = predefined_result

    @property
    def engine_name(self) -> str:
        return "mock"

    def set_predefined_result(self, result: OCRResult) -> None:
        self.predefined_result = result

    def extract_text(self, image: np.ndarray) -> OCRResult:
        if self.predefined_result is not None:
            return self.predefined_result
            
        # Default mock response
        default_regions = [
            OCRRegion(text="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", confidence=0.98, bbox=[50, 400, 750, 430]),
            OCRRegion(text="L898902C36UTO7408122F1204159ZE184226B<<<<<10", confidence=0.97, bbox=[50, 440, 750, 470]),
        ]
        raw_text = "PASSPORT\n" + "\n".join(r.text for r in default_regions)
        avg_conf = ConfidenceService.calculate_average_confidence(default_regions)
        return OCRResult(raw_text=raw_text, regions=default_regions, average_confidence=avg_conf)

    def extract_mrz_text(self, image: np.ndarray, psm: int = 6) -> OCRResult:
        if self.predefined_result is not None:
            return self.predefined_result
        return self.extract_text(image)

    def extract_field_text(self, image: np.ndarray, field_type: str, psm: int = 7) -> OCRResult:
        if self.predefined_result is not None:
            return self.predefined_result
        return self.extract_text(image)


class OCRService:
    """Main OCR Service orchestrating engine selection and execution."""

    def __init__(self, engine: Optional[BaseOCREngine] = None):
        self._engine = engine or self._initialize_engine()

    def _initialize_engine(self) -> BaseOCREngine:
        preferred = settings.OCR_ENGINE.lower()
        
        if preferred == "mock":
            logger.info("Using Mock OCR engine.")
            return MockOCREngine()
            
        if preferred == "paddleocr":
            try:
                engine = PaddleOCREngine()
                logger.info("PaddleOCR engine initialized successfully.")
                return engine
            except Exception as e:
                logger.warning(f"PaddleOCR unavailable ({str(e)}), attempting Tesseract fallback.")
                
        # Try Tesseract
        try:
            engine = TesseractOCREngine()
            logger.info("Tesseract OCR engine initialized.")
            return engine
        except Exception as e:
            logger.warning(f"Tesseract OCR unavailable ({str(e)}), falling back to Mock engine.")
            return MockOCREngine()

    @property
    def active_engine_name(self) -> str:
        return self._engine.engine_name

    def extract(self, image: np.ndarray) -> OCRResult:
        """Executes general OCR extraction on the processed document image."""
        return self._engine.extract_text(image)

    def extract_mrz(self, image: np.ndarray, psm: int = 6) -> OCRResult:
        """Executes specialized MRZ OCR extraction with restricted whitelist, line segmentation, and dictionary disabled."""
        return self._engine.extract_mrz_text(image, psm=psm)

    def extract_field(self, image: np.ndarray, field_type: str, psm: int = 7) -> OCRResult:
        """Executes targeted field OCR extraction with whitelist, line segmentation, and custom PSM."""
        return self._engine.extract_field_text(image, field_type=field_type, psm=psm)



