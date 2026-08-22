"""Unit tests for OCR engine abstraction and region extraction."""

import numpy as np
import pytest
from app.models.schemas import OCRRegion
from app.services.ocr_service import MockOCREngine, OCRResult, OCRService


class TestOCREngineAbstraction:
    """Tests for OCR service abstraction and adapters."""

    def test_mock_ocr_engine_default(self):
        engine = MockOCREngine()
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        result = engine.extract_text(dummy_img)
        
        assert isinstance(result, OCRResult)
        assert len(result.regions) >= 2
        assert result.average_confidence >= 0.9
        assert "P<UTOERIKSSON" in result.raw_text

    def test_mock_ocr_engine_custom(self):
        custom_regions = [
            OCRRegion(text="VISA", confidence=0.92, bbox=[10, 10, 50, 30]),
            OCRRegion(text="UNITED STATES", confidence=0.88, bbox=[60, 10, 150, 30]),
        ]
        custom_result = OCRResult(
            raw_text="VISA UNITED STATES",
            regions=custom_regions,
            average_confidence=0.90
        )
        engine = MockOCREngine(predefined_result=custom_result)
        result = engine.extract_text(np.zeros((50, 50), dtype=np.uint8))
        
        assert result.raw_text == "VISA UNITED STATES"
        assert len(result.regions) == 2
        assert result.average_confidence == 0.90

    def test_ocr_service_initialization(self):
        mock_engine = MockOCREngine()
        service = OCRService(engine=mock_engine)
        assert service.active_engine_name == "mock"
        
        res = service.extract(np.zeros((50, 50), dtype=np.uint8))
        assert isinstance(res, OCRResult)

    def test_mock_ocr_engine_extract_mrz(self):
        engine = MockOCREngine()
        dummy_img = np.zeros((100, 300), dtype=np.uint8)
        mrz_res = engine.extract_mrz_text(dummy_img)
        assert isinstance(mrz_res, OCRResult)
        assert "P<UTOERIKSSON" in mrz_res.raw_text
        assert len(mrz_res.regions) >= 2

    def test_ocr_service_extract_mrz_dispatch(self):
        mock_engine = MockOCREngine()
        service = OCRService(engine=mock_engine)
        res = service.extract_mrz(np.zeros((50, 200), dtype=np.uint8))
        assert isinstance(res, OCRResult)
        assert res.average_confidence >= 0.90

