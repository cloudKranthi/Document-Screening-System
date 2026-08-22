"""Unit tests for confidence normalization and aggregate calculation."""

import pytest
from app.models.schemas import OCRRegion
from app.services.confidence_service import ConfidenceService


class TestConfidenceService:
    """Unit tests for ConfidenceService."""

    def test_normalize_score(self):
        # 0-100 scale normalization
        assert ConfidenceService.normalize_score(95.0, raw_max=100.0) == 0.95
        assert ConfidenceService.normalize_score(100.0, raw_max=100.0) == 1.0
        assert ConfidenceService.normalize_score(0.0, raw_max=100.0) == 0.0
        
        # Already normalized [0, 1] scale
        assert ConfidenceService.normalize_score(0.85, raw_max=1.0) == 0.85
        
        # Negative / boundary clamping
        assert ConfidenceService.normalize_score(-5.0) == 0.0
        assert ConfidenceService.normalize_score(150.0, raw_max=100.0) == 1.0
        assert ConfidenceService.normalize_score(None) == 0.0

    def test_calculate_average_confidence_with_regions(self):
        regions = [
            OCRRegion(text="PASSPORT", confidence=0.90, bbox=[0, 0, 10, 10]),  # 8 chars
            OCRRegion(text="REPUBLIC", confidence=0.80, bbox=[0, 10, 10, 20]), # 8 chars
        ]
        avg = ConfidenceService.calculate_average_confidence(regions)
        assert avg == 0.85
        assert 0.0 <= avg <= 1.0

    def test_calculate_average_confidence_weighted_by_length(self):
        regions = [
            OCRRegion(text="A", confidence=0.50, bbox=[0, 0, 1, 1]),         # 1 char
            OCRRegion(text="LONGTEXTHERE", confidence=1.00, bbox=[0, 0, 10, 1]), # 12 chars
        ]
        # (0.50 * 1 + 1.00 * 12) / 13 = 12.5 / 13 = 0.9615
        avg = ConfidenceService.calculate_average_confidence(regions)
        assert 0.95 <= avg <= 0.97

    def test_calculate_average_confidence_empty(self):
        assert ConfidenceService.calculate_average_confidence([]) == 0.0
        
    def test_calculate_average_confidence_whitespace_only(self):
        regions = [
            OCRRegion(text="   ", confidence=0.90, bbox=[0, 0, 10, 10]),
            OCRRegion(text="", confidence=0.80, bbox=[0, 0, 10, 10]),
        ]
        assert ConfidenceService.calculate_average_confidence(regions) == 0.0
