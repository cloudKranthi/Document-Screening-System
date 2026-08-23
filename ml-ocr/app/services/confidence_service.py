"""Confidence calculation and normalization service for OCR extractions."""

from typing import List
from app.models.schemas import OCRRegion


class ConfidenceService:
    """Service to normalize and compute aggregate OCR confidence scores."""

    @staticmethod
    def normalize_score(score: float, raw_max: float = 100.0) -> float:
        """Normalizes a raw confidence score to the range [0.0, 1.0]."""
        if score is None or score < 0:
            return 0.0
        # If score is already in [0.0, 1.0]
        if score <= 1.0 and raw_max != 1.0:
            return max(0.0, min(1.0, float(score)))
        normalized = float(score) / raw_max
        return max(0.0, min(1.0, round(normalized, 4)))

    @staticmethod
    def calculate_average_confidence(regions: List[OCRRegion]) -> float:
        """Calculates weighted average confidence across OCR regions weighted by text length.
        
        Args:
            regions: List of OCRRegion objects.
            
        Returns:
            Normalized float between 0.0 and 1.0. If no regions, returns 0.0.
        """
        valid_regions = [r for r in regions if r.text and r.text.strip() and r.confidence >= 0.0]
        if not valid_regions:
            return 0.0
            
        total_weight = sum(len(r.text.strip()) for r in valid_regions)
        if total_weight == 0:
            avg = sum(r.confidence for r in valid_regions) / len(valid_regions)
            return round(avg, 4)
            
        weighted_sum = sum(r.confidence * len(r.text.strip()) for r in valid_regions)
        return round(weighted_sum / total_weight, 4)
