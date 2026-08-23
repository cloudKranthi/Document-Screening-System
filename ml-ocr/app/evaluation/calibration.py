"""Calibration and benchmark evaluation harness for document tampering detection."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.models.schemas import TamperingResult
from app.services.tampering_service import TamperingService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TamperingCalibrationHarness:
    """Harness to evaluate tampering detection performance across labeled dataset folders (clean vs tampered)."""

    def __init__(self, tampering_service: Optional[TamperingService] = None, default_threshold: float = 0.35):
        self.service = tampering_service or TamperingService()
        self.default_threshold = default_threshold

    def evaluate_directory(
        self,
        dataset_dir: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluates a directory containing 'clean/' and 'tampered/' subdirectories.
        
        Returns:
            Dictionary containing class sample counts, mean risk scores, confusion matrix,
            false positive rate (FPR), false negative rate (FNR), precision, recall, and F1 score.
        """
        th = threshold if threshold is not None else self.default_threshold
        root_path = Path(dataset_dir)
        clean_dir = root_path / "clean"
        tampered_dir = root_path / "tampered"

        if not clean_dir.exists() or not tampered_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory '{dataset_dir}' must contain both 'clean/' and 'tampered/' subdirectories."
            )

        clean_files = [f for f in clean_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')]
        tampered_files = [f for f in tampered_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')]

        clean_scores: List[float] = []
        tampered_scores: List[float] = []

        # Evaluate clean samples
        for fpath in clean_files:
            try:
                img_bytes = fpath.read_bytes()
                res = self.service.analyze_document(image_bytes=img_bytes)
                clean_scores.append(res.tampering_risk_score)
            except Exception as e:
                logger.error(f"Failed evaluating clean sample {fpath.name}: {str(e)}")

        # Evaluate tampered samples
        for fpath in tampered_files:
            try:
                img_bytes = fpath.read_bytes()
                res = self.service.analyze_document(image_bytes=img_bytes)
                tampered_scores.append(res.tampering_risk_score)
            except Exception as e:
                logger.error(f"Failed evaluating tampered sample {fpath.name}: {str(e)}")

        # Compute metrics
        n_clean = len(clean_scores)
        n_tampered = len(tampered_scores)

        mean_clean_score = float(np.mean(clean_scores)) if clean_scores else 0.0
        mean_tampered_score = float(np.mean(tampered_scores)) if tampered_scores else 0.0

        # Confusion matrix at threshold:
        # True Negative: clean <= threshold
        # False Positive: clean > threshold
        # False Negative: tampered <= threshold
        # True Positive: tampered > threshold
        tn = sum(1 for s in clean_scores if s <= th)
        fp = sum(1 for s in clean_scores if s > th)
        fn = sum(1 for s in tampered_scores if s <= th)
        tp = sum(1 for s in tampered_scores if s > th)

        fpr = (fp / n_clean) if n_clean > 0 else 0.0
        fnr = (fn / n_tampered) if n_tampered > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics = {
            "threshold": th,
            "sample_counts": {
                "clean": n_clean,
                "tampered": n_tampered,
                "total": n_clean + n_tampered
            },
            "mean_risk_scores": {
                "clean": round(mean_clean_score, 4),
                "tampered": round(mean_tampered_score, 4),
                "score_separation_delta": round(mean_tampered_score - mean_clean_score, 4)
            },
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            },
            "rates": {
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        }
        return metrics

    @staticmethod
    def generate_synthetic_benchmark_dataset(
        output_dir: str,
        num_clean: int = 5,
        num_tampered: int = 5
    ) -> Dict[str, int]:
        """Generates a small paired synthetic dataset (clean vs tampered) for repeatable local calibration and unit tests."""
        out_path = Path(output_dir)
        clean_dir = out_path / "clean"
        tampered_dir = out_path / "tampered"
        clean_dir.mkdir(parents=True, exist_ok=True)
        tampered_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_clean):
            # Clean document: white background with text-like strokes and uniform texture
            img = np.full((400, 600, 3), 245, dtype=np.uint8)
            cv2.putText(img, f"IDENTITY DOCUMENT #{1000 + i}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
            cv2.putText(img, "NAME: CITIZEN TEST", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.putText(img, "DATE OF BIRTH: 15/08/1990", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.putText(img, f"DOC NUMBER: ID-{9000 + i}", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            # Add uniform subtle noise
            noise = np.random.normal(0, 3.0, img.shape).astype(np.int16)
            noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(clean_dir / f"clean_sample_{i+1}.jpg"), noisy_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        for i in range(num_tampered):
            # Base document
            img = np.full((400, 600, 3), 245, dtype=np.uint8)
            cv2.putText(img, f"IDENTITY DOCUMENT #{2000 + i}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
            cv2.putText(img, "NAME: FORGED CITIZEN", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.putText(img, "DATE OF BIRTH: 01/01/1980", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.putText(img, f"DOC NUMBER: ID-{8000 + i}", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            
            # Type of tampering: copy-move clone or spliced patch
            if i % 2 == 0:
                # Copy-move patch: clone a region
                patch = img[40:120, 40:200].copy()
                img[240:320, 340:500] = patch
            else:
                # Spliced patch with high noise residual and JPEG compression mismatch
                patch = np.random.randint(0, 255, (70, 160, 3), dtype=np.uint8)
                cv2.putText(patch, "FORGERY", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                img[200:270, 350:510] = patch

            cv2.imwrite(str(tampered_dir / f"tampered_sample_{i+1}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return {"clean_generated": num_clean, "tampered_generated": num_tampered}
