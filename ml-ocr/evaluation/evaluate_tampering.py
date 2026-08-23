"""Evaluation utility script for document tampering detection on labeled clean and tampered datasets."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.services.tampering_service import TamperingService
from app.utils.logger import get_logger

logger = get_logger("evaluation.evaluate_tampering")


def evaluate_dataset(
    clean_dir: str = "evaluation/data/clean",
    tampered_dir: str = "evaluation/data/tampered",
    threshold: float = 0.30,
    generate_synthetic_if_missing: bool = True,
    synthetic_samples: int = 5
) -> Dict[str, Any]:
    """Evaluates tampering detector across clean and tampered document image directories.
    
    Returns structured metrics including sample counts, individual scores, means, risk category distributions,
    false positives, false negatives, and confusion matrix.
    """
    clean_path = Path(clean_dir)
    tampered_path = Path(tampered_dir)

    # Generate minimal synthetic dataset if directories are missing or empty
    if generate_synthetic_if_missing and (not clean_path.exists() or not tampered_path.exists() or len(list(clean_path.glob("*.*"))) == 0):
        logger.info(f"Populating benchmark directories at '{clean_dir}' and '{tampered_dir}' with {synthetic_samples} synthetic samples each...")
        clean_path.mkdir(parents=True, exist_ok=True)
        tampered_path.mkdir(parents=True, exist_ok=True)

        for i in range(synthetic_samples):
            # Base background document pre-compressed
            img = np.full((600, 800, 3), 245, dtype=np.uint8)
            cv2.rectangle(img, (40, 40), (760, 560), (220, 220, 220), 2)
            cv2.putText(img, f"PASSPORT {i+1}", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
            cv2.putText(img, "REPUBLIC OF TRAVEL", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
            cv2.putText(img, "DOB: 12/08/1974", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
            
            # Clean baseline encoded at controlled JPEG quality
            _, clean_enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            clean_file = clean_path / f"clean_sample_{i+1}.jpg"
            clean_file.write_bytes(clean_enc.tobytes())

            # Tampered sample: take pre-compressed base (at Q=70), paste uncompressed spliced patch, and save
            _, base_enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            base_dec = cv2.imdecode(base_enc, cv2.IMREAD_COLOR)
            patch = np.full((100, 220, 3), 200, dtype=np.uint8)
            cv2.putText(patch, "FORGED 999", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            patch_noise = np.random.normal(0, 18.0, patch.shape).astype(np.int16)
            patch = np.clip(patch.astype(np.int16) + patch_noise, 0, 255).astype(np.uint8)
            base_dec[160:260, 50:270] = patch

            _, tampered_enc = cv2.imencode(".jpg", base_dec, [cv2.IMWRITE_JPEG_QUALITY, 92])
            tampered_file = tampered_path / f"tampered_sample_{i+1}.jpg"
            tampered_file.write_bytes(tampered_enc.tobytes())



    tampering_srv = TamperingService()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    # Process Clean Samples
    clean_scores: List[Dict[str, Any]] = []
    clean_files = [f for f in clean_path.glob("*.*") if f.suffix.lower() in valid_exts] if clean_path.exists() else []

    for fpath in clean_files:
        try:
            with open(fpath, "rb") as f:
                raw_bytes = f.read()
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            res = tampering_srv.analyze_document(image_bytes=raw_bytes, document_image=img)
            clean_scores.append({
                "filename": fpath.name,
                "tampering_risk_score": res.tampering_risk_score,
                "risk_level": res.risk_level,
                "evidence_coverage": res.evidence_coverage,
                "predicted_tampered": res.tampering_risk_score >= threshold
            })
        except Exception as e:
            logger.error(f"Failed processing clean sample {fpath.name}: {str(e)}")

    # Process Tampered Samples
    tampered_scores: List[Dict[str, Any]] = []
    tampered_files = [f for f in tampered_path.glob("*.*") if f.suffix.lower() in valid_exts] if tampered_path.exists() else []

    for fpath in tampered_files:
        try:
            with open(fpath, "rb") as f:
                raw_bytes = f.read()
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            res = tampering_srv.analyze_document(image_bytes=raw_bytes, document_image=img)
            tampered_scores.append({
                "filename": fpath.name,
                "tampering_risk_score": res.tampering_risk_score,
                "risk_level": res.risk_level,
                "evidence_coverage": res.evidence_coverage,
                "predicted_tampered": res.tampering_risk_score >= threshold
            })
        except Exception as e:
            logger.error(f"Failed processing tampered sample {fpath.name}: {str(e)}")

    # Compute aggregate statistics
    n_clean = len(clean_scores)
    n_tampered = len(tampered_scores)

    clean_val_scores = [s["tampering_risk_score"] for s in clean_scores]
    tampered_val_scores = [s["tampering_risk_score"] for s in tampered_scores]

    avg_clean_score = round(float(np.mean(clean_val_scores)), 4) if clean_val_scores else 0.0
    avg_tampered_score = round(float(np.mean(tampered_val_scores)), 4) if tampered_val_scores else 0.0

    # Risk level distributions
    clean_counts = {
        "LOW": sum(1 for s in clean_scores if s["risk_level"] == "LOW"),
        "MEDIUM": sum(1 for s in clean_scores if s["risk_level"] == "MEDIUM"),
        "HIGH": sum(1 for s in clean_scores if s["risk_level"] == "HIGH"),
    }
    tampered_counts = {
        "LOW": sum(1 for s in tampered_scores if s["risk_level"] == "LOW"),
        "MEDIUM": sum(1 for s in tampered_scores if s["risk_level"] == "MEDIUM"),
        "HIGH": sum(1 for s in tampered_scores if s["risk_level"] == "HIGH"),
    }

    # Confusion matrix
    # Clean is Negative (0), Tampered is Positive (1)
    tn = sum(1 for s in clean_scores if not s["predicted_tampered"])
    fp = sum(1 for s in clean_scores if s["predicted_tampered"])
    tp = sum(1 for s in tampered_scores if s["predicted_tampered"])
    fn = sum(1 for s in tampered_scores if not s["predicted_tampered"])

    report = {
        "threshold": threshold,
        "sample_counts": {
            "clean": n_clean,
            "tampered": n_tampered,
            "total": n_clean + n_tampered
        },
        "mean_scores": {
            "clean_average": avg_clean_score,
            "tampered_average": avg_tampered_score,
            "score_separation": round(avg_tampered_score - avg_clean_score, 4)
        },
        "risk_level_counts": {
            "clean_samples": clean_counts,
            "tampered_samples": tampered_counts
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "true_positives": tp,
            "false_negatives": fn
        },
        "individual_results": {
            "clean": clean_scores,
            "tampered": tampered_scores
        }
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate document tampering detection performance on clean and tampered datasets.")
    parser.add_argument("--clean-dir", type=str, default="evaluation/data/clean", help="Path to directory containing clean document images.")
    parser.add_argument("--tampered-dir", type=str, default="evaluation/data/tampered", help="Path to directory containing tampered document images.")
    parser.add_argument("--threshold", type=float, default=0.30, help="Decision threshold for classification (default: 0.30).")
    parser.add_argument("--no-synthetic", action="store_true", help="Do not generate synthetic samples if directories are missing.")

    args = parser.parse_args()

    report = evaluate_dataset(
        clean_dir=args.clean_dir,
        tampered_dir=args.tampered_dir,
        threshold=args.threshold,
        generate_synthetic_if_missing=not args.no_synthetic
    )

    print("\n" + "=" * 65)
    print("      DOCUMENT TAMPERING & ELA FORENSICS EVALUATION REPORT      ")
    print("=" * 65)
    print(json.dumps(report, indent=2))
    print("=" * 65)


if __name__ == "__main__":
    main()
