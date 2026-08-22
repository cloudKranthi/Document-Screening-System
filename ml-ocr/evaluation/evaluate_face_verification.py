"""Dataset Evaluation & Calibration Harness for Biometric Face Verification.

Evaluates FaceVerificationService on genuine (same person) and impostor (different person)
document-selfie pairs, computing separation statistics and classification metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

# Adjust sys.path to enable execution both as a script and as a module
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import settings
from app.services.face_verification_service import FaceVerificationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation.evaluate_face_verification")


def _draw_synthetic_face(name: str, seed: int = 42) -> np.ndarray:
    """Draws a synthetic face canvas with eyes, nose, and mouth for benchmark testing."""
    np.random.seed(seed)
    canvas = np.full((300, 300, 3), 245, dtype=np.uint8)

    # Distinct skin tone per seed
    r_val = int(np.clip(210 + np.random.randint(-15, 15), 180, 240))
    g_val = int(np.clip(170 + np.random.randint(-15, 15), 140, 200))
    b_val = int(np.clip(135 + np.random.randint(-15, 15), 110, 165))
    skin_color = (b_val, g_val, r_val)

    # Head oval
    center = (150, 150)
    w_rad = int(65 + np.random.randint(-5, 10))
    h_rad = int(90 + np.random.randint(-5, 10))
    cv2.ellipse(canvas, center, (w_rad, h_rad), 0, 0, 360, skin_color, -1)
    cv2.ellipse(canvas, center, (w_rad, h_rad), 0, 0, 360, (70, 60, 50), 2)

    # Eyes
    eye_y = int(130 + np.random.randint(-4, 5))
    eye_spacing = int(28 + np.random.randint(-4, 5))
    cv2.circle(canvas, (150 - eye_spacing, eye_y), 8, (40, 30, 20), -1)
    cv2.circle(canvas, (150 + eye_spacing, eye_y), 8, (40, 30, 20), -1)

    # Nose
    nose_len = int(25 + np.random.randint(-4, 6))
    cv2.line(canvas, (150, 140), (150, 140 + nose_len), (80, 60, 50), 2)

    # Mouth
    mouth_w = int(25 + np.random.randint(-6, 6))
    cv2.ellipse(canvas, (150, 195), (mouth_w, 10), 0, 0, 180, (60, 40, 40), 2)

    cv2.putText(canvas, name, (15, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    return canvas



def run_face_verification_benchmark(
    genuine_dir: str = "evaluation/face_verification/genuine",
    impostor_dir: str = "evaluation/face_verification/impostor",
    threshold: Optional[float] = None,
    synthetic_samples: int = 4,
    generate_synthetic_if_missing: bool = True
) -> Dict[str, Any]:
    """Runs face verification benchmark on genuine and impostor image pairs."""
    eval_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD
    gen_path = Path(genuine_dir)
    imp_path = Path(impostor_dir)

    # Generate synthetic pairs if missing
    if generate_synthetic_if_missing and (not gen_path.exists() or not imp_path.exists() or len(list(gen_path.glob("*.*"))) == 0):
        gen_path.mkdir(parents=True, exist_ok=True)
        imp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Populating benchmark pairs at '{genuine_dir}' and '{impostor_dir}'...")

        for i in range(synthetic_samples):
            # Genuine pair: Person i (doc) and Person i (selfie)
            doc_img = _draw_synthetic_face(f"PERSON_{i+1}_DOC", seed=100 + i)
            selfie_img = _draw_synthetic_face(f"PERSON_{i+1}_SELFIE", seed=100 + i)

            cv2.imwrite(str(gen_path / f"pair_{i+1}_doc.jpg"), doc_img)
            cv2.imwrite(str(gen_path / f"pair_{i+1}_selfie.jpg"), selfie_img)

            # Impostor pair: Person i (doc) and Person i+10 (selfie)
            imp_doc = _draw_synthetic_face(f"PERSON_{i+1}_DOC", seed=100 + i)
            imp_selfie = _draw_synthetic_face(f"IMPOSTOR_{i+1}", seed=500 + i)

            cv2.imwrite(str(imp_path / f"pair_{i+1}_doc.jpg"), imp_doc)
            cv2.imwrite(str(imp_path / f"pair_{i+1}_selfie.jpg"), imp_selfie)

    service = FaceVerificationService(threshold=eval_threshold)

    # Evaluate Genuine Pairs
    genuine_scores: List[float] = []
    genuine_results: List[Dict[str, Any]] = []
    true_matches = 0
    false_non_matches = 0

    if gen_path.exists():
        pair_ids = sorted(list(set(f.stem.split("_")[1] for f in gen_path.glob("pair_*_doc.jpg"))))
        for pid in pair_ids:
            doc_file = gen_path / f"pair_{pid}_doc.jpg"
            selfie_file = gen_path / f"pair_{pid}_selfie.jpg"
            if doc_file.exists() and selfie_file.exists():
                d_img = cv2.imread(str(doc_file))
                s_img = cv2.imread(str(selfie_file))
                res = service.verify_faces(d_img, s_img)
                score = res.similarity_score if res.similarity_score is not None else 0.0
                genuine_scores.append(score)
                matched = bool(res.match is True)
                if matched:
                    true_matches += 1
                else:
                    false_non_matches += 1

                genuine_results.append({
                    "pair_id": pid,
                    "status": res.status,
                    "similarity_score": score,
                    "match": res.match,
                    "match_band": res.match_band,
                    "ui_color": res.ui_color
                })

    # Evaluate Impostor Pairs
    impostor_scores: List[float] = []
    impostor_results: List[Dict[str, Any]] = []
    true_non_matches = 0
    false_matches = 0

    if imp_path.exists():
        pair_ids = sorted(list(set(f.stem.split("_")[1] for f in imp_path.glob("pair_*_doc.jpg"))))
        for pid in pair_ids:
            doc_file = imp_path / f"pair_{pid}_doc.jpg"
            selfie_file = imp_path / f"pair_{pid}_selfie.jpg"
            if doc_file.exists() and selfie_file.exists():
                d_img = cv2.imread(str(doc_file))
                s_img = cv2.imread(str(selfie_file))
                res = service.verify_faces(d_img, s_img)
                score = res.similarity_score if res.similarity_score is not None else 0.0
                impostor_scores.append(score)
                matched = bool(res.match is True)
                if matched:
                    false_matches += 1
                else:
                    true_non_matches += 1

                impostor_results.append({
                    "pair_id": pid,
                    "status": res.status,
                    "similarity_score": score,
                    "match": res.match,
                    "match_band": res.match_band,
                    "ui_color": res.ui_color
                })


    gen_mean = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    imp_mean = float(np.mean(impostor_scores)) if impostor_scores else 0.0

    report = {
        "threshold": eval_threshold,
        "sample_counts": {
            "genuine_pairs": len(genuine_scores),
            "impostor_pairs": len(impostor_scores),
            "total_pairs": len(genuine_scores) + len(impostor_scores)
        },
        "mean_scores": {
            "genuine_average": round(gen_mean, 4),
            "impostor_average": round(imp_mean, 4),
            "score_separation": round(gen_mean - imp_mean, 4)
        },
        "classification_metrics": {
            "true_matches": true_matches,
            "false_matches_false_accept": false_matches,
            "true_non_matches": true_non_matches,
            "false_non_matches_false_reject": false_non_matches,
            "false_acceptance_rate": round(false_matches / max(1, len(impostor_scores)), 4),
            "false_rejection_rate": round(false_non_matches / max(1, len(genuine_scores)), 4)
        },
        "individual_results": {
            "genuine": genuine_results,
            "impostor": impostor_results
        }
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Biometric Face Verification Performance")
    parser.add_argument("--genuine-dir", default="evaluation/face_verification/genuine", help="Path to genuine pair images")
    parser.add_argument("--impostor-dir", default="evaluation/face_verification/impostor", help="Path to impostor pair images")
    parser.add_argument("--threshold", type=float, default=settings.FACE_MATCH_THRESHOLD, help="Match threshold (default: 0.75)")
    args = parser.parse_args()

    report = run_face_verification_benchmark(
        genuine_dir=args.genuine_dir,
        impostor_dir=args.impostor_dir,
        threshold=args.threshold
    )

    print("\n" + "=" * 65)
    print("      BIOMETRIC FACE VERIFICATION EVALUATION REPORT      ")
    print("=" * 65)
    print(json.dumps(report, indent=2))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
