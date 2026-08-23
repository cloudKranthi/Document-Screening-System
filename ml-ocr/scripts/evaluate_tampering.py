"""CLI script to run tampering evaluation and calibration harness on a directory of labeled samples."""

import argparse
import json
import sys
from pathlib import Path

# Add ml-ocr to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.calibration import TamperingCalibrationHarness
from app.services.tampering_service import TamperingService
from app.utils.logger import get_logger

logger = get_logger("scripts.evaluate_tampering")


def main():
    parser = argparse.ArgumentParser(description="Evaluate document tampering detection performance on a dataset.")
    parser.add_argument("--dataset-dir", type=str, default="data/benchmark", help="Directory containing 'clean/' and 'tampered/' subdirectories.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Tampering risk classification threshold (default: 0.35).")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic test dataset if directory is missing or empty.")
    parser.add_argument("--samples", type=int, default=10, help="Number of synthetic samples to generate per class if --generate-synthetic is used.")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if args.generate_synthetic or not dataset_path.exists():
        logger.info(f"Generating synthetic benchmark dataset in '{args.dataset_dir}' ({args.samples} samples per class)...")
        TamperingCalibrationHarness.generate_synthetic_benchmark_dataset(
            output_dir=args.dataset_dir,
            num_clean=args.samples,
            num_tampered=args.samples
        )

    logger.info(f"Evaluating dataset in '{args.dataset_dir}' at threshold {args.threshold}...")
    harness = TamperingCalibrationHarness(default_threshold=args.threshold)
    metrics = harness.evaluate_directory(dataset_dir=args.dataset_dir, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("      DOCUMENT TAMPERING DETECTION BENCHMARK REPORT      ")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
