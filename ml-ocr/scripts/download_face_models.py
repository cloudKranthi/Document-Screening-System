"""Deployment-time script to download pretrained OpenCV Zoo face models (YuNet and SFace).

Downloads the exact ONNX model weights required for biometric face detection and verification:
1. YuNet (face_detection_yunet_2023mar.onnx - ~232 KB)
2. SFace (face_recognition_sface_2021dec.onnx - ~37 MB)

Ensures models exist, verifies non-empty file size, and fails with a non-zero exit code on failure.
"""

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_face_models")

# Trusted OpenCV Zoo model definitions
MODEL_DEFINITIONS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_size_bytes": 100_000,  # ~232 KB expected
        "description": "YuNet Face Detector (OpenCV Zoo)"
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "min_size_bytes": 30_000_000,  # ~37 MB expected
        "description": "SFace Face Recognizer (OpenCV Zoo)"
    }
}


def download_model(name: str, config: Dict[str, Any], target_dir: Path) -> bool:
    """Downloads a single ONNX model file and validates its size."""
    filename = config["filename"]
    url = config["url"]
    min_size = config["min_size_bytes"]
    desc = config["description"]
    target_path = target_dir / filename

    if target_path.exists():
        actual_size = target_path.stat().st_size
        if actual_size >= min_size:
            logger.info(f"[{name}] {desc} already exists at {target_path} ({actual_size:,} bytes). Skipping download.")
            return True
        else:
            logger.warning(f"[{name}] {desc} at {target_path} is incomplete ({actual_size:,} < {min_size:,} bytes). Re-downloading...")
            target_path.unlink(missing_ok=True)

    logger.info(f"[{name}] Downloading {desc} from {url} to {target_path}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (SIH-Document-Screening-Deployment)"}
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            with open(target_path, "wb") as out_file:
                downloaded = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

        actual_size = target_path.stat().st_size
        if actual_size < min_size:
            logger.error(f"[{name}] Download validation failed: file size {actual_size:,} bytes is below expected {min_size:,} bytes.")
            target_path.unlink(missing_ok=True)
            return False

        logger.info(f"[{name}] Download complete and validated: {target_path} ({actual_size:,} bytes).")
        return True
    except Exception as e:
        logger.error(f"[{name}] Failed to download {desc} from {url}: {e}")
        target_path.unlink(missing_ok=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ONNX models for Biometric Face Verification")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("FACE_MODEL_DIR", "app/models/weights"),
        help="Directory where ONNX model weights should be saved (default: app/models/weights)"
    )
    args = parser.parse_args()

    target_dir = Path(args.model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring face models in directory: {target_dir.resolve()}")

    all_successful = True
    for model_name, config in MODEL_DEFINITIONS.items():
        success = download_model(model_name, config, target_dir)
        if not success:
            all_successful = False

    if not all_successful:
        logger.error("One or more face model downloads failed. Deployment cannot continue.")
        return 1

    logger.info("All face models successfully downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
