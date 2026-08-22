"""Tampering detection utility algorithms and signal processors."""

import io
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import ExifTags, Image

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Known digital photo editing & manipulation software identifiers
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop",
    "gimp",
    "canva",
    "paint.net",
    "picsart",
    "snapseed",
    "pixelmator",
    "affinity photo",
    "photopea",
    "seashore",
    "pixlr",
]


def compute_ela_residual(image: np.ndarray, quality: int = 90) -> Tuple[np.ndarray, float, float]:
    """Computes Error Level Analysis (ELA) residual by comparing image to a fixed-quality JPEG recompression.
    
    Returns:
        Tuple of (ela_diff_gray, global_mean_error, global_std_error).
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), 0.0, 0.0

    # Ensure 3-channel BGR
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    # Recompress in memory as JPEG at specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc_buf = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0, 0.0

    recompressed = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
    if recompressed is None:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0, 0.0

    # Absolute difference
    diff = cv2.absdiff(bgr, recompressed)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mean_err = float(np.mean(diff_gray))
    std_err = float(np.std(diff_gray))

    return diff_gray, mean_err, std_err


def compute_noise_residual(gray: np.ndarray) -> np.ndarray:
    """Estimates local high-frequency noise residual by subtracting a 3x3 median-filtered image from original."""
    if gray is None or gray.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    # 3x3 median filter suppresses high-frequency sensor noise
    denoised = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)
    return residual


def compute_local_gradient_and_sharpness(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes gradient magnitude (Sobel) and local Laplacian sharpness representation.
    
    Returns:
        Tuple of (gradient_magnitude, laplacian_abs).
    """
    if gray is None or gray.size == 0:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    return grad_mag, lap_abs


def detect_orb_copy_move_clusters(
    image: np.ndarray,
    min_cluster_size: int = 5,
    min_dist: float = 35.0,
    max_features: int = 1500
) -> List[Dict[str, Any]]:
    """Detects duplicated/cloned regions within the same image using ORB descriptors and spatial displacement clustering.
    
    Args:
        image: BGR or Grayscale input image.
        min_cluster_size: Minimum number of matched keypoints sharing consistent translation vector to form a cluster.
        min_dist: Minimum spatial Euclidean distance (px) between matched keypoints to prevent near-neighbor false alarms.
        max_features: Max ORB keypoints to extract.
        
    Returns:
        List of cluster records containing source/target bounding boxes, match count, and translation vector.
    """
    if image is None or image.size == 0:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=12, scaleFactor=1.2, nlevels=4)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 25:
        return []

    # Brute-force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(descriptors, descriptors, k=3)
    except Exception as e:
        logger.debug(f"ORB KNN matching failed: {str(e)}")
        return []

    candidate_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], float, float]] = []

    for match_group in raw_matches:
        if len(match_group) < 2:
            continue

        # Exclude self-match
        m1 = match_group[0]
        m2 = match_group[1]

        target_match = m1 if m1.queryIdx != m1.trainIdx else m2
        if target_match.queryIdx == target_match.trainIdx:
            if len(match_group) >= 3:
                target_match = match_group[2]
            else:
                continue

        # Ratio test against second-best distinct candidate
        ref_match = match_group[2] if (target_match == m1 and len(match_group) >= 3) else (match_group[1] if target_match != m1 else None)
        if ref_match and ref_match.distance > 0:
            if target_match.distance > 0.80 * ref_match.distance:
                continue

        pt1 = keypoints[target_match.queryIdx].pt
        pt2 = keypoints[target_match.trainIdx].pt

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = math.hypot(dx, dy)

        # Reject near-neighbor/self matches within min_dist
        if dist < min_dist:
            continue

        # Standardize vector direction so (dx, dy) and (-dx, -dy) group together
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
            pt1, pt2 = pt2, pt1

        candidate_pairs.append((pt1, pt2, dx, dy))

    if len(candidate_pairs) < min_cluster_size:
        return []

    # Cluster spatial displacement vectors
    vector_bins: Dict[Tuple[int, int], List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
    bin_size = 20.0  # 20 pixel binning for translation tolerance

    for pt1, pt2, dx, dy in candidate_pairs:
        bin_x = int(round(dx / bin_size))
        bin_y = int(round(dy / bin_size))
        key = (bin_x, bin_y)
        if key not in vector_bins:
            vector_bins[key] = []
        vector_bins[key].append((pt1, pt2))

    detected_clusters = []
    for (bx, by), pairs in vector_bins.items():
        if len(pairs) >= min_cluster_size:
            pts_src = [p[0] for p in pairs]
            pts_dst = [p[1] for p in pairs]

            src_x1 = max(0, int(min(p[0] for p in pts_src) - 15))
            src_y1 = max(0, int(min(p[1] for p in pts_src) - 15))
            src_x2 = min(gray.shape[1], int(max(p[0] for p in pts_src) + 15))
            src_y2 = min(gray.shape[0], int(max(p[1] for p in pts_src) + 15))

            dst_x1 = max(0, int(min(p[0] for p in pts_dst) - 15))
            dst_y1 = max(0, int(min(p[1] for p in pts_dst) - 15))
            dst_x2 = min(gray.shape[1], int(max(p[0] for p in pts_dst) + 15))
            dst_y2 = min(gray.shape[0], int(max(p[1] for p in pts_dst) + 15))
            # Ensure regions are spatially separated and have non-trivial 2D area (not 1D repeating font lines like <<<<<)
            src_w = src_x2 - src_x1
            src_h = src_y2 - src_y1
            dst_w = dst_x2 - dst_x1
            dst_h = dst_y2 - dst_y1


            is_1d_text_line = (src_h < 40 or dst_h < 40 or (src_w / max(1, src_h) > 2.5))
            min_req = max(10, min_cluster_size * 2) if is_1d_text_line else min_cluster_size

            overlap = not (src_x2 < dst_x1 or src_x1 > dst_x2 or src_y2 < dst_y1 or src_y1 > dst_y2)
            if (not overlap and len(pairs) >= min_req) or (len(pairs) >= min_cluster_size * 3):
                cluster_rec = {
                    "match_count": len(pairs),
                    "vector": [round(bx * bin_size, 1), round(by * bin_size, 1)],
                    "source_region": [src_x1, src_y1, src_x2, src_y2],
                    "target_region": [dst_x1, dst_y1, dst_x2, dst_y2],
                }
                detected_clusters.append(cluster_rec)


    # Sort by match count descending
    detected_clusters.sort(key=lambda c: -c["match_count"])
    return detected_clusters



def extract_image_metadata_signals(image_bytes: bytes) -> Dict[str, Any]:
    """Inspects EXIF / image container metadata for software tags, editor footprints, or timestamp discrepancies.
    
    Rules:
        - Absence of metadata is common and scores 0.0.
        - Editing software presence is treated as a weak indicator (score ~0.35-0.50).
    """
    res: Dict[str, Any] = {
        "has_metadata": False,
        "software": None,
        "is_editing_software": False,
        "detected_software_name": None,
        "timestamps": {},
        "timestamp_mismatch": False,
        "score": 0.0,
        "explanation": "No suspicious metadata anomalies detected."
    }

    if not image_bytes:
        return res

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return res

    exif_data = {}
    try:
        raw_exif = pil_img.getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_data[tag_name] = str(val).strip()
    except Exception as e:
        logger.debug(f"EXIF extraction note: {str(e)}")

    # Check info dictionary
    for k, v in getattr(pil_img, "info", {}).items():
        if isinstance(v, (str, bytes)):
            exif_data[str(k)] = str(v).strip()

    if not exif_data:
        res["explanation"] = "No EXIF or container metadata present in image."
        return res

    res["has_metadata"] = True

    # Check software / editor tags
    software_keys = ["Software", "ProcessingSoftware", "ImageDescription", "Artist", "CreatorTool", "Make"]
    found_software = []
    for k in software_keys:
        if k in exif_data and exif_data[k]:
            found_software.append(exif_data[k])

    joined_software = " | ".join(found_software)
    if joined_software:
        res["software"] = joined_software
        s_lower = joined_software.lower()
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in s_lower:
                res["is_editing_software"] = True
                res["detected_software_name"] = kw.title()
                res["score"] = 0.40  # Weak signal score
                res["explanation"] = f"Image metadata references digital image editor software: '{joined_software}'."
                break

    # Timestamp comparison
    orig_time = exif_data.get("DateTimeOriginal") or exif_data.get("DateTimeDigitized")
    mod_time = exif_data.get("DateTime")
    if orig_time and mod_time:
        res["timestamps"] = {"original": orig_time, "modified": mod_time}
        if orig_time != mod_time:
            res["timestamp_mismatch"] = True
            if res["score"] < 0.25:
                res["score"] = 0.25
                res["explanation"] = f"Metadata timestamp discrepancy: Created '{orig_time}' vs Modified '{mod_time}'."

    return res


def normalize_date_for_comparison(date_str: str) -> Optional[str]:
    """Normalizes various visual and MRZ date representations into standard YYMMDD format.
    
    Examples:
        - "15/08/1990" -> "900815"
        - "1990-08-15" -> "900815"
        - "15-AUG-1990" -> "900815"
        - "740812" -> "740812"
    """
    if not date_str or not isinstance(date_str, str):
        return None

    import re
    cleaned = date_str.strip().upper()
    if not cleaned:
        return None

    # If already 6 digits (MRZ YYMMDD format)
    if re.fullmatch(r'\d{6}', cleaned):
        return cleaned

    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m = re.match(r'^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$', cleaned)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y_short = y[-2:] if len(y) >= 2 else y.zfill(2)
        return f"{y_short}{mo:02d}{d:02d}"

    # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    m = re.match(r'^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})$', cleaned)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{y[-2:]}{mo:02d}{d:02d}"

    # DD MMM YYYY (e.g. 15 AUG 1990, 15-AUG-1990, 15AUG90)
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }
    m = re.match(r'^(\d{1,2})[\s/-]*([A-Z]{3,9})[\s/-]*(\d{2,4})$', cleaned)
    if m:
        d = int(m.group(1))
        m_str = m.group(2)[:3]
        y = m.group(3)
        if m_str in month_map:
            mo = month_map[m_str]
            y_short = y[-2:] if len(y) >= 2 else y.zfill(2)
            return f"{y_short}{mo:02d}{d:02d}"

    # Fallback: extract all digits
    digits = re.sub(r'\D', '', cleaned)
    if len(digits) == 6:
        return digits
    elif len(digits) == 8:
        # Assume DDMMYYYY or YYYYMMDD
        if int(digits[4:]) > 1900:  # DDMMYYYY
            return f"{digits[6:]}{digits[2:4]}{digits[0:2]}"
        elif int(digits[:4]) > 1900:  # YYYYMMDD
            return f"{digits[2:4]}{digits[4:6]}{digits[6:]}"

    return None


def normalize_doc_number_for_comparison(doc_str: str) -> str:
    """Normalizes document / passport / visa numbers by removing whitespace, hyphens, and filler characters."""
    if not doc_str or not isinstance(doc_str, str):
        return ""
    import re
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(doc_str)).upper()
    return cleaned


def normalize_name_for_comparison(name_str: str) -> str:
    """Normalizes person names for cross-zone consistency comparison."""
    if not name_str or not isinstance(name_str, str):
        return ""
    import re
    # Remove commas, periods, hyphens, extra whitespace
    cleaned = re.sub(r'[^A-Za-z\s]', ' ', str(name_str)).upper()
    tokens = [t for t in cleaned.split() if t]
    return " ".join(tokens)


def normalize_sex_for_comparison(sex_str: str) -> str:
    """Normalizes gender / sex string to single uppercase character ('M', 'F', 'X')."""
    if not sex_str or not isinstance(sex_str, str):
        return ""
    s = str(sex_str).strip().upper()
    if s in ("M", "MALE", "HOMME", "MASCULINE", "MASCULIN"):
        return "M"
    if s in ("F", "FEMALE", "FEMME", "FEMININE", "FEMININ"):
        return "F"
    if s in ("X", "UNSPECIFIED", "<"):
        return "X"
    return s[:1] if s else ""


def normalize_nationality_for_comparison(nat_str: str) -> str:
    """Normalizes 3-letter ICAO/ISO nationality or issuing state code."""
    if not nat_str or not isinstance(nat_str, str):
        return ""
    import re
    cleaned = re.sub(r'[^A-Za-z]', '', str(nat_str)).upper()
    return cleaned[:3]

