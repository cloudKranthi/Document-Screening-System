"""Forensic image analysis utility algorithms: Error Level Analysis (ELA), EXIF metadata analysis, and robust statistical signal processors."""

import io
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import ExifTags, Image

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Default editing software keyword signatures (case-insensitive)
DEFAULT_EDITING_SOFTWARE_KEYWORDS = [
    "adobe photoshop",
    "photoshop",
    "gimp",
    "gnu image manipulation program",
    "lightroom",
    "affinity photo",
    "paint.net",
    "photopea",
    "coreldraw",
    "pixelmator",
    "canva",
    "indesign",
    "picsart",
    "snapseed",
    "pixlr",
    "seashore",
]


def compute_error_level_analysis(
    image: np.ndarray,
    quality: int = 92,
    block_size: int = 32,
    layout_regions: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Computes Error Level Analysis (ELA) on an image using in-memory JPEG recompression and robust block statistics.
    
    ELA pipeline:
      1. Ensure input image is valid RGB/BGR NumPy array.
      2. Re-save temporary JPEG in memory at controlled quality (e.g. 90-95).
      3. Decode recompressed image.
      4. Calculate absolute pixel difference |I_orig - I_recomp|.
      5. Compute global and local tile/block statistics using robust median & MAD.
      6. Detect localized regions with divergent recompression error.
      7. Apply false-positive protection (uniform recompression yields LOW score).
      
    Args:
        image: BGR or Grayscale NumPy array of document image.
        quality: JPEG compression quality parameter (default: 92).
        block_size: Spatial tile dimension in pixels (default: 32).
        layout_regions: Optional OCR layout bounding boxes for text-density cross-checking.
        
    Returns:
        Dictionary containing score, evaluated, reason, summary, suspicious_regions, and quantitative metrics.
    """
    res: Dict[str, Any] = {
        "score": 0.0,
        "evaluated": False,
        "reason": "INSUFFICIENT_EVIDENCE",
        "summary": "ELA not evaluated: image data is empty or unreadable.",
        "suspicious_regions": [],
        "metrics": {
            "mean_error": 0.0,
            "median_error": 0.0,
            "std_error": 0.0,
            "max_error": 0.0,
            "percentile_95": 0.0,
            "mad": 0.0,
            "outlier_ratio": 0.0,
            "largest_deviation": 0.0,
            "suspicious_blocks_count": 0,
            "total_blocks_count": 0,
        },
        "debug": {
            "jpeg_quality": quality,
            "block_size": block_size,
            "ela_blocks_evaluated": 0,
            "ela_threshold": 0.0
        }
    }

    if image is None or image.size == 0:
        return res

    # 1. Convert safely to 3-channel BGR
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    h, w = bgr.shape[:2]
    if h < 50 or w < 50:
        res["summary"] = f"Image dimensions ({w}x{h} px) too small for reliable ELA evaluation."
        return res

    # 2. Re-save temporary JPEG in memory at controlled quality (no disk file created)
    quality_clamped = max(50, min(100, int(quality)))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_clamped]
    success, enc_buf = cv2.imencode(".jpg", bgr, encode_param)
    if not success or enc_buf is None:
        res["summary"] = "In-memory JPEG recompression failed."
        return res

    # 3. Decode recompressed image
    recompressed = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
    if recompressed is None:
        res["summary"] = "Failed to decode recompressed JPEG image."
        return res

    # 4. Calculate absolute pixel difference map
    diff = cv2.absdiff(bgr, recompressed)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 5. Global statistical metrics
    mean_err = float(np.mean(diff_gray))
    median_err = float(np.median(diff_gray))
    std_err = float(np.std(diff_gray))
    max_err = float(np.max(diff_gray))
    p95_err = float(np.percentile(diff_gray, 95))

    # 6. Block/tile-based spatial inconsistency analysis
    bs = max(8, int(block_size))
    n_blocks_y = h // bs
    n_blocks_x = w // bs
    total_blocks = n_blocks_y * n_blocks_x

    if total_blocks < 4:
        # Image too small for tiling
        res["evaluated"] = True
        res["reason"] = "NO_ANOMALY_FOUND"
        res["summary"] = "Image too small for tiled ELA; global recompression error is uniform."
        res["metrics"].update({
            "mean_error": round(mean_err, 2),
            "median_error": round(median_err, 2),
            "std_error": round(std_err, 2),
            "max_error": round(max_err, 2),
            "percentile_95": round(p95_err, 2),
        })
        return res

    block_means = []
    block_coords = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y1 = by * bs
            x1 = bx * bs
            y2 = min(h, y1 + bs)
            x2 = min(w, x1 + bs)

            tile = diff_gray[y1:y2, x1:x2]
            block_means.append(float(np.mean(tile)))
            block_coords.append([x1, y1, x2, y2])

    block_means_arr = np.array(block_means, dtype=np.float32)
    med_block_err = float(np.median(block_means_arr))
    abs_devs = np.abs(block_means_arr - med_block_err)
    mad_block_err = float(np.median(abs_devs))
    robust_scale = max(0.5, 1.4826 * mad_block_err)

    # Dynamic threshold: require localized residual divergence above document baseline
    dev_threshold = max(2.5, 2.5 * robust_scale)

    outlier_indices = []
    max_local_dev = 0.0

    for idx, b_mean in enumerate(block_means):
        dev = b_mean - med_block_err
        if dev > max_local_dev:
            max_local_dev = dev
        if dev >= dev_threshold:
            outlier_indices.append(idx)

    outlier_count = len(outlier_indices)
    outlier_ratio = float(outlier_count / max(1, total_blocks))

    # 7. Merge adjacent outlier blocks into spatial regions
    raw_outlier_boxes = [block_coords[idx] for idx in outlier_indices]
    merged_regions = _merge_bounding_boxes(raw_outlier_boxes, max_gap=bs * 1.5)

    # Filter out tiny isolated single-block noise unless it has a notable jump
    significant_regions = []
    for r in merged_regions:
        r_w = r[2] - r[0]
        r_h = r[3] - r[1]
        area = r_w * r_h
        if area >= (bs * bs * 0.9) or max_local_dev >= 8.0:
            significant_regions.append(r)

    # 8. False-Positive Safeguards:
    # If outlier ratio is zero or max deviation is negligible (< 2.5), treat as clean
    if len(significant_regions) == 0 or max_local_dev < 2.5:
        score = 0.0
        reason = "NO_ANOMALY_FOUND"
        summary = "Uniform JPEG recompression residual consistent with document baseline."
        suspicious_regions = []
    else:
        # Score scales smoothly with outlier density and peak deviation
        base_score = 0.20 + (outlier_ratio * 3.5) + (min(30.0, max_local_dev) / 30.0) * 0.45
        if len(significant_regions) >= 2 or max_local_dev >= 15.0:
            base_score = max(base_score, 0.45)
        if max_local_dev >= 25.0 and outlier_ratio >= 0.02:
            base_score = max(base_score, 0.70)

        score = float(round(max(0.0, min(1.0, base_score)), 4))
        reason = "EVALUATED"
        summary = (
            f"Detected {len(significant_regions)} localized region(s) with significant JPEG "
            f"recompression residual divergence (peak deviation: +{max_local_dev:.1f} vs baseline median: {med_block_err:.1f})."
        )
        suspicious_regions = significant_regions[:8]


    res.update({
        "score": score,
        "evaluated": True,
        "reason": reason,
        "summary": summary,
        "suspicious_regions": suspicious_regions,
        "metrics": {
            "mean_error": round(mean_err, 2),
            "median_error": round(median_err, 2),
            "std_error": round(std_err, 2),
            "max_error": round(max_err, 2),
            "percentile_95": round(p95_err, 2),
            "mad": round(mad_block_err, 2),
            "outlier_ratio": round(outlier_ratio, 4),
            "largest_deviation": round(float(max_local_dev), 2),
            "suspicious_blocks_count": outlier_count,
            "total_blocks_count": total_blocks,
        },
        "debug": {
            "jpeg_quality": quality_clamped,
            "block_size": bs,
            "ela_blocks_evaluated": total_blocks,
            "ela_threshold": round(dev_threshold, 2)
        }
    })
    return res


def extract_exif_metadata(
    image_bytes: Optional[bytes],
    software_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Inspects EXIF and container metadata for photo-editing software signatures using Pillow.
    
    Safety Rules:
      - Missing EXIF is NOT evidence of tampering (returns evaluated=False, reason="NO_METADATA", score=0.0).
      - Confirmed clean metadata (e.g. camera EXIF tags without editor signatures) returns evaluated=True, score=0.0.
      - Editing software presence returns evaluated=True, score=0.35, editing_software_detected=True.
      
    Args:
        image_bytes: Raw binary bytes of uploaded image.
        software_keywords: List of editing software identifier substrings (case-insensitive).
        
    Returns:
        Dictionary matching metadata signal schema.
    """
    res: Dict[str, Any] = {
        "score": 0.0,
        "evaluated": False,
        "editing_software_detected": False,
        "software": None,
        "metadata_present": False,
        "reason": "NO_METADATA",
        "summary": "No EXIF or container metadata present in image.",
        "metadata_fields": {},
        "debug": {
            "metadata_keys_found": []
        }
    }

    if not image_bytes or len(image_bytes) == 0:
        return res

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.debug(f"Pillow image decode for metadata failed: {str(e)}")
        res["summary"] = "Unable to read image container metadata."
        return res

    exif_data: Dict[str, str] = {}

    # 1. Read EXIF tags safely
    try:
        raw_exif = pil_img.getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if val is not None and not isinstance(val, (bytes, bytearray)):
                    exif_data[tag_name] = str(val).strip()
                elif isinstance(val, (bytes, bytearray)):
                    try:
                        exif_data[tag_name] = val.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"EXIF tag iteration note: {str(e)}")

    # 2. Inspect info dictionary (e.g. PNG text chunks, JPEG comments, XMP)
    for k, v in getattr(pil_img, "info", {}).items():
        if isinstance(v, str) and v.strip():
            exif_data[str(k)] = v.strip()
        elif isinstance(v, (bytes, bytearray)):
            try:
                decoded = v.decode("utf-8", errors="ignore").strip()
                if decoded and len(decoded) < 500:
                    exif_data[str(k)] = decoded
            except Exception:
                pass

    if not exif_data:
        res["metadata_present"] = False
        res["evaluated"] = False
        res["reason"] = "NO_METADATA"
        res["summary"] = "No EXIF or container metadata present in image."
        return res

    res["metadata_present"] = True
    res["debug"]["metadata_keys_found"] = list(exif_data.keys())

    # 3. Check for editing software signatures in relevant metadata fields
    keywords = software_keywords or settings.TAMPERING_EDITING_SOFTWARE_KEYWORDS or DEFAULT_EDITING_SOFTWARE_KEYWORDS
    target_fields = [
        "Software", "ProcessingSoftware", "ImageDescription", "Artist",
        "Make", "Model", "CreatorTool", "History", "XMP", "comment", "icc_profile"
    ]

    found_software_entries = []
    for tf in target_fields:
        if tf in exif_data and exif_data[tf]:
            found_software_entries.append(exif_data[tf])

    # Also search general values for explicit Photoshop/GIMP signatures
    all_values_str = " | ".join(list(exif_data.values()))
    software_display = " | ".join(found_software_entries) if found_software_entries else None
    res["software"] = software_display

    detected_editor = None
    all_values_lower = all_values_str.lower()

    for kw in keywords:
        kw_clean = kw.strip().lower()
        if kw_clean and kw_clean in all_values_lower:
            detected_editor = kw.title()
            break

    if detected_editor:
        res["score"] = 0.35  # Standard configurable metadata risk score
        res["evaluated"] = True
        res["editing_software_detected"] = True
        res["software"] = software_display or detected_editor
        res["reason"] = "EVALUATED"
        res["summary"] = f"Metadata reports digital image editing software: '{res['software']}'. This is a supporting risk signal only."
    else:
        # Confirmed metadata present with legitimate camera/device tags and no editor signatures
        res["score"] = 0.0
        res["evaluated"] = True
        res["editing_software_detected"] = False
        res["reason"] = "NO_ANOMALY_FOUND"
        res["summary"] = "Metadata present with device/capture tags and no editor signatures detected."

    # Store safe non-sensitive metadata fields
    safe_keys = ["Software", "ProcessingSoftware", "Make", "Model", "DateTime", "DateTimeOriginal", "ModifyDate", "Orientation"]
    res["metadata_fields"] = {k: exif_data[k] for k in safe_keys if k in exif_data}

    return res


def compute_noise_residual(
    gray: np.ndarray,
    block_size: int = 32
) -> Dict[str, Any]:
    """Estimates localized sensor noise variance distribution across tiles.
    
    Returns neutral explainable output without accusing the document without corroboration.
    """
    if gray is None or gray.size == 0:
        return {
            "score": 0.0,
            "evaluated": False,
            "reason": "INSUFFICIENT_EVIDENCE",
            "summary": "Noise analysis not evaluated: image empty.",
            "regions": [],
            "metrics": {}
        }

    h, w = gray.shape[:2]
    denoised = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)

    # Exclude high-contrast text edges to prevent normal font strokes from being flagged as noise anomalies
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    non_edge_mask = (grad_mag < 30.0)

    bs = max(8, int(block_size))
    n_by = h // bs
    n_bx = w // bs
    total_blocks = n_by * n_bx

    if total_blocks < 4:
        return {
            "score": 0.0,
            "evaluated": True,
            "reason": "NO_ANOMALY_FOUND",
            "summary": "Consistent sensor noise distribution across document.",
            "regions": [],
            "metrics": {}
        }

    block_stds = []
    block_coords = []

    for by in range(n_by):
        for bx in range(n_bx):
            y1 = by * bs
            x1 = bx * bs
            y2 = min(h, y1 + bs)
            x2 = min(w, x1 + bs)
            tile_res = residual[y1:y2, x1:x2]
            tile_mask = non_edge_mask[y1:y2, x1:x2]
            valid_pixels = tile_res[tile_mask]
            if len(valid_pixels) >= (bs * bs * 0.20):
                b_std = float(np.std(valid_pixels))
            else:
                b_std = float(np.std(tile_res))
            block_stds.append(b_std)
            block_coords.append([x1, y1, x2, y2])


    stds_arr = np.array(block_stds, dtype=np.float32)
    med_std = float(np.median(stds_arr))
    mad_std = float(np.median(np.abs(stds_arr - med_std)))
    scale = max(0.8, 1.4826 * mad_std)

    outlier_boxes = []
    max_dev = 0.0

    for idx, b_std in enumerate(block_stds):
        dev = abs(b_std - med_std)
        if dev > max_dev:
            max_dev = dev
        if dev >= max(14.0, 3.5 * scale):
            outlier_boxes.append(block_coords[idx])

    outlier_ratio = len(outlier_boxes) / max(1, total_blocks)
    merged_regions = _merge_bounding_boxes(outlier_boxes, max_gap=bs * 1.5)

    if len(merged_regions) == 0 or outlier_ratio < 0.015 or max_dev < 14.0:
        score = 0.0
        reason = "NO_ANOMALY_FOUND"
        summary = "Consistent sensor noise distribution across document."
        suspicious_regions = []
    else:
        score = float(round(min(1.0, 0.18 + (outlier_ratio * 3.5) + (min(30.0, max_dev) / 30.0) * 0.30), 4))
        reason = "EVALUATED"
        summary = f"Localized regions show noise statistics different from the image baseline (peak deviation: {max_dev:.1f})."
        suspicious_regions = merged_regions[:5]


    return {
        "score": score,
        "evaluated": True,
        "reason": reason,
        "summary": summary,
        "regions": suspicious_regions,
        "metrics": {
            "median_noise_std": round(med_std, 2),
            "max_noise_deviation": round(max_dev, 2),
            "outlier_ratio": round(outlier_ratio, 4)
        }
    }


def _merge_bounding_boxes(boxes: List[List[int]], max_gap: float = 40.0) -> List[List[int]]:
    """Merges spatially adjacent rectangular bounding boxes into cohesive region bounding boxes."""
    if not boxes:
        return []

    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        placed = False
        for m in merged:
            mx1, my1, mx2, my2 = m
            if not (x2 < mx1 - max_gap or x1 > mx2 + max_gap or y2 < my1 - max_gap or y1 > my2 + max_gap):
                m[0] = min(mx1, x1)
                m[1] = min(my1, y1)
                m[2] = max(mx2, x2)
                m[3] = max(my2, y2)
                placed = True
                break
        if not placed:
            merged.append([x1, y1, x2, y2])

    return merged
