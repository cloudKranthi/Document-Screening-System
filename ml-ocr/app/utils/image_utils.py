"""OpenCV image processing utilities for document boundary detection, perspective transformation, and enhancement."""

from typing import Optional, Tuple
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 coordinates as: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Numpy array of 4 coordinates [[x, y], ...]
        
    Returns:
        Ordered numpy array of 4 coordinates [tl, tr, br, bl] (float32).
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has the smallest sum (x + y), bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference (y - x), bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Performs a 4-point perspective warp on the image based on detected quad points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Compute height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    if max_width < 10 or max_height < 10:
        return image
        
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def detect_document_contour(image: np.ndarray, min_area_ratio: float = 0.15) -> Optional[np.ndarray]:
    """Detects 4-corner document contour in the image.
    
    Returns:
        np.ndarray of 4 points if found, else None.
    """
    height, width = image.shape[:2]
    total_area = height * width
    
    # Preprocess for contour detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection and morphological closing
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < total_area * min_area_ratio:
            continue
            
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        # Check if 4 vertices found and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            return pts
            
    return None


def resize_image_if_needed(image: np.ndarray, max_dim: int = 1400, min_dim: int = 0) -> Tuple[np.ndarray, float]:
    """Downscales image if dimensions exceed max_dim (preserving aspect ratio). Does NOT upscale small images."""
    h, w = image.shape[:2]
    scale = 1.0
    
    max_current = max(h, w)
    
    if max_current > max_dim:
        scale = max_dim / float(max_current)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif min_dim > 0 and min(h, w) < min_dim and min(h, w) > 0:
        scale = min_dim / float(min(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
    return image, scale



def enhance_for_ocr(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8) -> np.ndarray:
    """Applies grayscale conversion, CLAHE contrast enhancement, and noise reduction for OCR."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Apply Bilateral Filter to remove noise while keeping edges crisp
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(denoised)
    
    return enhanced


def deskew_image(gray_image: np.ndarray) -> np.ndarray:
    """Detects document skew angle and rotates the image to upright."""
    # Threshold image to find text regions
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return gray_image
        
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
        
    if abs(angle) < 0.5 or abs(angle) > 45:
        return gray_image
        
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray_image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def extract_mrz_region(image: np.ndarray, bottom_ratio: float = 0.35) -> np.ndarray:
    """Extracts the bottom region of the document (typically bottom 20-40%) where MRZ resides.
    
    Args:
        image: Numpy image (BGR or Grayscale).
        bottom_ratio: Fraction of height from bottom to crop (e.g. 0.35 = bottom 35%).
        
    Returns:
        Cropped numpy array containing the bottom MRZ band.
    """
    h, w = image.shape[:2]
    start_y = int(h * (1.0 - bottom_ratio))
    start_y = max(0, min(start_y, h - 10))
    return image[start_y:h, 0:w].copy()


def get_mrz_candidate_crops(image: np.ndarray, ratios: Optional[list[float]] = None) -> list[tuple[float, np.ndarray]]:
    """Extracts multiple candidate crops from the bottom of the image across varying height ratios.
    
    Ratios default to: [0.20, 0.25, 0.30, 0.35, 0.40] to ensure both lines are captured regardless of framing.
    """
    if ratios is None:
        ratios = [0.20, 0.25, 0.30, 0.35, 0.40]
    candidate_crops = []
    for r in ratios:
        crop = extract_mrz_region(image, bottom_ratio=r)
        if crop.shape[0] >= 20 and crop.shape[1] >= 50:
            candidate_crops.append((r, crop))
    return candidate_crops


def preprocess_mrz_crop(mrz_crop: np.ndarray, scale_factor: float = 3.0) -> list[tuple[str, np.ndarray]]:
    """Preprocesses the MRZ crop with 3x upscaling, grayscale, CLAHE, Otsu, Adaptive, and Blackhat variants.
    
    Returns:
        List of tuples: (variant_name, processed_image)
          1. 'clahe': 3.0x upscaled, Bilateral filtered, CLAHE enhanced grayscale
          2. 'otsu': Otsu binarized threshold
          3. 'adaptive': Adaptive Gaussian threshold
          4. 'blackhat': Morphological blackhat threshold (isolates dark text from bright patterned background)
    """
    if len(mrz_crop.shape) == 3:
        gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = mrz_crop.copy()
        
    # 1. Upscale MRZ crop 3x using bicubic interpolation for optimal OCR-B character resolution
    h, w = gray.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    upscaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Denoising while preserving sharp character strokes
    denoised = cv2.bilateralFilter(upscaled, d=7, sigmaColor=50, sigmaSpace=50)
    
    # 3. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(denoised)
    
    # 4. Otsu binarization
    _, otsu_thresh = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 5. Adaptive Gaussian thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        clahe_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    
    # 6. Morphological Blackhat (useful for dark MRZ text on textured/security laminate backgrounds)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(clahe_enhanced, cv2.MORPH_BLACKHAT, rect_kernel)
    _, blackhat_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return [
        ("clahe", clahe_enhanced),
        ("otsu", otsu_thresh),
        ("adaptive", adaptive_thresh),
        ("blackhat", blackhat_thresh),
    ]


def get_field_candidate_crops(
    image: np.ndarray,
    label_bbox: list[int],
    field_type: str = "dob"
) -> list[tuple[str, np.ndarray, list[int]]]:
    """Extracts dynamic ROI candidate crops around a detected field label bounding box.
    
    Args:
        image: Full document image (Grayscale or BGR).
        label_bbox: [x1, y1, x2, y2] bounding box coordinates of the label token.
        field_type: 'dob' or 'gender'.
        
    Returns:
        List of tuples: (crop_name, cropped_image, crop_bbox)
    """
    img_h, img_w = image.shape[:2]
    lx1, ly1, lx2, ly2 = label_bbox
    lh = max(12, ly2 - ly1)
    lw = max(12, lx2 - lx1)

    crops = []
    
    # Width reach based on field type
    w_mults = [5.0, 7.0, 9.0] if field_type == "dob" else [3.5, 5.5, 7.5]
    
    # 1. Primary: Right of label with varying vertical padding and width
    for idx, wm in enumerate(w_mults):
        pad_y = int(lh * (0.2 + 0.15 * idx))
        x1 = max(0, lx2 - 5)
        x2 = min(img_w, lx2 + int(lh * wm))
        y1 = max(0, ly1 - pad_y)
        y2 = min(img_h, ly2 + pad_y)
        
        if (x2 - x1) >= 20 and (y2 - y1) >= 10:
            crops.append((f"right_var_{idx+1}", image[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]))

    # 2. Secondary: Region immediately below label (for vertical/stacked card formats)
    pad_bx1 = max(0, lx1 - 10)
    pad_bx2 = min(img_w, lx2 + int(lh * 4.0))
    pad_by1 = max(0, ly2 - 2)
    pad_by2 = min(img_h, ly2 + int(lh * 2.2))
    if (pad_bx2 - pad_bx1) >= 20 and (pad_by2 - pad_by1) >= 10:
        crops.append(("below", image[pad_by1:pad_by2, pad_bx1:pad_bx2].copy(), [pad_bx1, pad_by1, pad_bx2, pad_by2]))

    # 3. Tertiary: Combined line region (label + value)
    pad_lx1 = max(0, lx1 - 5)
    pad_lx2 = min(img_w, lx2 + int(lh * 7.0))
    pad_ly1 = max(0, ly1 - int(lh * 0.25))
    pad_ly2 = min(img_h, ly2 + int(lh * 0.25))
    if (pad_lx2 - pad_lx1) >= 20 and (pad_ly2 - pad_ly1) >= 10:
        crops.append(("combined_line", image[pad_ly1:pad_ly2, pad_lx1:pad_lx2].copy(), [pad_lx1, pad_ly1, pad_lx2, pad_ly2]))

    return crops


def preprocess_field_crop(field_crop: np.ndarray, scale_factor: float = 3.0) -> list[tuple[str, np.ndarray]]:
    """Preprocesses a field ROI crop with 2x-4x upscaling, Bilateral filtering, CLAHE, sharpening, Otsu, and Adaptive thresholding.
    
    Returns:
        List of tuples: (prep_name, processed_image)
          1. 'clahe_sharp': 3x upscaled, Bilateral filtered, CLAHE enhanced + unsharp mask sharpening
          2. 'otsu': Otsu binarized threshold
          3. 'adaptive': Adaptive Gaussian threshold
          4. 'adaptive_sharp': Adaptive threshold on sharpened image
    """
    if len(field_crop.shape) == 3:
        gray = cv2.cvtColor(field_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = field_crop.copy()
        
    h, w = gray.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    if new_w < 10 or new_h < 10:
        return [("raw", gray)]
        
    # 1. Upscale using bicubic interpolation
    upscaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Bilateral noise reduction
    denoised = cv2.bilateralFilter(upscaled, d=5, sigmaColor=40, sigmaSpace=40)
    
    # 3. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(denoised)
    
    # 4. Unsharp mask sharpening
    gaussian = cv2.GaussianBlur(clahe_enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(clahe_enhanced, 1.5, gaussian, -0.5, 0)
    
    # 5. Otsu binarization
    _, otsu_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 6. Adaptive Gaussian thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        clahe_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    
    # 7. Adaptive threshold on sharpened
    adaptive_sharp = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    
    return [
        ("clahe_sharp", sharpened),
        ("otsu", otsu_thresh),
        ("adaptive", adaptive_thresh),
        ("adaptive_sharp", adaptive_sharp),
    ]



