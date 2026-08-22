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


def resize_image_if_needed(image: np.ndarray, max_dim: int = 2400, min_dim: int = 400) -> Tuple[np.ndarray, float]:
    """Resizes image if too large or too small, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = 1.0
    
    max_current = max(h, w)
    min_current = min(h, w)
    
    if max_current > max_dim:
        scale = max_dim / float(max_current)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif min_current < min_dim and min_current > 0:
        scale = min_dim / float(min_current)
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
