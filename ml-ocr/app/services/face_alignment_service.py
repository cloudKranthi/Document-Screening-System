"""Face Alignment Service for Identity Documents and Selfies.

Performs deterministic 5-point facial landmark affine alignment to standard
112x112 canonical dimensions expected by ArcFace / SFace embedding models.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard 112x112 reference landmarks for ArcFace / SFace canonical alignment
STANDARD_5_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],  # Right eye (subject's right, viewer's left)
    [73.5318, 51.6963],  # Left eye (subject's left, viewer's right)
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Right mouth corner
    [70.7299, 92.3655]   # Left mouth corner
], dtype=np.float32)


class FaceAlignmentService:
    """Aligns detected faces into canonical 112x112 crops using 5-point landmarks."""

    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        self.output_size = output_size

    def align_face(
        self,
        image: np.ndarray,
        face_info: Dict[str, Any],
        recognizer: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """Aligns a detected face using official SFace alignCrop or 5-point landmark affine warp.
        
        Args:
            image: Original BGR image.
            face_info: Face dictionary containing "raw_face", "landmarks", and "bbox".
            recognizer: Optional cv2.FaceRecognizerSF instance.
            
        Returns:
            Aligned BGR face crop of shape (112, 112, 3), or None if alignment fails.
        """
        if image is None or image.size == 0 or not face_info:
            return None

        # 1. Official SFace recognizer alignment if available
        if recognizer is not None and "raw_face" in face_info and face_info["raw_face"] is not None:
            try:
                aligned = recognizer.alignCrop(image, face_info["raw_face"])
                if aligned is not None and aligned.size > 0:
                    return aligned
            except Exception as e:
                logger.debug(f"recognizer.alignCrop fallback to affine transformation: {e}")

        # 2. 5-point landmark affine warp
        landmarks = face_info.get("landmarks")
        if landmarks is not None and len(landmarks) == 5:
            aligned = self.align_landmarks_5point(image, np.array(landmarks, dtype=np.float32))
            if aligned is not None:
                return aligned

        # 3. Bounding box fallback crop
        bbox = face_info.get("bbox")
        if bbox is not None and len(bbox) == 4:
            return self.crop_bbox_fallback(image, bbox)

        return None

    def align_landmarks_5point(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> Optional[np.ndarray]:
        """Calculates optimal similarity transform between source landmarks and canonical template."""
        if image is None or landmarks is None or len(landmarks) != 5:
            return None

        try:
            src_pts = landmarks.astype(np.float32)
            dst_pts = STANDARD_5_LANDMARKS_112.astype(np.float32)

            # Estimate affine partial 2D transformation (translation + rotation + uniform scale)
            tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
            if tform is None:
                # Fallback to standard 3-point affine on eyes and nose
                tform = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])

            aligned = cv2.warpAffine(
                image,
                tform,
                self.output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            return aligned
        except Exception as e:
            logger.error(f"5-point landmark alignment error: {e}")
            return None

    def crop_bbox_fallback(
        self,
        image: np.ndarray,
        bbox: list
    ) -> Optional[np.ndarray]:
        """Crops and resizes face bounding box directly if landmark alignment is unavailable."""
        try:
            x, y, w, h = bbox
            h_img, w_img = image.shape[:2]

            # Add 10% context margin
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            return cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logger.error(f"BBox fallback crop error: {e}")
            return None
