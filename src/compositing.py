from typing import Tuple

import cv2
import numpy as np


def composite_result(
    base_image: np.ndarray,
    inpainted_crop: np.ndarray,
    crop_bbox: Tuple,
    crop_mask: np.ndarray,
) -> np.ndarray:
    result       = base_image.copy()
    x1, y1, x2, y2 = crop_bbox
    crop_h, crop_w  = y2 - y1, x2 - x1

    if inpainted_crop.shape[:2] != (crop_h, crop_w):
        inpainted_crop = cv2.resize(inpainted_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    if crop_mask.shape[:2] != (crop_h, crop_w):
        crop_mask = cv2.resize(crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    alpha                    = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR) / 255.0
    region                   = result[y1:y2, x1:x2]
    result[y1:y2, x1:x2]    = (inpainted_crop * alpha + region * (1 - alpha)).astype(np.uint8)

    return result
