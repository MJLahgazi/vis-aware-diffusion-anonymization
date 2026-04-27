from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from config import (
    GUIDANCE_SCALE,
    INPAINT_STRENGTH,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    POSITIVE_PROMPT,
)
from masking import expand_mask


def get_mask_bbox(mask: np.ndarray, image_shape: Tuple) -> Tuple:
    h, w  = image_shape[:2]
    y_idx, x_idx = np.where(mask > 0)
    if len(x_idx) == 0:
        return 0, 0, w, h
    return int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())


def run_inpainting(image: np.ndarray, head_mask: np.ndarray, sdxl_pipe) -> Tuple:
    expanded_mask        = expand_mask(head_mask)
    x1, y1, x2, y2      = get_mask_bbox(expanded_mask, image.shape)
    crop_image, crop_mask = image[y1:y2, x1:x2], expanded_mask[y1:y2, x1:x2]

    with torch.inference_mode():
        result = sdxl_pipe(
            prompt=POSITIVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)),
            mask_image=Image.fromarray(crop_mask),
            strength=INPAINT_STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]

    inpainted = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return (x1, y1, x2, y2), inpainted, crop_mask
