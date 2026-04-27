from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    ANONYMIZE_FULL_BODY,
    BBOX_CUTOFF_RATIO,
    CROP_PADDING,
    DEEPFACE_MIN_CONFIDENCE,
    DEVICE,
    FACE_EXTENSION,
    FACIAL_KEYPOINT_CONF,
    SEGMENTATION_CONF,
)


def get_person_segmentation(image: np.ndarray, bbox: Tuple, model_seg) -> Optional[np.ndarray]:
    h, w   = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad    = 50
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    crop   = image[cy1:cy2, cx1:cx2]

    results = model_seg(crop, conf=SEGMENTATION_CONF, verbose=False, device=DEVICE, classes=[0])

    for result in results:
        if result.masks is None or not result.masks.xy:
            continue

        masks_xy = result.masks.xy
        best_idx = 0

        if len(masks_xy) > 1:
            target  = np.array([(x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1])
            boxes   = result.boxes.xyxy.cpu().numpy()
            best_idx = min(
                range(len(boxes)),
                key=lambda i: np.linalg.norm(
                    np.array([(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2]) - target
                ),
            )

        pts = masks_xy[best_idx]
        if len(pts) < 3:
            continue

        crop_mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
        cv2.fillPoly(crop_mask, [pts.astype(np.int32)], 255)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[cy1:cy2, cx1:cx2] = crop_mask
        return full_mask

    return None


def determine_cutoff_y(
    bbox: Tuple,
    face_bbox: Optional[Tuple],
    face_confidence: float,
    facial_kp_count: int,
    facial_positions: Optional[np.ndarray],
    facial_confs: Optional[np.ndarray],
    img_h: int,
) -> int:
    x1, y1, x2, y2 = bbox

    if face_bbox is not None and face_confidence >= DEEPFACE_MIN_CONFIDENCE:
        face_height = face_bbox[3] - face_bbox[1]
        cutoff_y    = face_bbox[3] + int(face_height * FACE_EXTENSION)

    elif facial_kp_count >= 2 and facial_positions is not None:
        visible_y   = [int(facial_positions[i][1]) for i in range(len(facial_confs))
                       if facial_confs[i] >= FACIAL_KEYPOINT_CONF]
        lowest_kp_y = max(visible_y)
        cutoff_y    = lowest_kp_y + int(0.5 * (lowest_kp_y - y1))

    else:
        cutoff_y = y1 + int((y2 - y1) * BBOX_CUTOFF_RATIO)

    return min(cutoff_y, y2 - 1, img_h - 1)


def _make_ellipse_from_face_bbox(
    face_bbox: Tuple, image_shape: Tuple, source_type: str = "face_detection"
) -> np.ndarray:
    h, w = image_shape[:2]
    fx1, fy1, fx2, fy2 = face_bbox
    fw, fh = fx2 - fx1, fy2 - fy1

    if source_type == "keypoints":
        head_w, head_h    = int(fw * 1.8), int(fh * 3.2)
        center_offset     = 0.4
    else:
        head_w, head_h    = int(fw * 1.4), int(fh * 1.6)
        center_offset     = 0.3

    center = ((fx1 + fx2) // 2, fy1 + int(fh * center_offset))
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, (head_w // 2, head_h // 2), 0, 0, 360, 255, -1)
    return mask


def _make_ellipse_from_person_bbox(bbox: Tuple, image_shape: Tuple) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    ph, pw  = y2 - y1, x2 - x1
    center  = ((x1 + x2) // 2, y1 + int(ph * 0.2))
    mask    = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, (int(pw * 0.7) // 2, int(ph * 0.3) // 2), 0, 0, 360, 255, -1)
    return mask


def extract_head_mask(
    person_mask: Optional[np.ndarray],
    cutoff_y: int,
    bbox: Tuple,
    face_bbox: Optional[Tuple],
    face_confidence: float,
    facial_kp_count: int,
    facial_positions: Optional[np.ndarray],
    facial_confs: Optional[np.ndarray],
    image_shape: Tuple,
) -> np.ndarray:
    if person_mask is not None:
        if ANONYMIZE_FULL_BODY:
            return person_mask.copy()
        head_mask = person_mask.copy()
        head_mask[cutoff_y:, :] = 0
        return head_mask

    if face_bbox is not None and face_confidence >= DEEPFACE_MIN_CONFIDENCE:
        return _make_ellipse_from_face_bbox(face_bbox, image_shape, "face_detection")

    if facial_kp_count >= 2 and facial_positions is not None:
        visible_kps = np.array([
            facial_positions[i] for i in range(len(facial_confs))
            if facial_confs[i] >= FACIAL_KEYPOINT_CONF
        ])
        kp_bbox = (
            int(visible_kps[:, 0].min()), int(visible_kps[:, 1].min()),
            int(visible_kps[:, 0].max()), int(visible_kps[:, 1].max()),
        )
        return _make_ellipse_from_face_bbox(kp_bbox, image_shape, "keypoints")

    return _make_ellipse_from_person_bbox(bbox, image_shape)


def expand_mask(head_mask: np.ndarray, padding: float = CROP_PADDING) -> np.ndarray:
    if padding <= 0:
        return head_mask.copy()

    y_idx, x_idx = np.where(head_mask > 0)
    if len(x_idx) == 0:
        return head_mask.copy()

    avg_dim  = ((x_idx.max() - x_idx.min()) + (y_idx.max() - y_idx.min())) / 2
    dilation = max(5, int(avg_dim * padding * 0.15))
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
    expanded = cv2.dilate(head_mask, kernel, iterations=1)
    return cv2.GaussianBlur(expanded, (21, 21), 0)
