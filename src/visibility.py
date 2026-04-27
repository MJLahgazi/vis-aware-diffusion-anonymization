from typing import Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

from config import (
    DEEPFACE_BOOST_MAX,
    DEVICE,
    FACIAL_KEYPOINT_CONF,
    TARGET_UPSCALE_SIZE,
    UPSCALE_THRESHOLD,
)


def detect_face(image: np.ndarray, bbox: Tuple) -> Tuple[Optional[Tuple], float]:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    h, w = crop.shape[:2]

    if h < UPSCALE_THRESHOLD or w < UPSCALE_THRESHOLD:
        scale = TARGET_UPSCALE_SIZE / max(h, w)
        crop  = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0

    face_objs = DeepFace.extract_faces(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        detector_backend="retinaface",
        enforce_detection=False,
        align=False,
    )

    if not face_objs:
        return None, 0.0

    best  = max(face_objs, key=lambda x: x["confidence"])
    r     = best["facial_area"]
    fx1   = x1 + int(r["x"] / scale)
    fy1   = y1 + int(r["y"] / scale)
    face_bbox = (fx1, fy1, fx1 + int(r["w"] / scale), fy1 + int(r["h"] / scale))

    return face_bbox, best["confidence"]


def detect_keypoints(image: np.ndarray, bbox: Tuple, model_pose) -> Tuple:
    x1, y1, x2, y2 = bbox
    results = model_pose(image[y1:y2, x1:x2], conf=0.1, verbose=False, device=DEVICE)

    for result in results:
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue
        kp_data = result.keypoints.data[0].cpu().numpy()
        if len(kp_data) < 17:
            continue

        keypoints   = kp_data[:, :2] + np.array([x1, y1])
        confidences = kp_data[:, 2]

        facial_indices = [0, 1, 2, 3, 4]
        facial_confs   = confidences[facial_indices]
        facial_count   = int(np.sum(facial_confs >= FACIAL_KEYPOINT_CONF))

        return facial_count, keypoints[facial_indices], facial_confs

    return 0, None, None


def compute_visibility_score(face_confidence: float, facial_kp_count: int) -> float:
    return min(1.0, (facial_kp_count / 5.0) + (face_confidence * DEEPFACE_BOOST_MAX))
