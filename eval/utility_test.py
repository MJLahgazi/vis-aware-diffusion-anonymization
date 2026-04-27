import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from ultralytics import YOLO

ORIGINAL_DIR     = "images/COCO"
DEEPPRIVACY2_DIR = "deepprivacy2/COCO"
FAMS_DIR         = "FAMS/face_anon_simple/anonymized_output/COCO"
OURS_DIR         = "ours/COCO"

METHODS     = ["DeepPrivacy2", "FAMS", "Ours"]
DIRECTORIES = [ORIGINAL_DIR, DEEPPRIVACY2_DIR, FAMS_DIR, OURS_DIR]

yolo_model = YOLO("yolo11n.pt")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _yolo_detect(img_path: str) -> Tuple[bool, float]:
    results     = yolo_model(img_path, verbose=False)
    persons     = [d for d in results[0].boxes if int(d.cls) == 0]
    if not persons:
        return False, 0.0
    return True, max(float(d.conf) for d in persons)


def _deepface_detect(img_path: str) -> Tuple[bool, float]:
    try:
        face_objs = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface",
                                           enforce_detection=False, align=False)
        valid = [f for f in face_objs if f["confidence"] > 0]
        if not valid:
            return False, 0.0
        return True, valid[0]["confidence"]
    except Exception:
        return False, 0.0


def _deepface_attributes(img_path: str) -> Tuple[Optional[str], Optional[float]]:
    try:
        analysis = DeepFace.analyze(img_path=img_path, actions=["gender", "age"],
                                    detector_backend="retinaface", enforce_detection=False,
                                    silent=True)
        result = analysis[0] if isinstance(analysis, list) else analysis
        return result["dominant_gender"], result["age"]
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_yolo_boxes(img: np.ndarray, img_path: str) -> np.ndarray:
    results = yolo_model(img_path, verbose=False)
    out     = img.copy()
    for det in [d for d in results[0].boxes if int(d.cls) == 0]:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{float(det.conf):.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def _draw_deepface_boxes(img: np.ndarray, img_path: str) -> np.ndarray:
    out = img.copy()
    try:
        face_objs = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface",
                                           enforce_detection=False, align=False)
        for f in face_objs:
            if f["confidence"] <= 0:
                continue
            x, y, w, h = f["facial_area"]["x"], f["facial_area"]["y"], f["facial_area"]["w"], f["facial_area"]["h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(out, f"{f['confidence']:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    except Exception:
        pass
    return out


def create_montage(filename: str, draw_fn, out_path: str):
    titles    = ["Original"] + METHODS
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, (dir_path, title) in enumerate(zip(DIRECTORIES, titles)):
        img_path = os.path.join(dir_path, filename)
        img      = cv2.imread(img_path)
        out      = draw_fn(img, img_path) if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        axes[idx].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_directory(dir_path: str, image_files: List[str]) -> Dict:
    results = {"yolo_det": [], "yolo_conf": [], "face_det": [], "face_conf": [], "gender": [], "age": []}

    for filename in tqdm(image_files, desc=os.path.basename(dir_path)):
        img_path           = os.path.join(dir_path, filename)
        yolo_det, yolo_conf = _yolo_detect(img_path)
        face_det, face_conf = _deepface_detect(img_path)
        gender, age         = _deepface_attributes(img_path)

        results["yolo_det"].append(yolo_det)
        results["yolo_conf"].append(yolo_conf if yolo_det else np.nan)
        results["face_det"].append(face_det)
        results["face_conf"].append(face_conf if face_det else np.nan)
        results["gender"].append(gender)
        results["age"].append(age)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> str:
    arr = np.array([v for v in values if not np.isnan(v)])
    return f"{np.mean(arr):.2f}±{np.std(arr):.2f}" if len(arr) else "N/A"


def compute_metrics(orig: Dict, method: Dict) -> Dict:
    gender_pairs  = [(o, m) for o, m in zip(orig["gender"], method["gender"]) if o and m]
    gender_acc    = np.mean([o == m for o, m in gender_pairs]) * 100 if gender_pairs else 0.0

    age_errors    = [abs(o - m) for o, m in zip(orig["age"], method["age"]) if o is not None and m is not None]
    age_mae       = f"{np.mean(age_errors):.1f}" if age_errors else "N/A"

    return {
        "YOLO Det Rate":      f"{np.mean(method['yolo_det']) * 100:.1f}%",
        "YOLO Confidence":    _mean_std(method["yolo_conf"]),
        "Face Det Rate":      f"{np.mean(method['face_det']) * 100:.1f}%",
        "Face Confidence":    _mean_std(method["face_conf"]),
        "Gender Accuracy":    f"{gender_acc:.1f}%",
        "Age MAE (years)":    age_mae,
    }


def build_results_table(all_results: Dict[str, Dict], orig: Dict) -> pd.DataFrame:
    rows = {name: compute_metrics(orig, res) for name, res in all_results.items()}
    rows = {"Original": compute_metrics(orig, orig), **rows}
    metric_names = list(next(iter(rows.values())).keys())
    return pd.DataFrame({"Metric": metric_names, **{name: list(r.values()) for name, r in rows.items()}})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    image_files = sorted(f for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"Found {len(image_files)} images\n")

    create_montage(image_files[0], _draw_yolo_boxes, "yolo_detection_sample.png")
    create_montage(image_files[0], _draw_deepface_boxes, "deepface_detection_sample.png")

    orig_results = process_directory(ORIGINAL_DIR, image_files)
    all_results  = {method: process_directory(d, image_files) for method, d in zip(METHODS, DIRECTORIES[1:])}

    df = build_results_table(all_results, orig_results)
    print(df.to_string(index=False))
    df.to_csv("utility_results.csv", index=False)


if __name__ == "__main__":
    main()
