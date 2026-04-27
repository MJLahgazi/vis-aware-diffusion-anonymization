import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

ORIGINAL_DIR     = "COCO/input"
DEEPPRIVACY2_DIR = "COCO/dp2"
FAMS_DIR         = "COCO/fams"
OURS_DIR         = "COCO/ours"

METHODS      = ["DeepPrivacy2", "FAMS", "Ours"]
DIRECTORIES  = [ORIGINAL_DIR, DEEPPRIVACY2_DIR, FAMS_DIR, OURS_DIR]
FACE_CONF_THRESHOLD = 0.3
BBOX_EXPANSION      = 0.3


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _get_file_mapping(directory: str) -> Dict[str, str]:
    files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return {os.path.splitext(f)[0]: f for f in files}


def get_common_base_names(directories: List[str]) -> Tuple[List[str], List[Dict]]:
    mappings = [_get_file_mapping(d) for d in directories]
    common   = sorted(set(mappings[0]).intersection(*[m.keys() for m in mappings[1:]]))
    return common, mappings


# ---------------------------------------------------------------------------
# Face detection / cropping
# ---------------------------------------------------------------------------

def _expand_bbox(bbox: Dict, img_shape: Tuple, expansion: float = BBOX_EXPANSION) -> Dict:
    x, y, w, h   = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    ih, iw        = img_shape[:2]
    ew, eh        = int(w * expansion), int(h * expansion)
    nx, ny        = max(0, x - ew), max(0, y - eh)
    return {"x": nx, "y": ny, "w": min(iw - nx, w + 2 * ew), "h": min(ih - ny, h + 2 * eh)}


def _crop_face(img_path: str, bbox: Dict) -> Optional[np.ndarray]:
    img = cv2.imread(img_path)
    if img is None:
        return None
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    return img[y:y + h, x:x + w]


def _detect_face_bboxes(img_path: str) -> List[Tuple[Dict, float]]:
    img = cv2.imread(img_path)
    if img is None:
        return []
    try:
        face_objs = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface",
                                           enforce_detection=False, align=False)
    except Exception:
        return []
    return [
        (_expand_bbox(f["facial_area"], img.shape), f["confidence"])
        for f in face_objs if f["confidence"] >= FACE_CONF_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# ArcFace similarity
# ---------------------------------------------------------------------------

def _arcface_embedding(crop: np.ndarray) -> Optional[np.ndarray]:
    _, buf = cv2.imencode(".jpg", crop)
    tmp    = "tmp_arcface.jpg"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    try:
        result = DeepFace.represent(img_path=tmp, model_name="ArcFace", enforce_detection=False)
        return np.array(result[0]["embedding"])
    except Exception:
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))


def compute_similarity_scores(base_names: List[str], mappings: List[Dict]) -> Dict[str, List[float]]:
    scores = {m: [] for m in METHODS}

    for base_name in tqdm(base_names, desc="ArcFace"):
        original_path = os.path.join(ORIGINAL_DIR, mappings[0][base_name])
        face_bboxes   = _detect_face_bboxes(original_path)

        for bbox, _ in face_bboxes:
            orig_crop = _crop_face(original_path, bbox)
            if orig_crop is None:
                continue
            orig_emb = _arcface_embedding(orig_crop)
            if orig_emb is None:
                continue

            for i, method in enumerate(METHODS, start=1):
                path = os.path.join(DIRECTORIES[i], mappings[i][base_name])
                crop = _crop_face(path, bbox)
                if crop is None:
                    continue
                emb = _arcface_embedding(crop)
                if emb is not None:
                    scores[method].append(_cosine_similarity(orig_emb, emb))

    return scores


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_visualization(base_name: str, mappings: List[Dict], out_path: str = "arcface_similarity_sample.png"):
    original_path = os.path.join(ORIGINAL_DIR, mappings[0][base_name])
    face_bboxes   = _detect_face_bboxes(original_path)
    if not face_bboxes:
        return

    orig_crops = [(_crop_face(original_path, bbox), bbox) for bbox, _ in face_bboxes]
    orig_embs  = [(crop, _arcface_embedding(crop), bbox) for crop, bbox in orig_crops if crop is not None]

    titles = ["Original"] + METHODS
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, (dir_path, title) in enumerate(zip(DIRECTORIES, titles)):
        img = cv2.imread(os.path.join(dir_path, mappings[idx][base_name]))
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title, fontsize=14, fontweight="bold")
        axes[idx].axis("off")

        for _, orig_emb, bbox in orig_embs:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            color = "lime" if idx == 0 else "red"
            axes[idx].add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"))

            if idx > 0 and orig_emb is not None:
                crop = _crop_face(os.path.join(dir_path, mappings[idx][base_name]), bbox)
                if crop is not None:
                    emb = _arcface_embedding(crop)
                    if emb is not None:
                        sim = _cosine_similarity(orig_emb, emb)
                        axes[idx].text(x, y - 10, f"{sim:.3f}", color="red", fontsize=12,
                                       fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def build_results_table(scores: Dict[str, List[float]]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        arr = np.array(scores[method])
        rows.append({
            "Method":                method,
            "Mean Similarity ↑":     f"{np.mean(arr):.3f}",
            "Std":                   f"{np.std(arr):.3f}",
            "< 0.3 Strong Anon (%)": f"{np.mean(arr < 0.3) * 100:.1f}%",
            "< 0.5 Mod Anon (%)":    f"{np.mean(arr < 0.5) * 100:.1f}%",
            "Count":                 len(arr),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base_names, mappings = get_common_base_names(DIRECTORIES)
    print(f"Found {len(base_names)} common images\n")

    create_visualization(base_names[0], mappings)

    scores = compute_similarity_scores(base_names, mappings)

    df = build_results_table(scores)
    print(df.to_string(index=False))
    df.to_csv("arcface_similarity_results.csv", index=False)


if __name__ == "__main__":
    main()
