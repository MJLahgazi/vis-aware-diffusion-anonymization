import os
from typing import Dict, List, Optional, Tuple

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deepface import DeepFace
from scipy import linalg
from tqdm import tqdm

ORIGINAL_DIR     = "COCO/input"
DEEPPRIVACY2_DIR = "COCO/dp2"
FAMS_DIR         = "COCO/fams"
OURS_DIR         = "COCO/ours"

METHODS       = ["DeepPrivacy2", "FAMS", "Ours"]
DIRECTORIES   = [ORIGINAL_DIR, DEEPPRIVACY2_DIR, FAMS_DIR, OURS_DIR]
MIN_CROP_SIZE = 32
BBOX_EXPANSION = 0.3
FACE_CONF_THRESHOLD = 0.3

lpips_model = lpips.LPIPS(net="alex")
if torch.cuda.is_available():
    lpips_model = lpips_model.cuda()


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
    x, y, w, h     = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    img_h, img_w   = img_shape[:2]
    ew, eh         = int(w * expansion), int(h * expansion)
    nx, ny         = max(0, x - ew), max(0, y - eh)
    nx2, ny2       = min(img_w, x + w + ew), min(img_h, y + h + eh)
    return {"x": nx, "y": ny, "w": nx2 - nx, "h": ny2 - ny}


def _valid_bbox(bbox: Dict) -> bool:
    return bbox["w"] >= MIN_CROP_SIZE and bbox["h"] >= MIN_CROP_SIZE and bbox["x"] >= 0 and bbox["y"] >= 0


def _crop_face(img_path: str, bbox: Dict) -> Optional[np.ndarray]:
    img = cv2.imread(img_path)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    x  = max(0, min(bbox["x"], iw - 1))
    y  = max(0, min(bbox["y"], ih - 1))
    x2 = max(x + 1, min(x + bbox["w"], iw))
    y2 = max(y + 1, min(y + bbox["h"], ih))
    crop = img[y:y2, x:x2]
    return crop if crop.shape[0] >= MIN_CROP_SIZE and crop.shape[1] >= MIN_CROP_SIZE else None


def _detect_face_bboxes(img_path: str) -> List[Dict]:
    img = cv2.imread(img_path)
    if img is None:
        return []
    try:
        face_objs = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface",
                                           enforce_detection=False, align=False)
    except Exception:
        return []

    bboxes = []
    for f in face_objs:
        if f["confidence"] < FACE_CONF_THRESHOLD:
            continue
        expanded = _expand_bbox(f["facial_area"], img.shape)
        if _valid_bbox(expanded):
            bboxes.append(expanded)
    return bboxes


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def _to_lpips_tensor(crop: np.ndarray) -> Optional[torch.Tensor]:
    t = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return t.cuda() if torch.cuda.is_available() else t


def _compute_lpips(crop1: np.ndarray, crop2: np.ndarray) -> Optional[float]:
    t1, t2 = _to_lpips_tensor(crop1), _to_lpips_tensor(crop2)
    if t1.shape[2:] != t2.shape[2:]:
        h, w = max(t1.shape[2], t2.shape[2]), max(t1.shape[3], t2.shape[3])
        t1   = torch.nn.functional.interpolate(t1, size=(h, w), mode="bilinear", align_corners=False)
        t2   = torch.nn.functional.interpolate(t2, size=(h, w), mode="bilinear", align_corners=False)
    with torch.no_grad():
        return float(lpips_model(t1, t2).cpu().item())


def compute_lpips_scores(base_names: List[str], mappings: List[Dict]) -> Dict[str, List[float]]:
    scores = {m: [] for m in METHODS}
    for base_name in tqdm(base_names, desc="LPIPS"):
        original_path = os.path.join(ORIGINAL_DIR, mappings[0][base_name])
        bboxes        = _detect_face_bboxes(original_path)
        for bbox in bboxes:
            original_crop = _crop_face(original_path, bbox)
            if original_crop is None:
                continue
            for i, method in enumerate(METHODS, start=1):
                path = os.path.join(DIRECTORIES[i], mappings[i][base_name])
                crop = _crop_face(path, bbox)
                if crop is not None:
                    scores[method].append(_compute_lpips(original_crop, crop))
    return scores


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def _collect_face_crops(base_names: List[str], mappings: List[Dict]) -> Dict[str, List[np.ndarray]]:
    crops = {"Original": [], **{m: [] for m in METHODS}}
    for base_name in tqdm(base_names, desc="Collecting crops"):
        original_path = os.path.join(ORIGINAL_DIR, mappings[0][base_name])
        bboxes        = _detect_face_bboxes(original_path)
        for bbox in bboxes:
            batch, valid = [], True
            for i, key in enumerate(["Original"] + METHODS):
                path = os.path.join(DIRECTORIES[i], mappings[i][base_name])
                crop = _crop_face(path, bbox)
                if crop is None:
                    valid = False
                    break
                batch.append((key, cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (299, 299))))
            if valid:
                for key, crop in batch:
                    crops[key].append(crop)
    return crops


def _inception_activations(images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
    from torchvision.models import Inception_V3_Weights, inception_v3
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    arr  = np.array(images)
    feats = []

    for i in tqdm(range(0, len(arr), batch_size), desc="Inception features"):
        t = torch.from_numpy(arr[i:i+batch_size]).float().permute(0, 3, 1, 2) / 255.0
        t = (t - mean) / std
        if torch.cuda.is_available():
            t = t.cuda()
        with torch.no_grad():
            feats.append(model(t).cpu().numpy())

    return np.concatenate(feats, axis=0)


def _fid(mu1, s1, mu2, s2, eps=1e-6) -> float:
    s1 += np.eye(s1.shape[0]) * eps
    s2 += np.eye(s2.shape[0]) * eps
    diff    = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1 + np.eye(s1.shape[0]) * eps).dot(s2 + np.eye(s2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def compute_fid_scores(base_names: List[str], mappings: List[Dict]) -> Dict[str, float]:
    crops           = _collect_face_crops(base_names, mappings)
    orig_feats      = _inception_activations(crops["Original"])
    mu_o, sigma_o   = np.mean(orig_feats, axis=0), np.cov(orig_feats, rowvar=False)
    fid_scores      = {}
    for method in METHODS:
        feats              = _inception_activations(crops[method])
        mu_m, sigma_m      = np.mean(feats, axis=0), np.cov(feats, rowvar=False)
        fid_scores[method] = _fid(mu_o, sigma_o, mu_m, sigma_m)
    return fid_scores


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_visualization(base_name: str, mappings: List[Dict], out_path: str = "quality_assessment_sample.png"):
    original_path = os.path.join(ORIGINAL_DIR, mappings[0][base_name])
    bboxes        = _detect_face_bboxes(original_path)
    if not bboxes:
        return

    bbox          = bboxes[0]
    original_crop = _crop_face(original_path, bbox)
    if original_crop is None:
        return

    original_rgb = cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB)
    fig, axes    = plt.subplots(1, 4 + 1, figsize=(25, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(original_rgb)
    lpips_self = _compute_lpips(original_crop, original_crop)
    axes[1].set_title(f"Original (copy)\nLPIPS: {lpips_self:.6f}", fontweight="bold")
    axes[1].axis("off")

    for idx, method in enumerate(METHODS, start=2):
        path = os.path.join(DIRECTORIES[idx - 1], mappings[idx - 1][base_name])
        crop = _crop_face(path, bbox)
        if crop is None:
            axes[idx].text(0.5, 0.5, "Crop Failed", ha="center", va="center")
            axes[idx].axis("off")
            continue
        score = _compute_lpips(original_crop, crop)
        axes[idx].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"{method}\nLPIPS: {score:.4f}", fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("Quality Assessment — LPIPS on Face Crops (Lower = Better)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def build_results_table(lpips_scores: Dict, fid_scores: Optional[Dict] = None) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        arr = np.array(lpips_scores[method])
        row = {
            "Method":       method,
            "Mean LPIPS ↓": f"{np.mean(arr):.4f}",
            "Std LPIPS":    f"{np.std(arr):.4f}",
            "Median LPIPS": f"{np.median(arr):.4f}",
            "FID ↓":        f"{fid_scores[method]:.2f}" if fid_scores else "N/A",
            "Count":        len(arr),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base_names, mappings = get_common_base_names(DIRECTORIES)
    print(f"Found {len(base_names)} common images\n")

    create_visualization(base_names[0], mappings)

    lpips_scores = compute_lpips_scores(base_names, mappings)
    fid_scores   = compute_fid_scores(base_names, mappings)

    df = build_results_table(lpips_scores, fid_scores)
    print(df.to_string(index=False))
    df.to_csv("quality_assessment_results.csv", index=False)


if __name__ == "__main__":
    main()
