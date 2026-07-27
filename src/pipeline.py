import os
import warnings
from collections import namedtuple
from typing import List, Optional

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import cv2
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from tqdm import tqdm
from ultralytics import YOLO

from compositing import composite_result
from config import (
    DEVICE,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    PERSON_DETECTION_CONF,
    VISIBILITY_THRESHOLD,
)
from inpainting import run_inpainting
from masking import determine_cutoff_y, extract_head_mask, get_person_segmentation
from utils import get_image_files
from visibility import compute_visibility_score, detect_face, detect_keypoints


PersonResult = namedtuple("PersonResult", ["person_id", "bbox", "head_mask", "visibility_score"])


def _process_person(image, bbox, person_id, model_pose, model_seg) -> PersonResult:
    face_bbox, face_conf             = detect_face(image, bbox)
    kp_count, kp_positions, kp_confs = detect_keypoints(image, bbox, model_pose)

    visibility_score = compute_visibility_score(face_conf, kp_count)
    person_mask      = get_person_segmentation(image, bbox, model_seg)
    cutoff_y         = determine_cutoff_y(bbox, face_bbox, face_conf, kp_count, kp_positions, kp_confs, image.shape[0])
    head_mask        = extract_head_mask(person_mask, cutoff_y, bbox, face_bbox, face_conf, kp_count, kp_positions, kp_confs, image.shape)

    return PersonResult(person_id, bbox, head_mask, visibility_score)


def _detect_all_persons(image, model_person, model_pose, model_seg) -> List[PersonResult]:
    results = model_person(image, conf=PERSON_DETECTION_CONF, iou=0.5, verbose=False, device=DEVICE)

    persons, person_id = [], 1
    for result in results:
        if result.boxes is None:
            continue

        for j, box in enumerate(result.boxes.xyxy.cpu().numpy()):
            if result.boxes.cls[j].item() != 0:
                continue
            persons.append(_process_person(image, tuple(map(int, box)), person_id, model_pose, model_seg))
            person_id += 1

    return persons


def process_image(image, model_person, model_pose, model_seg, sdxl_pipe) -> Optional:
    persons  = _detect_all_persons(image, model_person, model_pose, model_seg)
    high_vis = [p for p in persons if p.visibility_score > VISIBILITY_THRESHOLD]

    if not high_vis:
        return None

    crops  = [run_inpainting(image, p.head_mask, sdxl_pipe) for p in high_vis]
    result = image.copy()
    for crop_bbox, inpainted_crop, crop_mask in crops:
        result = composite_result(result, inpainted_crop, crop_bbox, crop_mask)

    return result


def batch_process(input_folder, output_folder, model_person, model_pose, model_seg, sdxl_pipe):
    image_files = get_image_files(input_folder)
    if not image_files:
        print(f"No supported images found in {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for image_path in tqdm(image_files, desc="Processing"):
        image  = cv2.imread(image_path)
        result = process_image(image, model_person, model_pose, model_seg, sdxl_pipe)
        if result is not None:
            cv2.imwrite(os.path.join(output_folder, os.path.basename(image_path)), result)


def main():
    print("Loading models...")
    model_person = YOLO("yolo11n.pt")
    model_seg    = YOLO("yolo11n-seg.pt")
    model_pose   = YOLO("yolo11n-pose.pt")

    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    sdxl_pipe   = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch_dtype,
        variant="fp16" if DEVICE == "cuda" else None,
    ).to(DEVICE)
    print("Models loaded.\n")

    batch_process(INPUT_FOLDER, OUTPUT_FOLDER, model_person, model_pose, model_seg, sdxl_pipe)


if __name__ == "__main__":
    main()
