#!/usr/bin/env python3

import os
import sys
import glob
import yaml
import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "part1_parameters.yaml"


# ============================================================
# CONFIG LOADING
# ============================================================
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml(CONFIG_PATH)

DATASET_DIR = PROJECT_ROOT / CFG["paths"]["input_dir"]
OUTPUT_LABEL_DIR = PROJECT_ROOT / CFG["paths"]["output_label_dir"]
OUTPUT_VIZ_DIR = PROJECT_ROOT / CFG["paths"]["output_viz_dir"]

YOLO_SEG_MODEL_PATH = PROJECT_ROOT / CFG["models"]["yolo_seg_model"]
YOLO_AUX_MODEL_PATH = PROJECT_ROOT / CFG["models"]["yolo_aux_model"]
SAM3_ROOT = PROJECT_ROOT / CFG["models"]["sam3_root"]

DEVICE = CFG["runtime"]["device"]
IMG_SIZE = int(CFG["runtime"]["img_size"])
CONF_THRES = float(CFG["runtime"]["conf_thres"])
IOU_THRES = float(CFG["runtime"]["iou_thres"])

USE_SAM3 = bool(CFG["runtime"]["use_sam3"])
SAM3_CONFIDENCE = float(CFG["runtime"]["sam3_confidence"])
SAVE_VIZ = bool(CFG["runtime"]["save_viz"])

MIN_MASK_AREA = int(CFG["merge"]["min_mask_area"])
IOU_MATCH_THRESH = float(CFG["merge"]["iou_match_thresh"])

CLASS_NAME_TO_ID = dict(CFG["classes"]["name_to_id"])

YOLO_SEG_LABELS = set(CFG["classes"]["yolo26_labels"])
YOLO_AUX_LABELS = set(CFG["classes"]["yolo11_labels"])
SAM3_PROMPTS = list(CFG["classes"]["sam3_prompts"])


# ============================================================
# TORCH / SAM3 SETUP
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

# Make local SAM3 repo importable
sys.path.insert(0, str(SAM3_ROOT))

import sam3  # noqa: E402
from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


# ============================================================
# UTILS
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


def normalize_points(points_xy: np.ndarray, w: int, h: int):
    pts = points_xy.astype(np.float32).copy()
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    return pts


def order_corners_clockwise(pts: np.ndarray):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]

    sums = pts[:, 0] + pts[:, 1]
    start_idx = np.argmin(sums)
    pts = np.roll(pts, -start_idx, axis=0)
    return pts


def xyxy_to_obb(xyxy: np.ndarray):
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)
    return order_corners_clockwise(pts)


def mask_to_obb(mask: np.ndarray):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < MIN_MASK_AREA:
        return None, area

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_corners_clockwise(box)
    return box.astype(np.float32), area


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / union


def resize_mask(mask: np.ndarray, image_shape):
    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def save_yolo_obb_txt(txt_path: Path, rows):
    with open(txt_path, "w", encoding="utf-8") as f:
        for class_id, pts_norm in rows:
            vals = [str(class_id)] + [f"{v:.6f}" for v in pts_norm.reshape(-1)]
            f.write(" ".join(vals) + "\n")


def draw_overlay(image: np.ndarray, detections: list):
    out = image.copy()

    for det in detections:
        cls_id = det["class_id"]
        score = det.get("score", 0.0)
        label = det.get("label", str(cls_id))
        color = tuple(int(c) for c in det.get("color", (0, 255, 255)))

        if "mask" in det and det["mask"] is not None:
            mask = det["mask"]
            color_mask = np.zeros_like(out)
            color_mask[:, :, 1] = (mask > 0).astype(np.uint8) * 180
            out = cv2.addWeighted(out, 1.0, color_mask, 0.25, 0)

        obb = det["obb"]
        obb_i = obb.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [obb_i], True, color, 2)

        x, y = obb_i[0, 0]
        text = f"{label} ({cls_id}) {score:.2f}"
        cv2.putText(out, text, (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out


# ============================================================
# YOLO 26 SEGMENTATION MODEL
# ============================================================
def get_yolo_seg_masks(model: YOLO, image_bgr: np.ndarray):
    results = model.predict(
        source=image_bgr,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    detections = []
    if not results:
        return detections

    r = results[0].cpu()
    if r.boxes is None or r.masks is None:
        return detections

    names = model.names

    try:
        masks_np = r.masks.data.numpy()
    except Exception:
        masks_np = np.array(r.masks.data)

    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].item())
        conf = float(r.boxes.conf[i].item())
        cls_name = str(names[cls_id]).lower().strip()

        if cls_name not in YOLO_SEG_LABELS:
            continue
        if cls_name not in CLASS_NAME_TO_ID:
            continue

        mask = (masks_np[i] > 0.5).astype(np.uint8)
        mask = resize_mask(mask, image_bgr.shape)

        if mask.sum() < MIN_MASK_AREA:
            continue

        detections.append({
            "source": "yolo26_seg",
            "label": cls_name,
            "class_id": CLASS_NAME_TO_ID[cls_name],
            "score": conf,
            "mask": mask,
        })

    return detections


# ============================================================
# YOLO 11 MODEL
# Supports seg / obb / boxes
# ============================================================
def get_yolo_aux_obbs(model: YOLO, image_bgr: np.ndarray):
    results = model.predict(
        source=image_bgr,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    detections = []
    if not results:
        return detections

    r = results[0].cpu()
    names = model.names

    # Case 1: segmentation output
    if r.boxes is not None and r.masks is not None:
        try:
            masks_np = r.masks.data.numpy()
        except Exception:
            masks_np = np.array(r.masks.data)

        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            cls_name = str(names[cls_id]).lower().strip()

            if cls_name not in YOLO_AUX_LABELS:
                continue
            if cls_name not in CLASS_NAME_TO_ID:
                continue

            mask = (masks_np[i] > 0.5).astype(np.uint8)
            mask = resize_mask(mask, image_bgr.shape)
            obb, area = mask_to_obb(mask)
            if obb is None:
                continue

            detections.append({
                "source": "yolo11_seg",
                "label": cls_name,
                "class_id": CLASS_NAME_TO_ID[cls_name],
                "score": conf,
                "mask": mask,
                "obb": obb,
            })

        return detections

    # Case 2: OBB output
    if getattr(r, "obb", None) is not None and r.obb is not None:
        obb_obj = r.obb
        cls_arr = obb_obj.cls.numpy()
        conf_arr = obb_obj.conf.numpy()
        corners_arr = obb_obj.xyxyxyxy.numpy()

        for i in range(len(cls_arr)):
            cls_id = int(cls_arr[i])
            conf = float(conf_arr[i])
            cls_name = str(names[cls_id]).lower().strip()

            if cls_name not in YOLO_AUX_LABELS:
                continue
            if cls_name not in CLASS_NAME_TO_ID:
                continue

            detections.append({
                "source": "yolo11_obb",
                "label": cls_name,
                "class_id": CLASS_NAME_TO_ID[cls_name],
                "score": conf,
                "mask": None,
                "obb": order_corners_clockwise(corners_arr[i].astype(np.float32)),
            })

        return detections

    # Case 3: regular boxes output
    if r.boxes is not None:
        boxes_xyxy = r.boxes.xyxy.numpy()
        cls_arr = r.boxes.cls.numpy()
        conf_arr = r.boxes.conf.numpy()

        for i in range(len(r.boxes)):
            cls_id = int(cls_arr[i])
            conf = float(conf_arr[i])
            cls_name = str(names[cls_id]).lower().strip()

            if cls_name not in YOLO_AUX_LABELS:
                continue
            if cls_name not in CLASS_NAME_TO_ID:
                continue

            obb = xyxy_to_obb(boxes_xyxy[i])

            detections.append({
                "source": "yolo11_box",
                "label": cls_name,
                "class_id": CLASS_NAME_TO_ID[cls_name],
                "score": conf,
                "mask": None,
                "obb": obb,
            })

    return detections


# ============================================================
# SAM3
# ============================================================
class Sam3BatchSegmenter:
    def __init__(self):
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        self.model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(self.model)

    def segment_prompt(self, pil_image, prompt, conf_thresh=0.45):
        state = self.processor.set_image(pil_image)
        state = self.processor.set_confidence_threshold(conf_thresh, state)
        state = self.processor.set_text_prompt(prompt, state)

        detections = []

        masks = state.get("masks", [])
        scores = state.get("scores", [])

        for i in range(len(masks)):
            try:
                mask_np = masks[i][0].cpu().numpy()
            except Exception:
                mask_np = masks[i][0].numpy()

            mask_np = (mask_np > 0.5).astype(np.uint8)
            if mask_np.sum() < MIN_MASK_AREA:
                continue

            score = 0.0
            try:
                score = float(scores[i].item())
            except Exception:
                try:
                    score = float(scores[i])
                except Exception:
                    pass

            detections.append({
                "source": "sam3",
                "label": prompt,
                "class_id": CLASS_NAME_TO_ID[prompt],
                "score": score,
                "mask": mask_np,
            })

        return detections


# ============================================================
# MERGING
# YOLO26 + SAM3 only for classes in SAM3 prompts
# ============================================================
def merge_yolo_sam(yolo_dets, sam_dets, image_shape):
    final_dets = []
    used_sam = set()

    for yd in yolo_dets:
        ymask = resize_mask(yd["mask"], image_shape)
        merged_mask = ymask.copy()
        best_score = yd["score"]

        for si, sd in enumerate(sam_dets):
            if sd["class_id"] != yd["class_id"]:
                continue

            smask = resize_mask(sd["mask"], image_shape)
            iou = mask_iou(ymask, smask)

            if iou >= IOU_MATCH_THRESH:
                merged_mask = np.logical_or(merged_mask > 0, smask > 0).astype(np.uint8)
                best_score = max(best_score, sd["score"])
                used_sam.add(si)

        final_dets.append({
            "label": yd["label"],
            "class_id": yd["class_id"],
            "score": best_score,
            "mask": merged_mask,
        })

    for si, sd in enumerate(sam_dets):
        if si in used_sam:
            continue

        smask = resize_mask(sd["mask"], image_shape)
        final_dets.append({
            "label": sd["label"],
            "class_id": sd["class_id"],
            "score": sd["score"],
            "mask": smask,
        })

    return final_dets


def convert_mask_detections_to_obb(detections, image_shape):
    h, w = image_shape[:2]
    rows = []
    kept = []

    for det in detections:
        mask = det["mask"]
        obb, area = mask_to_obb(mask)
        if obb is None:
            continue

        obb_norm = normalize_points(obb, w, h)
        rows.append((det["class_id"], obb_norm))

        kept.append({
            "label": det["label"],
            "class_id": det["class_id"],
            "score": det["score"],
            "mask": mask,
            "obb": obb,
        })

    return rows, kept


def convert_direct_obb_detections(detections, image_shape):
    h, w = image_shape[:2]
    rows = []
    kept = []

    for det in detections:
        obb = det["obb"]
        obb_norm = normalize_points(obb, w, h)
        rows.append((det["class_id"], obb_norm))

        kept.append({
            "label": det["label"],
            "class_id": det["class_id"],
            "score": det["score"],
            "mask": det.get("mask", None),
            "obb": obb,
        })

    return rows, kept


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUTPUT_LABEL_DIR)
    if SAVE_VIZ:
        ensure_dir(OUTPUT_VIZ_DIR)

    image_paths = list_images(DATASET_DIR)
    if not image_paths:
        print(f"[WARN] No images found in {DATASET_DIR}")
        return

    print("[INFO] Configuration loaded from:", CONFIG_PATH)
    print("[INFO] Input directory:", DATASET_DIR)
    print("[INFO] Output labels:", OUTPUT_LABEL_DIR)
    print("[INFO] Output visualizations:", OUTPUT_VIZ_DIR)
    print("[INFO] YOLO26 model (boat/person):", YOLO_SEG_MODEL_PATH)
    print("[INFO] YOLO11 model (sam/buoy/lolo/catamaran):", YOLO_AUX_MODEL_PATH)
    print("[INFO] SAM3 root:", SAM3_ROOT)

    print("[INFO] Loading YOLO26 segmentation model...")
    yolo_seg_model = YOLO(str(YOLO_SEG_MODEL_PATH))

    print("[INFO] Loading YOLO11 auxiliary model...")
    yolo_aux_model = YOLO(str(YOLO_AUX_MODEL_PATH))

    sam3_segmenter = None
    if USE_SAM3:
        print("[INFO] Loading SAM3...")
        sam3_segmenter = Sam3BatchSegmenter()

    for img_path in image_paths:
        print(f"[INFO] Processing {img_path.name}")

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Part A: YOLO26 segmentation for boat/person
        yolo_seg_dets = get_yolo_seg_masks(yolo_seg_model, image_bgr)

        # Part B: SAM3 prompts for boat/person
        sam_dets = []
        if sam3_segmenter is not None:
            for prompt in SAM3_PROMPTS:
                if prompt not in CLASS_NAME_TO_ID:
                    continue
                prompt_dets = sam3_segmenter.segment_prompt(
                    pil_image,
                    prompt,
                    conf_thresh=SAM3_CONFIDENCE
                )
                sam_dets.extend(prompt_dets)

        # Merge boat/person branch
        merged_mask_dets = merge_yolo_sam(yolo_seg_dets, sam_dets, image_bgr.shape)
        mask_rows, mask_viz_dets = convert_mask_detections_to_obb(merged_mask_dets, image_bgr.shape)

        # Part C: YOLO11 branch for sam/buoy/lolo/catamaran
        aux_dets = get_yolo_aux_obbs(yolo_aux_model, image_bgr)
        aux_rows, aux_viz_dets = convert_direct_obb_detections(aux_dets, image_bgr.shape)

        # Combine outputs
        all_rows = mask_rows + aux_rows
        all_viz_dets = mask_viz_dets + aux_viz_dets

        # Save labels
        txt_path = OUTPUT_LABEL_DIR / f"{img_path.stem}.txt"
        save_yolo_obb_txt(txt_path, all_rows)

        # Save visualization
        if SAVE_VIZ:
            viz = draw_overlay(image_bgr, all_viz_dets)
            viz_path = OUTPUT_VIZ_DIR / f"{img_path.stem}.jpg"
            cv2.imwrite(str(viz_path), viz)

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()