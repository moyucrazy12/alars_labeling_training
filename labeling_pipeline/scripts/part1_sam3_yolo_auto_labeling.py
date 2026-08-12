#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ============================================================
# PATHS / CONFIG LOADING
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = PIPELINE_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "part_parameters.yaml"

CONFIG_PATH = DEFAULT_CONFIG_PATH
CFG = {}

DATASET_DIR = None
OUTPUT_LABEL_DIR = None
OUTPUT_VIZ_DIR = None

YOLO26_MODEL_PATH = None
YOLO11_MODEL_PATH = None
SAM3_ROOT = None

DEVICE = 0
IMG_SIZE = 1024
IOU_THRES = 0.45
SAVE_VIZ = True

LEGACY_CONFIDENCE = 0.45
LABEL_WRITE_MODE = "append_unique"
USE_OBB = True

MIN_MASK_AREA = 20
IOU_MATCH_THRESH = 0.15

CLASS_ID_TO_NAME = {}
CLASS_NAME_TO_ID = {}

USE_YOLO26 = False
YOLO26_CONFIDENCE = 0.45
YOLO26_LABELS = set()

USE_YOLO11 = False
YOLO11_CONFIDENCE = 0.45
YOLO11_LABELS = set()

USE_SAM3 = False
SAM3_CONFIDENCE = 0.45
SAM3_PROMPTS = []

# ============================================================
# CONFIG LOADING
# ============================================================
def resolve_config_path(config_path_arg) -> Path:
    """
    Resolve config paths in a forgiving way.

    Supports:
      --config labeling_pipeline/config/<file>.yaml
      --config config/<file>.yaml
      --config /absolute/path/to/<file>.yaml
    """
    path = Path(config_path_arg).expanduser()

    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        PROJECT_ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the root-relative path for a clear FileNotFoundError.
    return (Path.cwd() / path).resolve()


def resolve_project_path(path_value) -> Path:
    """
    Resolve paths from the YAML.

    Paths like:
      dataset_to_label/images
      models/sam2
    are interpreted relative to labeling_pipeline/.

    Paths like:
      labeling_pipeline/dataset_to_label/images
    are interpreted relative to the repository root.
    """
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        PROJECT_ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if path.parts and path.parts[0] == "labeling_pipeline":
        return (REPO_ROOT / path).resolve()

    return (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def validate_configured_classes(section_name: str, class_names):
    unknown = sorted(set(class_names) - set(CLASS_NAME_TO_ID))
    if unknown:
        raise ValueError(
            f"Unknown class names in {section_name}: {unknown}. "
            "Add them to classes.id_to_name first."
        )

def validate_confidence(name: str, value: float):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

def load_config(config_path: Path):
    global CONFIG_PATH, CFG
    global DATASET_DIR, OUTPUT_LABEL_DIR, OUTPUT_VIZ_DIR
    global YOLO26_MODEL_PATH, YOLO11_MODEL_PATH, SAM3_ROOT
    global DEVICE, IMG_SIZE, IOU_THRES, SAVE_VIZ
    global LEGACY_CONFIDENCE, LABEL_WRITE_MODE, USE_OBB
    global MIN_MASK_AREA, IOU_MATCH_THRESH
    global CLASS_ID_TO_NAME, CLASS_NAME_TO_ID
    global USE_YOLO26, YOLO26_CONFIDENCE, YOLO26_LABELS
    global USE_YOLO11, YOLO11_CONFIDENCE, YOLO11_LABELS
    global USE_SAM3, SAM3_CONFIDENCE, SAM3_PROMPTS

    CONFIG_PATH = resolve_config_path(config_path)
    CFG = load_yaml(CONFIG_PATH)

    DATASET_DIR = resolve_project_path(CFG["paths"]["input_dir"])
    OUTPUT_LABEL_DIR = resolve_project_path(CFG["paths"]["output_label_dir"])
    OUTPUT_VIZ_DIR = resolve_project_path(CFG["paths"].get("output_viz_dir", "visualizations_part1"))

    YOLO26_MODEL_PATH = resolve_project_path(CFG["models"]["yolo26_model"])
    YOLO11_MODEL_PATH = resolve_project_path(CFG["models"]["yolo11_model"])
    SAM3_ROOT = resolve_project_path(CFG["models"]["sam3_root"])

    runtime_cfg = CFG.get("runtime", {})
    DEVICE = runtime_cfg.get("device", 0)
    IMG_SIZE = int(runtime_cfg.get("img_size", 1024))
    IOU_THRES = float(runtime_cfg.get("iou_thres", 0.45))
    SAVE_VIZ = bool(runtime_cfg.get("save_viz", True))

    # Keep backward compatibility with older configs that used
    # runtime.conf_thres as one shared YOLO confidence threshold.
    LEGACY_CONFIDENCE = float(runtime_cfg.get("conf_thres", 0.45))

    output_cfg = CFG.get("output", {})

    LABEL_WRITE_MODE = str(
        output_cfg.get("label_write_mode", "append_unique")
    ).lower().strip()
    if LABEL_WRITE_MODE not in {"overwrite", "append", "append_unique"}:
        raise ValueError(
            "output.label_write_mode must be one of: "
            "overwrite, append, append_unique"
        )

    raw_use_obb = output_cfg.get("use_obb", True)
    if not isinstance(raw_use_obb, bool):
        raise ValueError(
            "output.use_obb must be a YAML boolean: true or false"
        )
    USE_OBB = raw_use_obb

    MIN_MASK_AREA = int(CFG["merge"].get("min_mask_area", 20))
    IOU_MATCH_THRESH = float(CFG["merge"].get("iou_match_thresh", 0.15))

    CLASS_ID_TO_NAME = {
        int(class_id): str(class_name).lower().strip()
        for class_id, class_name in CFG["classes"]["id_to_name"].items()
    }
    CLASS_NAME_TO_ID = {
        class_name: class_id for class_id, class_name in CLASS_ID_TO_NAME.items()
    }

    if len(CLASS_NAME_TO_ID) != len(CLASS_ID_TO_NAME):
        raise ValueError("Class names in classes.id_to_name must be unique")

    yolo26_cfg = CFG.get("yolo26", {})
    USE_YOLO26 = bool(yolo26_cfg.get("enabled", False))
    YOLO26_CONFIDENCE = float(yolo26_cfg.get("confidence", LEGACY_CONFIDENCE))
    YOLO26_LABELS = {
        str(label).lower().strip() for label in yolo26_cfg.get("labels", [])
    }

    yolo11_cfg = CFG.get("yolo11", {})
    USE_YOLO11 = bool(yolo11_cfg.get("enabled", False))
    YOLO11_CONFIDENCE = float(yolo11_cfg.get("confidence", LEGACY_CONFIDENCE))
    YOLO11_LABELS = {
        str(label).lower().strip() for label in yolo11_cfg.get("labels", [])
    }

    sam3_cfg = CFG.get("sam3", {})
    USE_SAM3 = bool(sam3_cfg.get("enabled", False))
    SAM3_CONFIDENCE = float(sam3_cfg.get("confidence", LEGACY_CONFIDENCE))
    SAM3_PROMPTS = [
        str(prompt).lower().strip() for prompt in sam3_cfg.get("prompts", [])
    ]

    validate_configured_classes("yolo26.labels", YOLO26_LABELS)
    validate_configured_classes("yolo11.labels", YOLO11_LABELS)
    validate_configured_classes("sam3.prompts", SAM3_PROMPTS)

    validate_confidence("yolo26.confidence", YOLO26_CONFIDENCE)
    validate_confidence("yolo11.confidence", YOLO11_CONFIDENCE)
    validate_confidence("sam3.confidence", SAM3_CONFIDENCE)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Part 1 automatic labeling with YOLO/SAM3 and selectable "
            "YOLO OBB or normal BB labels."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file."
    )
    return parser.parse_args()

# ============================================================
# TORCH / SAM3 SETUP
# ============================================================
def configure_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

# SAM3 is imported lazily inside Sam3BatchSegmenter only when enabled.

# ============================================================
# UTILS
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def list_images_recursive(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )

def get_relative_output_paths(img_path: Path, input_root: Path, label_root: Path, viz_root: Path):
    rel_path = img_path.relative_to(input_root)
    txt_path = label_root / rel_path.with_suffix(".txt")
    viz_path = viz_root / rel_path.with_suffix(".jpg")
    return txt_path, viz_path

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

def obb_to_xyxy(obb: np.ndarray) -> np.ndarray:
    """Convert four OBB corner points to an axis-aligned xyxy box."""
    x_min = float(np.min(obb[:, 0]))
    y_min = float(np.min(obb[:, 1]))
    x_max = float(np.max(obb[:, 0]))
    y_max = float(np.max(obb[:, 1]))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def xyxy_to_normalized_xywh(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert pixel xyxy coordinates to normalized YOLO xywh coordinates."""
    x1, y1, x2, y2 = xyxy.astype(np.float32)

    x1 = float(np.clip(x1, 0.0, float(w)))
    x2 = float(np.clip(x2, 0.0, float(w)))
    y1 = float(np.clip(y1, 0.0, float(h)))
    y2 = float(np.clip(y2, 0.0, float(h)))

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0

    return np.array(
        [
            x_center / float(w),
            y_center / float(h),
            box_w / float(w),
            box_h / float(h),
        ],
        dtype=np.float32,
    )

def format_yolo_row(class_id: int, coords_norm: np.ndarray) -> str:
    """Format either an OBB row or a normal detection row."""
    flat_coords = np.asarray(coords_norm, dtype=np.float32).reshape(-1)
    expected_count = 8 if USE_OBB else 4

    if len(flat_coords) != expected_count:
        format_name = "OBB" if USE_OBB else "normal BB"
        raise ValueError(
            f"Expected {expected_count} coordinates for {format_name}, "
            f"got {len(flat_coords)}"
        )

    values = [str(class_id)] + [f"{value:.6f}" for value in flat_coords]
    return " ".join(values)

def canonicalize_yolo_line(line: str):
    """Normalize a row for duplicate checks in the selected output format."""
    parts = line.strip().split()
    if not parts:
        return None

    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        # Preserve malformed or non-standard existing lines without treating
        # them as generated detections.
        return line.strip()

    expected_count = 8 if USE_OBB else 4
    if len(coords) != expected_count:
        return line.strip()

    return " ".join(
        [str(class_id)] + [f"{value:.6f}" for value in coords]
    )

def validate_existing_label_format(txt_path: Path, existing_text: str):
    """Prevent mixing OBB and normal-BB labels in one annotation file."""
    expected_fields = 9 if USE_OBB else 5
    expected_name = "OBB" if USE_OBB else "normal BB"

    for line_number, raw_line in enumerate(existing_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != expected_fields:
            raise ValueError(
                f"Existing label format mismatch in {txt_path}:{line_number}. "
                f"output.use_obb={USE_OBB} expects {expected_fields} fields "
                f"per row ({expected_name}), but found {len(parts)}. "
                "Use a different output_label_dir or set "
                "output.label_write_mode: overwrite when changing formats."
            )

        try:
            [float(value) for value in parts]
        except ValueError as error:
            raise ValueError(
                f"Non-numeric label row in {txt_path}:{line_number}: {line}"
            ) from error

def save_yolo_txt(txt_path: Path, rows):
    """Save OBB or normal-BB detections according to the configured mode."""
    ensure_dir(txt_path.parent)
    new_lines = [format_yolo_row(class_id, coords_norm) for class_id, coords_norm in rows]

    if LABEL_WRITE_MODE == "overwrite":
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")
        return len(new_lines)

    existing_text = ""
    if txt_path.exists():
        existing_text = txt_path.read_text(encoding="utf-8")
        validate_existing_label_format(txt_path, existing_text)

    if LABEL_WRITE_MODE == "append_unique":
        existing_rows = {
            canonical
            for line in existing_text.splitlines()
            if (canonical := canonicalize_yolo_line(line)) is not None
        }

        filtered_lines = []
        for line in new_lines:
            canonical = canonicalize_yolo_line(line)
            if canonical in existing_rows:
                continue
            existing_rows.add(canonical)
            filtered_lines.append(line)
        new_lines = filtered_lines

    if not new_lines:
        return 0

    needs_leading_newline = bool(existing_text) and not existing_text.endswith("\n")
    with open(txt_path, "a", encoding="utf-8") as f:
        if needs_leading_newline:
            f.write("\n")
        for line in new_lines:
            f.write(line + "\n")

    return len(new_lines)

def draw_overlay(image: np.ndarray, detections: list):
    """Draw masks and either oriented or axis-aligned boxes."""
    out = image.copy()

    for det in detections:
        cls_id = det["class_id"]
        score = det.get("score", 0.0)
        label = det.get("label", str(cls_id))
        color = tuple(int(c) for c in det.get("color", (0, 255, 255)))

        if det.get("mask") is not None:
            mask = det["mask"]
            color_mask = np.zeros_like(out)
            color_mask[:, :, 1] = (mask > 0).astype(np.uint8) * 180
            out = cv2.addWeighted(out, 1.0, color_mask, 0.25, 0)

        obb = det["obb"].astype(np.float32)

        if USE_OBB:
            obb_i = obb.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [obb_i], True, color, 2)
            text_x, text_y = obb_i[0, 0]
        else:
            x1, y1, x2, y2 = obb_to_xyxy(obb).astype(np.int32)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text_x, text_y = x1, y1

        text = f"{label} ({cls_id}) {score:.2f}"
        cv2.putText(out, text, (int(text_x), max(18, int(text_y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out

# ============================================================
# YOLO 26 SEGMENTATION MODEL
# ============================================================
def get_yolo_seg_masks(model: YOLO, image_bgr: np.ndarray):
    results = model.predict(
        source=image_bgr,
        imgsz=IMG_SIZE,
        conf=YOLO26_CONFIDENCE,
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

        if cls_name not in YOLO26_LABELS:
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
        conf=YOLO11_CONFIDENCE,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    detections = []
    if not results:
        return detections

    r = results[0].cpu()
    names = model.names

    if r.boxes is not None and r.masks is not None:
        try:
            masks_np = r.masks.data.numpy()
        except Exception:
            masks_np = np.array(r.masks.data)

        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            cls_name = str(names[cls_id]).lower().strip()

            if cls_name not in YOLO11_LABELS:
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

    if getattr(r, "obb", None) is not None and r.obb is not None:
        obb_obj = r.obb
        cls_arr = obb_obj.cls.numpy()
        conf_arr = obb_obj.conf.numpy()
        corners_arr = obb_obj.xyxyxyxy.numpy()

        for i in range(len(cls_arr)):
            cls_id = int(cls_arr[i])
            conf = float(conf_arr[i])
            cls_name = str(names[cls_id]).lower().strip()

            if cls_name not in YOLO11_LABELS:
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

    if r.boxes is not None:
        boxes_xyxy = r.boxes.xyxy.numpy()
        cls_arr = r.boxes.cls.numpy()
        conf_arr = r.boxes.conf.numpy()

        for i in range(len(r.boxes)):
            cls_id = int(cls_arr[i])
            conf = float(conf_arr[i])
            cls_name = str(names[cls_id]).lower().strip()

            if cls_name not in YOLO11_LABELS:
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
    def __init__(self, sam3_repo_root: Path):
        # Do not require SAM3 to be installed/importable when it is disabled.
        sys.path.insert(0, str(sam3_repo_root))

        # Import SAM3 lazily to avoid requiring it when not used.
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        installed_sam3_root = Path(sam3.__file__).resolve().parent.parent
        bpe_path = installed_sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        self.model = build_sam3_image_model(bpe_path=str(bpe_path))
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

def output_coords_from_obb(obb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Return normalized OBB or normal-BB coordinates from pixel OBB points."""
    if USE_OBB:
        return normalize_points(obb, w, h)

    xyxy = obb_to_xyxy(obb)
    return xyxy_to_normalized_xywh(xyxy, w, h)

def convert_mask_detections(detections, image_shape):
    h, w = image_shape[:2]
    rows = []
    kept = []

    for det in detections:
        mask = det["mask"]
        obb, _ = mask_to_obb(mask)
        if obb is None:
            continue

        coords_norm = output_coords_from_obb(obb, w, h)
        rows.append((det["class_id"], coords_norm))

        kept.append({
            "label": det["label"],
            "class_id": det["class_id"],
            "score": det["score"],
            "mask": mask,
            "obb": obb,
        })

    return rows, kept

def convert_direct_detections(detections, image_shape):
    h, w = image_shape[:2]
    rows = []
    kept = []

    for det in detections:
        obb = det["obb"]
        coords_norm = output_coords_from_obb(obb, w, h)
        rows.append((det["class_id"], coords_norm))

        kept.append({
            "label": det["label"],
            "class_id": det["class_id"],
            "score": det["score"],
            "mask": det.get("mask"),
            "obb": obb,
        })

    return rows, kept

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    load_config(args.config)
    configure_torch()

    ensure_dir(OUTPUT_LABEL_DIR)
    if SAVE_VIZ:
        ensure_dir(OUTPUT_VIZ_DIR)

    image_paths = list_images_recursive(DATASET_DIR)
    if not image_paths:
        print(f"[WARN] No images found in {DATASET_DIR}")
        return

    print("[INFO] Configuration loaded from:", CONFIG_PATH)
    print("[INFO] Input directory:", DATASET_DIR)
    print("[INFO] Output labels:", OUTPUT_LABEL_DIR)
    print("[INFO] Save visualizations:", SAVE_VIZ)
    if SAVE_VIZ:
        print("[INFO] Output visualizations:", OUTPUT_VIZ_DIR)
    print("[INFO] Label write mode:", LABEL_WRITE_MODE)
    print("[INFO] Bounding-box format:", "OBB" if USE_OBB else "normal BB")
    print("[INFO] Number of images found:", len(image_paths))
    print(
        f"[INFO] YOLO26 enabled: {USE_YOLO26}; "
        f"confidence: {YOLO26_CONFIDENCE}; labels: {sorted(YOLO26_LABELS)}"
    )
    print(
        f"[INFO] YOLO11 enabled: {USE_YOLO11}; "
        f"confidence: {YOLO11_CONFIDENCE}; labels: {sorted(YOLO11_LABELS)}"
    )
    print(
        f"[INFO] SAM3 enabled: {USE_SAM3}; "
        f"confidence: {SAM3_CONFIDENCE}; prompts: {SAM3_PROMPTS}"
    )

    if not any((USE_YOLO26, USE_YOLO11, USE_SAM3)):
        raise ValueError("At least one of yolo26, yolo11, or sam3 must be enabled")

    yolo26_model = None
    if USE_YOLO26:
        print("[INFO] Loading YOLO26 segmentation model:", YOLO26_MODEL_PATH)
        yolo26_model = YOLO(str(YOLO26_MODEL_PATH))

    yolo11_model = None
    if USE_YOLO11:
        print("[INFO] Loading YOLO11 auxiliary model:", YOLO11_MODEL_PATH)
        yolo11_model = YOLO(str(YOLO11_MODEL_PATH))

    sam3_segmenter = None
    if USE_SAM3:
        print("[INFO] Loading SAM3 from:", SAM3_ROOT)
        sam3_segmenter = Sam3BatchSegmenter(SAM3_ROOT)

    for img_path in image_paths:
        rel_img = img_path.relative_to(DATASET_DIR)
        print(f"[INFO] Processing {rel_img}")

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        yolo_seg_dets = []
        if yolo26_model is not None:
            yolo_seg_dets = get_yolo_seg_masks(yolo26_model, image_bgr)

        sam_dets = []
        if sam3_segmenter is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            for prompt in SAM3_PROMPTS:
                prompt_dets = sam3_segmenter.segment_prompt(
                    pil_image,
                    prompt,
                    conf_thresh=SAM3_CONFIDENCE
                )
                sam_dets.extend(prompt_dets)

        merged_mask_dets = merge_yolo_sam(yolo_seg_dets, sam_dets, image_bgr.shape)
        mask_rows, mask_viz_dets = convert_mask_detections(merged_mask_dets, image_bgr.shape)

        aux_dets = []
        if yolo11_model is not None:
            aux_dets = get_yolo_aux_obbs(yolo11_model, image_bgr)
        aux_rows, aux_viz_dets = convert_direct_detections(aux_dets, image_bgr.shape)

        all_rows = mask_rows + aux_rows
        all_viz_dets = mask_viz_dets + aux_viz_dets

        txt_path, viz_path = get_relative_output_paths(
            img_path, DATASET_DIR, OUTPUT_LABEL_DIR, OUTPUT_VIZ_DIR
        )

        added_rows = save_yolo_txt(txt_path, all_rows)
        if LABEL_WRITE_MODE != "overwrite":
            print(f"[INFO] Added {added_rows} new label row(s) to {txt_path.name}")

        if SAVE_VIZ:
            ensure_dir(viz_path.parent)
            viz = draw_overlay(image_bgr, all_viz_dets)
            cv2.imwrite(str(viz_path), viz)

    print("[DONE] Finished.")

if __name__ == "__main__":
    main()