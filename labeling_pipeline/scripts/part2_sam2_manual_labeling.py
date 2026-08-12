#!/usr/bin/env python3

import argparse
import sys
import yaml
import cv2
import numpy as np

from pathlib import Path

# =========================================================
# PATHS / CONFIG LOADING
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = PIPELINE_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "part2_parameters.yaml"

CONFIG_PATH = DEFAULT_CONFIG_PATH
CFG = {}

IMAGE_DIR = None
LABEL_DIR = None
VIS_DIR = None

SAM2_ROOT = None
SAM2_CHECKPOINT = None
SAM2_MODEL_CFG = None

DEVICE = "cuda"
MIN_MASK_AREA = 20
SAVE_VIZ = True
USE_OBB = True

CLASS_NAMES = {}

WINDOW_NAME = "annotator"
RESULT_WINDOW = "preview"

# =========================================================
# CONFIG LOADING
# =========================================================
def resolve_config_path(config_path_arg) -> Path:
    """Resolve relative or absolute config paths."""
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

    return (Path.cwd() / path).resolve()


def resolve_project_path(path_value) -> Path:
    """Resolve YAML paths relative to the project/repository."""
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

def load_config(config_path: Path):
    global CONFIG_PATH, CFG
    global IMAGE_DIR, LABEL_DIR, VIS_DIR
    global SAM2_ROOT, SAM2_CHECKPOINT, SAM2_MODEL_CFG
    global DEVICE, MIN_MASK_AREA, SAVE_VIZ, USE_OBB
    global CLASS_NAMES, WINDOW_NAME, RESULT_WINDOW

    CONFIG_PATH = resolve_config_path(config_path)
    CFG = load_yaml(CONFIG_PATH)

    IMAGE_DIR = resolve_project_path(CFG["paths"]["image_dir"])
    LABEL_DIR = resolve_project_path(CFG["paths"]["label_dir"])
    VIS_DIR = resolve_project_path(CFG["paths"].get("vis_dir", "visualizations_final"))

    SAM2_ROOT = resolve_project_path(CFG["models"]["sam2_root"])
    SAM2_CHECKPOINT = resolve_project_path(CFG["models"]["sam2_checkpoint"])
    SAM2_MODEL_CFG = CFG["models"]["sam2_model_cfg"]

    runtime_cfg = CFG.get("runtime", {})
    DEVICE = runtime_cfg.get("device", "cuda")
    MIN_MASK_AREA = int(runtime_cfg.get("min_mask_area", 20))
    SAVE_VIZ = bool(runtime_cfg.get("save_viz", True))

    output_cfg = CFG.get("output", {})
    USE_OBB = bool(output_cfg.get("use_obb", True))

    CLASS_NAMES = {
        int(class_id): str(class_name)
        for class_id, class_name in CFG["classes"]["id_to_name"].items()
    }

    ui_cfg = CFG.get("ui", {})
    WINDOW_NAME = ui_cfg.get("window_name", "annotator")
    RESULT_WINDOW = ui_cfg.get("result_window", "preview")

    sys.path.insert(0, str(SAM2_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Part 2 manual correction UI using SAM2 with selectable "
            "OBB or normal BB labels."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file."
    )
    return parser.parse_args()

# =========================================================
# GLOBAL STATE
# =========================================================
points = []
point_labels = []
saved_objects = []
current_class_id = 0
current_image_index = 0
selected_object_index = 0

manual_obb_mode = False
manual_obb_points = []

image_paths = []
current_image_bgr = None
current_image_rgb = None

sam2_predictor = None

orientation_line_mode = False
orientation_line_points = []

# =========================================================
# KEY CODES
# =========================================================
# OpenCV arrow key codes can differ depending on backend/platform.
KEY_LEFT_CODES = {81, 2424832, 65361}
KEY_UP_CODES = {82, 2490368, 65362}
KEY_RIGHT_CODES = {83, 2555904, 65363}
KEY_DOWN_CODES = {84, 2621440, 65364}

# =========================================================
# GEOMETRY UTILS
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def list_images_recursive(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )

def order_box_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]

    s = pts.sum(axis=1)
    start_idx = np.argmin(s)
    pts = np.roll(pts, -start_idx, axis=0)
    return pts

def quad_to_rect(box4):
    return cv2.minAreaRect(np.array(box4, dtype=np.float32))

def axis_aligned_box_from_xyxy(x1, y1, x2, y2) -> np.ndarray:
    x_min, x_max = sorted((float(x1), float(x2)))
    y_min, y_max = sorted((float(y1), float(y2)))

    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )

def axis_aligned_box_from_quad(box4: np.ndarray) -> np.ndarray:
    box4 = np.asarray(box4, dtype=np.float32)
    return axis_aligned_box_from_xyxy(
        np.min(box4[:, 0]),
        np.min(box4[:, 1]),
        np.max(box4[:, 0]),
        np.max(box4[:, 1]),
    )

def build_box_metadata(box: np.ndarray, area=None, rectangularity=None, angle_deg=None):
    box = order_box_points_clockwise(np.asarray(box, dtype=np.float32))
    rect = quad_to_rect(box)
    (_, _), (width, height), _ = rect

    if width <= 0 or height <= 0:
        return None

    if area is None:
        area = float(cv2.contourArea(box.astype(np.float32)))

    long_side = max(width, height)
    short_side = min(width, height)

    result = {
        "rect": rect,
        "box": box,
        "area": float(area),
        "aspect_ratio": long_side / (short_side + 1e-6),
        "rectangularity": rectangularity,
    }

    if angle_deg is not None:
        result["angle_deg"] = float(angle_deg)

    return result

def mask_to_obb(mask: np.ndarray, min_area: float = 20):
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < min_area:
        return None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    (_, _), (width, height), _ = rect

    if width <= 0 or height <= 0:
        return None

    rectangularity = area / (width * height + 1e-6)
    return build_box_metadata(box, area=area, rectangularity=rectangularity)

def mask_to_normal_bb(mask: np.ndarray, min_area: float = 20, padding_px: float = 0.0):
    mask_u8 = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 0)

    if len(xs) < min_area:
        return None

    x_min = max(0.0, float(xs.min()) - padding_px)
    x_max = float(xs.max()) + padding_px
    y_min = max(0.0, float(ys.min()) - padding_px)
    y_max = float(ys.max()) + padding_px

    box = axis_aligned_box_from_xyxy(x_min, y_min, x_max, y_max)
    box_area = max((x_max - x_min) * (y_max - y_min), 1e-6)
    mask_area = float(np.count_nonzero(mask_u8))

    return build_box_metadata(
        box,
        area=mask_area,
        rectangularity=mask_area / box_area,
        angle_deg=0.0,
    )

def mask_to_selected_box(mask: np.ndarray, min_area: float = 20):
    if USE_OBB:
        return mask_to_obb(mask, min_area=min_area)
    return mask_to_normal_bb(mask, min_area=min_area)

def get_obb_angle_deg(obb):
    box = obb["box"].astype(np.float32)
    edge = box[1] - box[0]
    return float(np.rad2deg(np.arctan2(edge[1], edge[0])))


def get_display_box(obj) -> np.ndarray:
    box = obj["obb"]["box"].astype(np.float32)
    if USE_OBB:
        return box
    return axis_aligned_box_from_quad(box)


def set_selected_obb_angle_from_two_points(p1, p2):
    global saved_objects, selected_object_index

    if not USE_OBB:
        print("Orientation editing is disabled in normal-BB mode.")
        return

    if not saved_objects:
        print("No object selected")
        return

    obj = saved_objects[selected_object_index]
    mask = obj.get("mask", None)

    if mask is None:
        print("Selected object has no mask; cannot refit OBB from segmentation.")
        return

    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    delta = p2 - p1
    if np.linalg.norm(delta) < 5.0:
        print("Orientation line too short")
        return

    angle_deg = float(np.rad2deg(np.arctan2(delta[1], delta[0])))
    new_obb = obb_from_mask_at_angle(
        mask,
        angle_deg,
        min_area=MIN_MASK_AREA,
        padding_px=3.0
    )

    if new_obb is None:
        print("Could not create OBB from selected orientation")
        return

    obj["obb"] = new_obb

    print(f"Set selected OBB orientation from two points: {angle_deg:.1f} deg")

def obb_from_mask_at_angle(mask: np.ndarray, angle_deg: float, min_area: float = 20, padding_px: float = 2.0):
    """
    Build an OBB that fully contains the mask, while forcing the box orientation.
    This recomputes the box extents from the mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < min_area:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)

    theta = np.deg2rad(angle_deg)

    axis_u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    axis_v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    proj_u = pts @ axis_u
    proj_v = pts @ axis_v

    u_min = float(proj_u.min()) - padding_px
    u_max = float(proj_u.max()) + padding_px
    v_min = float(proj_v.min()) - padding_px
    v_max = float(proj_v.max()) + padding_px

    corners = np.array([
        u_min * axis_u + v_min * axis_v,
        u_max * axis_u + v_min * axis_v,
        u_max * axis_u + v_max * axis_v,
        u_min * axis_u + v_max * axis_v,
    ], dtype=np.float32)

    box = order_box_points_clockwise(corners)

    rect = cv2.minAreaRect(box)
    (_, _), (width, height), _ = rect

    if width <= 0 or height <= 0:
        return None

    area = float(np.count_nonzero(mask_u8))
    rectangularity = area / (width * height + 1e-6)

    return build_box_metadata(
        box,
        area=area,
        rectangularity=rectangularity,
        angle_deg=angle_deg,
    )


def rotate_selected_obb(angle_delta_deg):
    global saved_objects, selected_object_index

    if not USE_OBB:
        print("Rotation is disabled in normal-BB mode.")
        return

    if not saved_objects:
        print("No object selected to rotate")
        return

    selected_object_index = min(selected_object_index, len(saved_objects) - 1)
    obj = saved_objects[selected_object_index]

    current_angle = obj["obb"].get("angle_deg", None)
    if current_angle is None:
        current_angle = get_obb_angle_deg(obj["obb"])

    new_angle = current_angle + angle_delta_deg
    mask = obj.get("mask", None)

    if mask is not None:
        new_obb = obb_from_mask_at_angle(
            mask,
            new_angle,
            min_area=MIN_MASK_AREA,
            padding_px=3.0
        )

        if new_obb is None:
            print("Could not refit OBB from mask at new angle")
            return

        obj["obb"] = new_obb
        print(
            f"Rotated and refit object {selected_object_index}: "
            f"{current_angle:.1f} -> {new_angle:.1f} deg"
        )

    else:
        print(
            "Selected object has no mask, so rotation cannot guarantee "
            "full segmentation coverage."
        )

def create_obb_from_4_points(points4):
    pts = np.array(points4, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    return build_box_metadata(box, rectangularity=None)

def create_normal_bb_from_2_points(points2):
    pts = np.array(points2, dtype=np.float32)
    if pts.shape != (2, 2):
        return None

    box = axis_aligned_box_from_xyxy(
        pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]
    )
    return build_box_metadata(box, rectangularity=None, angle_deg=0.0)

def point_inside_box(x, y, box):
    contour = np.array(box, dtype=np.float32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def find_object_at_point(x, y, objects):
    hits = []
    for idx, obj in enumerate(objects):
        box = get_display_box(obj)
        if point_inside_box(x, y, box):
            area = abs(float(cv2.contourArea(box.astype(np.float32))))
            hits.append((area, idx))

    if not hits:
        return None

    hits.sort(key=lambda item: item[0])
    return hits[0][1]

# =========================================================
# FILE IO
# =========================================================
def get_relative_paths(img_path: Path):
    rel_path = img_path.relative_to(IMAGE_DIR)
    label_path = LABEL_DIR / rel_path.with_suffix(".txt")
    vis_path = VIS_DIR / rel_path.with_name(f"{rel_path.stem}_vis.jpg")
    return label_path, vis_path

def get_label_path(img_path: Path) -> Path:
    label_path, _ = get_relative_paths(img_path)
    return label_path

def get_vis_path(img_path: Path) -> Path:
    _, vis_path = get_relative_paths(img_path)
    return vis_path

def save_yolo_labels(txt_path: Path, image_shape, objects):
    """Save either OBB or normal YOLO detection labels."""
    ensure_dir(txt_path.parent)
    height, width = image_shape[:2]

    with open(txt_path, "w", encoding="utf-8") as f:
        for obj in objects:
            class_id = int(obj["class_id"])
            box = obj["obb"]["box"].astype(np.float32).copy()

            if USE_OBB:
                box = order_box_points_clockwise(box)
                box[:, 0] = np.clip(box[:, 0] / width, 0.0, 1.0)
                box[:, 1] = np.clip(box[:, 1] / height, 0.0, 1.0)

                values = [str(class_id)] + [
                    f"{value:.6f}" for point in box for value in point
                ]
            else:
                x_min = float(np.clip(np.min(box[:, 0]), 0, width))
                x_max = float(np.clip(np.max(box[:, 0]), 0, width))
                y_min = float(np.clip(np.min(box[:, 1]), 0, height))
                y_max = float(np.clip(np.max(box[:, 1]), 0, height))

                x_center = ((x_min + x_max) / 2.0) / width
                y_center = ((y_min + y_max) / 2.0) / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                values = [
                    str(class_id),
                    f"{x_center:.6f}",
                    f"{y_center:.6f}",
                    f"{box_width:.6f}",
                    f"{box_height:.6f}",
                ]

            f.write(" ".join(values) + "\n")

def load_yolo_labels(txt_path: Path, image_shape):
    """
    Load either OBB rows (9 fields) or normal-BB rows (5 fields).

    Internally, both formats are represented as four corner points. Saving
    converts every object to the format selected by output.use_obb.
    """
    height, width = image_shape[:2]
    objects = []

    if not txt_path.exists():
        return objects

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    detected_formats = set()

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        try:
            class_id = int(float(parts[0]))
        except (ValueError, IndexError):
            print(f"[WARN] Invalid class ID at {txt_path}:{line_number}; skipped")
            continue

        try:
            if len(parts) == 9:
                detected_formats.add("OBB")
                coords = np.array(
                    list(map(float, parts[1:])), dtype=np.float32
                ).reshape(4, 2)
                coords[:, 0] *= width
                coords[:, 1] *= height
                coords = order_box_points_clockwise(coords)

            elif len(parts) == 5:
                detected_formats.add("normal BB")
                x_center, y_center, box_width, box_height = map(float, parts[1:])

                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                coords = axis_aligned_box_from_xyxy(
                    x_center - box_width / 2.0,
                    y_center - box_height / 2.0,
                    x_center + box_width / 2.0,
                    y_center + box_height / 2.0,
                )

            else:
                print(
                    f"[WARN] Unsupported row with {len(parts)} fields at "
                    f"{txt_path}:{line_number}; skipped"
                )
                continue

        except ValueError:
            print(f"[WARN] Invalid coordinates at {txt_path}:{line_number}; skipped")
            continue

        metadata = build_box_metadata(coords, rectangularity=None)
        if metadata is None:
            print(f"[WARN] Degenerate box at {txt_path}:{line_number}; skipped")
            continue

        objects.append(
            {
                "class_id": class_id,
                "source": "file",
                "confidence": 1.0,
                "obb": metadata,
                "mask": None,
            }
        )

    selected_format = "OBB" if USE_OBB else "normal BB"
    for detected_format in sorted(detected_formats):
        if detected_format != selected_format:
            print(
                f"[INFO] Loaded {detected_format} labels. They will be converted "
                f"to {selected_format} when saved."
            )

    return objects

# =========================================================
# DRAWING
# =========================================================
def draw_cross(img, x, y, color):
    cv2.drawMarker(
        img,
        (int(x), int(y)),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=8,
        thickness=1
    )

def object_color(source: str):
    if source == "sam2":
        return (0, 255, 0)
    if source in {"manual_obb", "manual_bb"}:
        return (255, 0, 255)
    if source == "file":
        return (200, 200, 200)
    return (200, 200, 200)

def draw_ui(image, current_points, current_point_labels, objects, active_class_id, img_name, img_idx, total_imgs, selected_idx):
    vis = image.copy()

    for index, obj in enumerate(objects):
        class_id = obj["class_id"]
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        source = obj.get("source", "unknown")
        confidence = obj.get("confidence", None)

        box = get_display_box(obj).astype(np.int32)
        color = object_color(source)
        thickness = 3 if index == selected_idx else 2
        cv2.polylines(vis, [box], True, color, thickness)

        center_x, center_y = np.mean(box, axis=0).astype(int)
        label = f"{index}:{class_id}-{class_name}"
        if confidence is not None:
            label += f" {confidence:.2f}"
        label += f" [{source}]"

        cv2.putText(
            vis, label,(center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )

    for (x, y), point_label in zip(current_points, current_point_labels):
        color = (0, 255, 0) if point_label == 1 else (0, 0, 255)
        draw_cross(vis, x, y, color)

    global manual_obb_mode, manual_obb_points
    if manual_obb_mode:
        required_points = 4 if USE_OBB else 2
        mode_name = "MANUAL OBB" if USE_OBB else "MANUAL BB"

        for index, (point_x, point_y) in enumerate(manual_obb_points):
            cv2.circle(vis, (int(point_x), int(point_y)), 4, (255, 0, 255), -1)
            cv2.putText(
                vis, str(index + 1), (int(point_x) + 4, int(point_y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA,
            )

        if len(manual_obb_points) >= 2:
            pts = np.array(manual_obb_points, dtype=np.int32)
            if USE_OBB:
                cv2.polylines(vis, [pts], False, (255, 0, 255), 1)
            else:
                preview_box = axis_aligned_box_from_xyxy(
                    pts[0, 0], pts[0, 1], pts[-1, 0], pts[-1, 1]
                ).astype(np.int32)
                cv2.polylines(vis, [preview_box], True, (255, 0, 255), 1)

        cv2.putText(
            vis,
            f"{mode_name} MODE: click {required_points} point(s) ({len(manual_obb_points)}/{required_points})",
            (10, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    global orientation_line_mode, orientation_line_points
    if orientation_line_mode:
        for index, (point_x, point_y) in enumerate(orientation_line_points):
            cv2.circle(vis, (int(point_x), int(point_y)), 5, (0, 255, 255), -1)
            cv2.putText(
                vis, str(index + 1),
                (int(point_x) + 5, int(point_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if len(orientation_line_points) == 2:
            pts = np.array(orientation_line_points, dtype=np.int32)
            cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2)

        cv2.putText(
            vis,
            f"ORIENTATION MODE: click 2 points ({len(orientation_line_points)}/2)",
            (10, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    active_name = CLASS_NAMES.get(active_class_id, str(active_class_id))
    box_format = "OBB" if USE_OBB else "normal BB"

    header1 = f"Image [{img_idx + 1}/{total_imgs}] {img_name} | Format: {box_format}"
    header2 = f"Active class: {active_class_id} ({active_name}) | Selected object: {selected_idx if objects else 'none'}"
    
    if USE_OBB:
        header3 = "Keys: 0-9 class | Space add SAM2 obj | arrows rotate/refit OBB | o align OBB | m manual OBB | x delete"
    else:
        header3 = "Keys: 0-9 class | Space add SAM2 obj | m manual BB (2 corners) | x delete"

    header4 = "s save | a/d prev/next | w save+next | u undo | c clear clicks | Esc cancel manual | q quit"
    header5 = "Mouse: click box=select | Left empty=positive | Right empty=negative | Middle box=delete"

    cv2.putText(vis, header1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, header2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, header3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, header4, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, header5, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return vis

def build_result_preview(image, mask, box_data, class_id, source="sam2"):
    result = image.copy()

    overlay = result.copy()
    overlay[mask > 0] = [0, 200, 255]
    result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

    if box_data is not None:
        box = box_data["box"].astype(np.int32)
        cv2.polylines(result, [box], True, object_color(source), 2)

        (center_x, center_y), (width, height), angle = box_data["rect"]
        class_name = CLASS_NAMES.get(class_id, str(class_id))

        if USE_OBB:
            text = (
                f"{class_id}:{class_name} [{source}] | "
                f"w={width:.1f} h={height:.1f} a={angle:.1f}"
            )
        else:
            text = (
                f"{class_id}:{class_name} [{source}] | "
                f"w={width:.1f} h={height:.1f}"
            )

        cv2.putText(
            result,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return result

# =========================================================
# MOUSE
# =========================================================
def mouse_callback(event, x, y, flags, param):
    global points, point_labels, saved_objects, selected_object_index
    global manual_obb_mode, manual_obb_points, current_class_id
    global orientation_line_mode, orientation_line_points

    if orientation_line_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            orientation_line_points.append([x, y])
            print(f"Orientation point {len(orientation_line_points)}: ({x}, {y})")

            if len(orientation_line_points) == 2:
                set_selected_obb_angle_from_two_points(
                    orientation_line_points[0],
                    orientation_line_points[1]
                )
                orientation_line_points = []
                orientation_line_mode = False
        return

    if manual_obb_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            manual_obb_points.append([x, y])
            required_points = 4 if USE_OBB else 2
            box_name = "OBB" if USE_OBB else "BB"
            print(f"Manual {box_name} point {len(manual_obb_points)}: ({x}, {y})")

            if len(manual_obb_points) == required_points:
                if USE_OBB:
                    box_data = create_obb_from_4_points(manual_obb_points)
                    source = "manual_obb"
                else:
                    box_data = create_normal_bb_from_2_points(manual_obb_points)
                    source = "manual_bb"

                if box_data is not None:
                    saved_objects.append(
                        {
                            "class_id": current_class_id,
                            "source": source,
                            "confidence": 1.0,
                            "obb": box_data,
                            "mask": None,
                        }
                    )
                    selected_object_index = len(saved_objects) - 1
                    print(
                        f"Added manual {box_name}: class {current_class_id} "
                        f"({CLASS_NAMES[current_class_id]})"
                    )

                manual_obb_points = []
                manual_obb_mode = False
        return

    clicked_idx = find_object_at_point(x, y, saved_objects)

    if event == cv2.EVENT_LBUTTONDOWN:
        if clicked_idx is not None:
            selected_object_index = clicked_idx
            print(f"Selected object: {selected_object_index}")
        else:
            points.append([x, y])
            point_labels.append(1)
            print(f"Positive: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_idx is not None:
            selected_object_index = clicked_idx
            print(f"Selected object: {selected_object_index}")
        else:
            points.append([x, y])
            point_labels.append(0)
            print(f"Negative: ({x}, {y})")

    elif event == cv2.EVENT_MBUTTONDOWN:
        if clicked_idx is not None:
            removed = saved_objects.pop(clicked_idx)
            if saved_objects:
                selected_object_index = min(clicked_idx, len(saved_objects) - 1)
            else:
                selected_object_index = 0
            print(f"Deleted object class {removed['class_id']} ({CLASS_NAMES.get(removed['class_id'], removed['class_id'])})")

# =========================================================
# IMAGE LOADING
# =========================================================
def load_image_at_index(idx):
    global current_image_index, current_image_bgr, current_image_rgb
    global saved_objects, points, point_labels, selected_object_index
    global manual_obb_mode, manual_obb_points
    global orientation_line_mode, orientation_line_points

    current_image_index = idx
    img_path = image_paths[current_image_index]

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    current_image_bgr = image
    current_image_rgb = image_rgb

    sam2_predictor.set_image(current_image_rgb)

    points = []
    point_labels = []
    selected_object_index = 0
    manual_obb_mode = False
    manual_obb_points = []
    orientation_line_mode = False
    orientation_line_points = []

    label_path = get_label_path(img_path)
    if label_path.exists():
        saved_objects = load_yolo_labels(label_path, current_image_bgr.shape)
        print(f"\nLoaded existing labels from file: {label_path}")
    else:
        saved_objects = []
        print("\nNo existing labels found. Starting empty.")

    rel_img = img_path.relative_to(IMAGE_DIR)
    print(f"Loaded image {current_image_index + 1}/{len(image_paths)}: {rel_img}")

def save_current_image():
    img_path = image_paths[current_image_index]
    txt_path = get_label_path(img_path)

    save_yolo_labels(txt_path, current_image_bgr.shape, saved_objects)

    selected_format = "OBB" if USE_OBB else "normal BB"
    print(f"Saved {selected_format} labels: {txt_path}")

    if SAVE_VIZ:
        vis_path = get_vis_path(img_path)
        vis = draw_ui(
            current_image_bgr, [], [], saved_objects, current_class_id,
            str(img_path.relative_to(IMAGE_DIR)), current_image_index, len(image_paths), selected_object_index
        )
        ensure_dir(vis_path.parent)
        cv2.imwrite(str(vis_path), vis)
        print(f"Saved visualization: {vis_path}")

    print(f"Objects saved: {len(saved_objects)}")

# =========================================================
# MAIN
# =========================================================
def main():
    global sam2_predictor
    global current_class_id, selected_object_index, saved_objects
    global points, point_labels, image_paths
    global manual_obb_mode, manual_obb_points
    global orientation_line_mode, orientation_line_points

    args = parse_args()
    load_config(args.config)

    from sam2.build_sam import build_sam2  # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_VIZ:
        VIS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images_recursive(IMAGE_DIR)
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    selected_format = "OBB" if USE_OBB else "normal BB"

    print("[INFO] Configuration loaded from:", CONFIG_PATH)
    print("[INFO] Image directory:", IMAGE_DIR)
    print("[INFO] Label directory:", LABEL_DIR)
    print("[INFO] Bounding-box format:", selected_format)
    print("[INFO] Save visualizations:", SAVE_VIZ)
    if SAVE_VIZ:
        print("[INFO] Visualization directory:", VIS_DIR)
    print("[INFO] Number of images found:", len(image_paths))
    print("[INFO] SAM2 root:", SAM2_ROOT)
    print("[INFO] SAM2 cfg:", SAM2_MODEL_CFG)
    print("[INFO] SAM2 checkpoint:", SAM2_CHECKPOINT)

    print("Loading SAM2...")
    sam2_model = build_sam2(SAM2_MODEL_CFG, str(SAM2_CHECKPOINT), device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("\nClasses:")
    for key, value in CLASS_NAMES.items():
        print(f"  {key} = {value}")

    print("\nControls:")
    print("  Left click   = select object if on box, otherwise positive SAM2 point")
    print("  Right click  = select object if on box, otherwise negative SAM2 point")
    print("  Middle click = delete object if on box")
    print("  0..9         = select class")
    print("  Space        = run SAM2 and add object")

    if USE_OBB:
        print("  m            = manual OBB mode (click 4 corners)")
        print("  o            = orientation mode: click 2 points to align selected OBB")
        print("  Left/Right   = rotate selected OBB -/+ 1 degree")
        print("  Down/Up      = rotate selected OBB -/+ 5 degrees")
    else:
        print("  m            = manual normal-BB mode (click 2 opposite corners)")
        print("  Rotation/orientation controls are disabled in normal-BB mode")

    print("  Backspace    = remove last manual-box point")
    print("  Esc          = cancel manual-box/orientation mode")
    print("  x            = delete selected object")
    print("  c            = clear current SAM2 clicks")
    print("  u            = undo last object")
    print("  s            = save current image")
    print("  a / d        = previous / next image")
    print("  w            = save and next")
    print("  q            = quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    load_image_at_index(0)

    while True:
        rel_name = str(image_paths[current_image_index].relative_to(IMAGE_DIR))
        vis = draw_ui(
            current_image_bgr,
            points,
            point_labels,
            saved_objects,
            current_class_id,
            rel_name,
            current_image_index,
            len(image_paths),
            selected_object_index
        )
        cv2.imshow(WINDOW_NAME, vis)

        key_raw = cv2.waitKeyEx(10)
        if key_raw < 0:
            continue

        key = key_raw & 0xFF

        if key in [ord(str(i)) for i in range(10)]:
            selected = int(chr(key))
            if selected in CLASS_NAMES:
                current_class_id = selected
                print(f"Selected class: {current_class_id} ({CLASS_NAMES[current_class_id]})")

        elif key == ord("x"):
            if saved_objects:
                removed = saved_objects.pop(selected_object_index)
                selected_object_index = min(selected_object_index, max(0, len(saved_objects) - 1))
                print(f"Deleted object class {removed['class_id']} ({CLASS_NAMES.get(removed['class_id'], removed['class_id'])})")
            else:
                print("No objects to delete")

        elif key_raw in KEY_LEFT_CODES:
            rotate_selected_obb(-1.0)

        elif key_raw in KEY_RIGHT_CODES:
            rotate_selected_obb(1.0)

        elif key_raw in KEY_DOWN_CODES:
            rotate_selected_obb(-5.0)

        elif key_raw in KEY_UP_CODES:
            rotate_selected_obb(5.0)

        elif key == ord("o"):
            if not USE_OBB:
                print("Orientation mode is disabled in normal-BB mode.")
                continue

            orientation_line_mode = True
            orientation_line_points = []
            points = []
            point_labels = []
            manual_obb_mode = False
            print("Orientation line mode: click 2 points along desired OBB direction")

        elif key == ord(" "):
            if manual_obb_mode:
                print("Finish or cancel manual box mode first.")
                continue

            if len(points) == 0:
                print("No points")
                continue

            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(point_labels, dtype=np.int32)

            masks, scores, _ = sam2_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]

            box_data = mask_to_selected_box(best_mask, min_area=MIN_MASK_AREA)
            if box_data is None:
                print("No valid box found from SAM2 mask")
                continue

            if USE_OBB:
                box_data["angle_deg"] = get_obb_angle_deg(box_data)

            saved_objects.append({
                "class_id": current_class_id,
                "source": "sam2",
                "confidence": float(scores[best_idx]),
                "obb": box_data,
                "mask": best_mask.copy(),
            })
            selected_object_index = len(saved_objects) - 1

            preview = build_result_preview(current_image_bgr, best_mask, box_data, current_class_id, source="sam2")
            cv2.imshow(RESULT_WINDOW, preview)

            box_name = "OBB" if USE_OBB else "normal BB"
            print(f"Added SAM2 {box_name}: class {current_class_id} ({CLASS_NAMES[current_class_id]})")
            points = []
            point_labels = []

        elif key == ord("m"):
            manual_obb_mode = True
            manual_obb_points = []
            points = []
            point_labels = []
            orientation_line_mode = False
            orientation_line_points = []

            box_name = "OBB" if USE_OBB else "normal BB"
            required_points = 4 if USE_OBB else 2
            print(f"Manual {box_name} mode enabled for class {current_class_id} ({CLASS_NAMES[current_class_id]}). Click {required_points} point(s).")

        elif key in {8, 127} or key_raw in {8, 127, 65288}:
            if manual_obb_mode and manual_obb_points:
                manual_obb_points.pop()
                print("Removed last manual-box point")

        elif key == 27:
            if manual_obb_mode:
                manual_obb_mode = False
                manual_obb_points = []
                print("Cancelled manual box mode")

            if orientation_line_mode:
                orientation_line_mode = False
                orientation_line_points = []
                print("Cancelled orientation mode")

        elif key == ord("c"):
            points = []
            point_labels = []
            print("Cleared current SAM2 clicks")

        elif key == ord("u"):
            if saved_objects:
                removed = saved_objects.pop()
                selected_object_index = min(selected_object_index, max(0, len(saved_objects) - 1))
                print(f"Removed last object: class {removed['class_id']} ({CLASS_NAMES.get(removed['class_id'], removed['class_id'])})")
            else:
                print("No objects to undo")

        elif key == ord("s"):
            save_current_image()

        elif key == ord("a"):
            save_current_image()
            prev_idx = max(0, current_image_index - 1)
            load_image_at_index(prev_idx)

        elif key == ord("d"):
            save_current_image()
            next_idx = min(len(image_paths) - 1, current_image_index + 1)
            load_image_at_index(next_idx)

        elif key == ord("w"):
            save_current_image()
            next_idx = min(len(image_paths) - 1, current_image_index + 1)
            load_image_at_index(next_idx)

        elif key == ord("q"):
            print("Exiting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()