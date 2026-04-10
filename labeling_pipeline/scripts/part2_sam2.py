#!/usr/bin/env python3

import sys
import yaml
import cv2
import numpy as np

from pathlib import Path


# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "part2_parameters.yaml"


# =========================================================
# CONFIG LOADING
# =========================================================
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml(CONFIG_PATH)

IMAGE_DIR = PROJECT_ROOT / CFG["paths"]["image_dir"]
LABEL_DIR = PROJECT_ROOT / CFG["paths"]["label_dir"]
VIS_DIR = PROJECT_ROOT / CFG["paths"]["vis_dir"]

SAM2_ROOT = PROJECT_ROOT / CFG["models"]["sam2_root"]
SAM2_CHECKPOINT = PROJECT_ROOT / CFG["models"]["sam2_checkpoint"]
SAM2_MODEL_CFG = PROJECT_ROOT / CFG["models"]["sam2_model_cfg"]

DEVICE = CFG["runtime"]["device"]

MIN_MASK_AREA = int(CFG["runtime"]["min_mask_area"])

CLASS_NAMES = {int(k): v for k, v in CFG["classes"]["id_to_name"].items()}

WINDOW_NAME = CFG["ui"]["window_name"]
RESULT_WINDOW = CFG["ui"]["result_window"]

ENABLE_AMG_FALLBACK = bool(CFG["sam2"]["enable_amg_fallback"])
TRACKBAR_NAME = CFG["sam2"]["trackbar_name"]
TRACKBAR_MIN_AREA_INIT = int(CFG["sam2"]["trackbar_min_area_init"])
TRACKBAR_MIN_AREA_MAX = int(CFG["sam2"]["trackbar_min_area_max"])

FALLBACK_MAX_MASKS = int(CFG["sam2"]["fallback_max_masks"])
FALLBACK_MIN_ASPECT_RATIO = float(CFG["sam2"]["fallback_min_aspect_ratio"])
FALLBACK_MIN_RECTANGULARITY = float(CFG["sam2"]["fallback_min_rectangularity"])
FALLBACK_MAX_RECTANGULARITY = float(CFG["sam2"]["fallback_max_rectangularity"])

AMG_POINTS_PER_SIDE = int(CFG["sam2"]["amg_points_per_side"])
AMG_PRED_IOU_THRESH = float(CFG["sam2"]["amg_pred_iou_thresh"])
AMG_STABILITY_SCORE_THRESH = float(CFG["sam2"]["amg_stability_score_thresh"])
AMG_CROP_N_LAYERS = int(CFG["sam2"]["amg_crop_n_layers"])
AMG_CROP_N_POINTS_DOWNSCALE_FACTOR = int(CFG["sam2"]["amg_crop_n_points_downscale_factor"])
AMG_BOX_NMS_THRESH = float(CFG["sam2"]["amg_box_nms_thresh"])
AMG_MIN_MASK_REGION_AREA = int(CFG["sam2"]["amg_min_mask_region_area"])


# =========================================================
# MAKE SAM2 IMPORTABLE
# =========================================================
sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402


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
sam2_mask_generator = None


# =========================================================
# GEOMETRY UTILS
# =========================================================
def list_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
        files.extend(folder.glob(ext.upper()))
    return sorted(files)


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


def mask_to_obb(mask: np.ndarray, min_area: float = 20):
    mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_box_points_clockwise(box)

    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return None

    long_side = max(w, h)
    short_side = min(w, h)
    aspect_ratio = long_side / (short_side + 1e-6)
    rectangularity = area / (w * h + 1e-6)

    return {
        "rect": rect,
        "box": box,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
    }


def quad_to_rect(box4):
    return cv2.minAreaRect(np.array(box4, dtype=np.float32))


def create_obb_from_4_points(points4):
    pts = np.array(points4, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = order_box_points_clockwise(box)

    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return None

    area = cv2.contourArea(box.astype(np.float32))
    long_side = max(w, h)
    short_side = min(w, h)
    aspect_ratio = long_side / (short_side + 1e-6)

    return {
        "rect": rect,
        "box": box,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "rectangularity": None,
    }


def point_inside_obb(x, y, box):
    contour = np.array(box, dtype=np.float32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def find_object_at_point(x, y, objects):
    hits = []
    for idx, obj in enumerate(objects):
        box = obj["obb"]["box"]
        if point_inside_obb(x, y, box):
            area = obj["obb"].get("area", 1e9)
            hits.append((area, idx))

    if not hits:
        return None

    hits.sort(key=lambda t: t[0])
    return hits[0][1]


# =========================================================
# FILE IO
# =========================================================
def get_label_path(img_path: Path) -> Path:
    return LABEL_DIR / f"{img_path.stem}.txt"


def get_vis_path(img_path: Path) -> Path:
    return VIS_DIR / f"{img_path.stem}_vis.jpg"


def save_yolo_obb(txt_path: Path, image_shape, objects):
    h, w = image_shape[:2]

    with open(txt_path, "w", encoding="utf-8") as f:
        for obj in objects:
            class_id = obj["class_id"]
            box = obj["obb"]["box"].astype(np.float32).copy()

            box[:, 0] /= w
            box[:, 1] /= h

            vals = [str(class_id)] + [f"{v:.6f}" for pt in box for v in pt]
            f.write(" ".join(vals) + "\n")


def load_yolo_obb(txt_path: Path, image_shape):
    h, w = image_shape[:2]
    objects = []

    if not txt_path.exists():
        return objects

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 9:
            continue

        class_id = int(float(parts[0]))
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = order_box_points_clockwise(coords)

        objects.append({
            "class_id": class_id,
            "source": "file",
            "confidence": 1.0,
            "obb": {
                "rect": quad_to_rect(coords),
                "box": coords,
                "area": cv2.contourArea(coords.astype(np.float32)),
                "aspect_ratio": None,
                "rectangularity": None,
            },
            "mask": None,
        })

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
    if source == "sam2_amg":
        return (255, 0, 255)
    if source == "manual_obb":
        return (255, 0, 255)
    if source == "file":
        return (200, 200, 200)
    return (200, 200, 200)


def draw_ui(image, current_points, current_point_labels, objects, active_class_id,
            img_name, img_idx, total_imgs, selected_idx):
    vis = image.copy()

    for i, obj in enumerate(objects):
        obb = obj["obb"]
        class_id = obj["class_id"]
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        source = obj.get("source", "unknown")
        conf = obj.get("confidence", None)

        box = obb["box"].astype(np.int32)
        color = object_color(source)
        thickness = 3 if i == selected_idx else 2
        cv2.polylines(vis, [box], True, color, thickness)

        cx, cy = np.mean(box, axis=0).astype(int)
        label = f"{i}:{class_id}-{class_name}"
        if conf is not None:
            label += f" {conf:.2f}"
        label += f" [{source}]"

        cv2.putText(
            vis, label, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

    for (x, y), lab in zip(current_points, current_point_labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        draw_cross(vis, x, y, color)

    global manual_obb_mode, manual_obb_points
    if manual_obb_mode:
        for i, (px, py) in enumerate(manual_obb_points):
            cv2.circle(vis, (int(px), int(py)), 4, (255, 0, 255), -1)
            cv2.putText(
                vis, str(i + 1), (int(px) + 4, int(py) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA
            )

        if len(manual_obb_points) >= 2:
            pts = np.array(manual_obb_points, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (255, 0, 255), 1)

        cv2.putText(
            vis,
            f"MANUAL OBB MODE: click 4 corners ({len(manual_obb_points)}/4)",
            (10, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    active_name = CLASS_NAMES.get(active_class_id, str(active_class_id))
    area_val = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)

    header1 = f"Image [{img_idx + 1}/{total_imgs}] {img_name}"
    header2 = f"Active class: {active_class_id} ({active_name}) | Selected object: {selected_idx if objects else 'none'}"
    header3 = "Keys: 0-9 class | Space add SAM2 obj | m manual OBB | x delete selected | c clear clicks"
    header4 = "s save | a prev | d next | w save+next | u undo last | r rerun AMG fallback | esc cancel manual"
    header5 = f"Mouse: click OBB=select | Left empty=positive | Right empty=negative | Middle OBB=delete | Fallback area: {area_val}"

    cv2.putText(vis, header1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, header2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, header3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, header4, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, header5, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def build_result_preview(image, mask, obb, class_id, source="sam2"):
    result = image.copy()

    overlay = result.copy()
    overlay[mask > 0] = [0, 200, 255]
    result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

    if obb is not None:
        box = obb["box"].astype(np.int32)
        cv2.polylines(result, [box], True, object_color(source), 2)

        (cx, cy), (w, h), angle = obb["rect"]
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        txt = f"{class_id}:{class_name} [{source}] | w={w:.1f} h={h:.1f} a={angle:.1f}"
        cv2.putText(
            result, txt, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )

    return result


# =========================================================
# MOUSE
# =========================================================
def mouse_callback(event, x, y, flags, param):
    global points, point_labels, saved_objects, selected_object_index
    global manual_obb_mode, manual_obb_points, current_class_id

    if manual_obb_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            manual_obb_points.append([x, y])
            print(f"Manual OBB point {len(manual_obb_points)}: ({x}, {y})")

            if len(manual_obb_points) == 4:
                obb = create_obb_from_4_points(manual_obb_points)
                if obb is not None:
                    saved_objects.append({
                        "class_id": current_class_id,
                        "source": "manual_obb",
                        "confidence": 1.0,
                        "obb": obb,
                        "mask": None,
                    })
                    selected_object_index = len(saved_objects) - 1
                    print(f"Added manual OBB: class {current_class_id} ({CLASS_NAMES[current_class_id]})")

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
# SAM2 AMG FALLBACK
# =========================================================
def score_sam_candidate(obb):
    if obb is None:
        return None

    area = obb["area"]
    aspect_ratio = obb["aspect_ratio"]
    rectangularity = obb["rectangularity"]

    if aspect_ratio is None or rectangularity is None:
        return None

    if aspect_ratio < FALLBACK_MIN_ASPECT_RATIO:
        return None
    if rectangularity < FALLBACK_MIN_RECTANGULARITY or rectangularity > FALLBACK_MAX_RECTANGULARITY:
        return None

    return (1.0 * area) + (800.0 * aspect_ratio) + (1200.0 * rectangularity)


def detect_sam_fallback(image_bgr):
    if sam2_mask_generator is None:
        return None

    min_area = max(1, cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        masks = sam2_mask_generator.generate(image_rgb)
    except Exception as e:
        print(f"AMG fallback failed: {e}")
        return None

    if not masks:
        return None

    masks = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)[:FALLBACK_MAX_MASKS]

    best = None
    best_score = -1.0

    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        obb = mask_to_obb(seg, min_area=min_area)
        if obb is None:
            continue

        score = score_sam_candidate(obb)
        if score is None:
            continue

        if score > best_score:
            best_score = score
            best = {
                "class_id": 0,
                "source": "sam2_amg",
                "confidence": float(score),
                "obb": obb,
                "mask": seg.copy(),
            }

    return best


def get_auto_proposals_for_current_image(image_bgr):
    proposals = []

    if ENABLE_AMG_FALLBACK:
        fallback_obj = detect_sam_fallback(image_bgr)
        if fallback_obj is not None:
            proposals.append(fallback_obj)
            print("Added SAM2 AMG fallback proposal")

    return proposals


# =========================================================
# IMAGE LOADING
# =========================================================
def load_image_at_index(idx):
    global current_image_index, current_image_bgr, current_image_rgb
    global saved_objects, points, point_labels, selected_object_index
    global manual_obb_mode, manual_obb_points

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

    label_path = get_label_path(img_path)
    if label_path.exists():
        saved_objects = load_yolo_obb(label_path, current_image_bgr.shape)
        print(f"\nLoaded existing labels from file: {label_path}")
    else:
        saved_objects = get_auto_proposals_for_current_image(current_image_bgr)
        print(f"\nAuto-loaded proposals: {len(saved_objects)}")

    if saved_objects and any(obj["source"] == "sam2_amg" for obj in saved_objects):
        fb = next(obj for obj in saved_objects if obj["source"] == "sam2_amg")
        preview = build_result_preview(current_image_bgr, fb["mask"], fb["obb"], fb["class_id"], source="sam2_amg")
        cv2.imshow(RESULT_WINDOW, preview)

    print(f"Loaded image {current_image_index + 1}/{len(image_paths)}: {img_path.name}")


def rerun_auto_proposals():
    global saved_objects, selected_object_index, points, point_labels
    global manual_obb_mode, manual_obb_points

    points = []
    point_labels = []
    manual_obb_mode = False
    manual_obb_points = []
    saved_objects = get_auto_proposals_for_current_image(current_image_bgr)
    selected_object_index = 0

    print(f"Recomputed proposals: {len(saved_objects)}")

    if saved_objects and any(obj["source"] == "sam2_amg" for obj in saved_objects):
        fb = next(obj for obj in saved_objects if obj["source"] == "sam2_amg")
        preview = build_result_preview(current_image_bgr, fb["mask"], fb["obb"], fb["class_id"], source="sam2_amg")
        cv2.imshow(RESULT_WINDOW, preview)


def save_current_image():
    img_path = image_paths[current_image_index]
    txt_path = get_label_path(img_path)
    vis_path = get_vis_path(img_path)

    save_yolo_obb(txt_path, current_image_bgr.shape, saved_objects)

    vis = draw_ui(
        current_image_bgr, [], [], saved_objects, current_class_id,
        img_path.name, current_image_index, len(image_paths), selected_object_index
    )
    cv2.imwrite(str(vis_path), vis)

    print(f"Saved labels: {txt_path}")
    print(f"Saved visualization: {vis_path}")
    print(f"Objects saved: {len(saved_objects)}")


# =========================================================
# MAIN
# =========================================================
def main():
    global sam2_predictor, sam2_mask_generator
    global current_class_id, selected_object_index, saved_objects
    global points, point_labels, image_paths
    global manual_obb_mode, manual_obb_points

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    print("[INFO] Configuration loaded from:", CONFIG_PATH)
    print("[INFO] Image directory:", IMAGE_DIR)
    print("[INFO] Label directory:", LABEL_DIR)
    print("[INFO] Visualization directory:", VIS_DIR)
    print("[INFO] SAM2 root:", SAM2_ROOT)
    print("[INFO] SAM2 cfg:", SAM2_MODEL_CFG)
    print("[INFO] SAM2 checkpoint:", SAM2_CHECKPOINT)

    print("Loading SAM2...")
    sam2_model = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT), device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    if ENABLE_AMG_FALLBACK:
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=AMG_POINTS_PER_SIDE,
            pred_iou_thresh=AMG_PRED_IOU_THRESH,
            stability_score_thresh=AMG_STABILITY_SCORE_THRESH,
            crop_n_layers=AMG_CROP_N_LAYERS,
            crop_n_points_downscale_factor=AMG_CROP_N_POINTS_DOWNSCALE_FACTOR,
            box_nms_thresh=AMG_BOX_NMS_THRESH,
            min_mask_region_area=AMG_MIN_MASK_REGION_AREA,
        )

    print("\nClasses:")
    for k, v in CLASS_NAMES.items():
        print(f"  {k} = {v}")

    print("\nControls:")
    print("  Left click  = select object if on OBB, otherwise positive SAM2 point")
    print("  Right click = select object if on OBB, otherwise negative SAM2 point")
    print("  Middle click= delete object if on OBB")
    print("  0..9        = select class")
    print("  Space       = run SAM2 and add object")
    print("  m           = manual OBB mode (click 4 corners)")
    print("  Backspace   = remove last manual OBB point")
    print("  Esc         = cancel manual OBB mode")
    print("  x           = delete selected object")
    print("  c           = clear current SAM2 clicks")
    print("  u           = undo last object")
    print("  s           = save current image")
    print("  a / d       = previous / next image")
    print("  w           = save and next")
    print("  r           = rerun AMG fallback for current image")
    print("  q           = quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, TRACKBAR_MIN_AREA_INIT, TRACKBAR_MIN_AREA_MAX, lambda x: None)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    load_image_at_index(0)

    while True:
        img_name = image_paths[current_image_index].name
        vis = draw_ui(
            current_image_bgr,
            points,
            point_labels,
            saved_objects,
            current_class_id,
            img_name,
            current_image_index,
            len(image_paths),
            selected_object_index
        )
        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(20) & 0xFF

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

        elif key == ord(" "):
            if manual_obb_mode:
                print("Finish or cancel manual OBB mode first.")
                continue

            if len(points) == 0:
                print("No points")
                continue

            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(point_labels, dtype=np.int32)

            masks, scores, _ = sam2_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]

            obb = mask_to_obb(best_mask, min_area=MIN_MASK_AREA)
            if obb is None:
                print("No valid OBB found from SAM2 mask")
                continue

            saved_objects.append({
                "class_id": current_class_id,
                "source": "sam2",
                "confidence": float(scores[best_idx]),
                "obb": obb,
                "mask": best_mask.copy(),
            })
            selected_object_index = len(saved_objects) - 1

            preview = build_result_preview(current_image_bgr, best_mask, obb, current_class_id, source="sam2")
            cv2.imshow(RESULT_WINDOW, preview)

            print(f"Added SAM2 object: class {current_class_id} ({CLASS_NAMES[current_class_id]})")
            points = []
            point_labels = []

        elif key == ord("m"):
            manual_obb_mode = True
            manual_obb_points = []
            points = []
            point_labels = []
            print(f"Manual OBB mode enabled for class {current_class_id} ({CLASS_NAMES[current_class_id]})")

        elif key == 8:
            if manual_obb_mode and manual_obb_points:
                manual_obb_points.pop()
                print("Removed last manual OBB point")

        elif key == 27:
            if manual_obb_mode:
                manual_obb_mode = False
                manual_obb_points = []
                print("Cancelled manual OBB mode")

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

        elif key == ord("r"):
            rerun_auto_proposals()

        elif key == ord("q"):
            print("Exiting...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()