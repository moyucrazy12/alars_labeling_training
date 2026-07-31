#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path.cwd()

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained YOLO model on the combined dataset."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to test config YAML.",
    )

    return parser.parse_args()

def resolve_repo_path(path_value: str) -> Path:
    path = Path(str(path_value).strip()).expanduser()

    if path.is_absolute():
        return path

    return (REPO_ROOT / path).resolve()

def load_yaml(path: Path) -> dict:
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty config file: {path}")

    return cfg

def find_images_recursive(root: Path):
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]

    if not root.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {root}")

    images = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)

def write_prediction_source_list(image_paths, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for image_path in image_paths:
            f.write(str(image_path.resolve()) + "\n")

    return output_path

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    model_path = resolve_repo_path(cfg["model"]["path"])
    data_yaml = resolve_repo_path(cfg["data"]["yaml"])
    test_source = resolve_repo_path(cfg["test"]["source"])

    # Example: obb, detect, segment, pose, classify
    task = str(cfg["test"].get("task", "obb"))

    project = resolve_repo_path(cfg["runs"]["project"])
    test_name = str(cfg["runs"].get("name", "test_current_model"))
    prediction_name = str(cfg["runs"].get("prediction_name", "pred_current_model"))

    save_predictions = bool(cfg.get("output", {}).get("save_predictions", True))

    val_cfg = cfg.get("validation", {})
    imgsz = int(val_cfg.get("imgsz", 1024))
    batch = int(val_cfg.get("batch", 8))
    conf = float(val_cfg.get("conf", 0.25))
    iou = float(val_cfg.get("iou", 0.45))
    rect = bool(val_cfg.get("rect", False))
    plots = bool(val_cfg.get("plots", True))

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    if save_predictions and not test_source.exists():
        raise FileNotFoundError(f"Test image folder not found: {test_source}")

    print("[INFO] Testing model")
    print(f"[INFO] Model:           {model_path}")
    print(f"[INFO] Data YAML:       {data_yaml}")
    print(f"[INFO] Test source:     {test_source}")
    print(f"[INFO] Task:            {task}")
    print(f"[INFO] Output project:  {project}")
    print(f"[INFO] Test name:       {test_name}")
    print(f"[INFO] Prediction name: {prediction_name}")
    print(f"[INFO] Image size:      {imgsz}")
    print(f"[INFO] Batch:           {batch}")
    print(f"[INFO] Conf:            {conf}")
    print(f"[INFO] IoU:             {iou}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        task=task,
        split="test",
        imgsz=imgsz,
        batch=batch,
        rect=rect,
        plots=plots,
        project=str(project),
        name=test_name,
    )

    print("[DONE] Quantitative test finished.")
    print(metrics)

    if save_predictions:
        image_paths = find_images_recursive(test_source)

        if not image_paths:
            raise FileNotFoundError(
                f"No images found recursively inside:\n"
                f"  {test_source}\n\n"
                f"Check test.source in your config."
            )

        print(f"[INFO] Prediction images found recursively: {len(image_paths)}")

        source_list_path = project / f"{prediction_name}_sources.txt"
        source_list_path = write_prediction_source_list(
            image_paths=image_paths,
            output_path=source_list_path,
        )

        print(f"[INFO] Prediction source list: {source_list_path}")

        model.predict(
            source=str(source_list_path),
            task=task,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=True,
            save_txt=False,
            project=str(project),
            name=prediction_name,
        )

        print("[DONE] Prediction visualization finished.")

if __name__ == "__main__":
    main()