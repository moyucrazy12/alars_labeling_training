#!/usr/bin/env python3
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a YOLO best.pt from runs/ into trained_models/ with a clean name."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to export config YAML.",
    )

    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Optional override for source.model_key, for example: pretrained or finetune.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional override for output.filename.",
    )

    parser.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="Optional override for source.weights_file, usually best.pt or last.pt.",
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
        raise ValueError(f"Empty YAML file: {path}")

    return cfg

def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def resolve_collision(dst_path: Path, if_exists: str) -> Path:
    if not dst_path.exists():
        return dst_path

    if if_exists == "overwrite":
        return dst_path

    if if_exists == "error":
        raise FileExistsError(
            f"Output model already exists:\n"
            f"  {dst_path}\n\n"
            f"Change output.if_exists to overwrite or timestamp."
        )

    if if_exists == "timestamp":
        stamped_name = f"{dst_path.stem}_{timestamp_string()}{dst_path.suffix}"
        return dst_path.with_name(stamped_name)

    raise ValueError(
        "output.if_exists must be one of: error, overwrite, timestamp"
    )

def get_source_model_path(
    export_cfg: dict,
    model_key_override=None,
    weights_file_override=None,
):
    source_cfg = export_cfg["source"]

    # Optional direct mode:
    #
    # source:
    #   path: runs_mixed_hook/my_run/weights/best.pt
    if "path" in source_cfg:
        source_path = resolve_repo_path(source_cfg["path"])
        model_key = str(source_cfg.get("model_key", "manual"))
        weights_file = source_path.name
        return source_path, model_key, weights_file

    training_config_path = resolve_repo_path(source_cfg["training_config"])
    training_cfg = load_yaml(training_config_path)

    model_key = model_key_override or str(source_cfg["model_key"])
    weights_file = weights_file_override or str(source_cfg.get("weights_file", "best.pt"))

    if model_key not in training_cfg["names"]:
        available = ", ".join(training_cfg["names"].keys())
        raise KeyError(
            f"model_key '{model_key}' was not found in training config names.\n"
            f"Available keys: {available}"
        )

    project_dir = resolve_repo_path(training_cfg["runs"]["project"])
    run_name = str(training_cfg["names"][model_key])

    source_path = project_dir / run_name / "weights" / weights_file

    return source_path, model_key, weights_file

def main():
    args = parse_args()
    export_cfg = load_yaml(args.config)

    source_path, model_key, weights_file = get_source_model_path(
        export_cfg=export_cfg,
        model_key_override=args.model_key,
        weights_file_override=args.weights_file,
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source model does not exist:\n"
            f"  {source_path}\n\n"
            f"Check that training finished and that the run name in train_combined_paths.yaml is correct."
        )

    output_cfg = export_cfg["output"]

    output_dir = resolve_repo_path(output_cfg.get("dir", "trained_models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = args.filename or str(output_cfg["filename"])
    dst_path = output_dir / filename

    if_exists = str(output_cfg.get("if_exists", "timestamp")).strip().lower()
    dst_path = resolve_collision(dst_path, if_exists)

    print("[INFO] Exporting model")
    print(f"[INFO] Model key:    {model_key}")
    print(f"[INFO] Weights file: {weights_file}")
    print(f"[INFO] Source:       {source_path}")
    print(f"[INFO] Destination:  {dst_path}")

    shutil.copy2(source_path, dst_path)

    print("[DONE] Model exported.")
    print(f"[DONE] Exported model: {dst_path}")

if __name__ == "__main__":
    main()