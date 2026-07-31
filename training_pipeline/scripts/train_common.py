#!/usr/bin/env python3
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO_ROOT = Path.cwd()
DEFAULT_CONFIG_PATH = Path("training_pipeline/config/train_combined_paths.yaml")

def resolve_repo_path(path_value: str) -> Path:
    path = Path(str(path_value).strip()).expanduser()

    if path.is_absolute():
        return path

    return (REPO_ROOT / path).resolve()

def looks_like_local_path(value: str) -> bool:
    value = str(value).strip()

    return (
        "/" in value
        or value.startswith(".")
        or value.startswith("~")
    )

def download_ultralytics_model(source_model: str, target_path: Path) -> Path:
    target_path = resolve_repo_path(str(target_path))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"[INFO] Pretrained model already exists: {target_path}")
        return target_path

    print("[INFO] Downloading Ultralytics pretrained model")
    print(f"[INFO] Source: {source_model}")
    print(f"[INFO] Target: {target_path}")

    model = YOLO(source_model)

    # Save a local copy of the downloaded model in your chosen folder.
    model.save(str(target_path))

    if not target_path.exists():
        raise FileNotFoundError(
            f"Model was downloaded but not saved to expected path: {target_path}"
        )

    print(f"[INFO] Saved pretrained model to: {target_path}")
    return target_path

def resolve_model_value(model_cfg) -> str:
    """
    Supported formats:

    finetune:
      path: runs_mixed_hook/old_run/weights/best.pt

    pretrained:
      source: yolo11n-obb.pt
      dir: training_pipeline/base_models
      download_if_missing: true

    Also supports old simple format:

    pretrained: yolo11n-obb.pt
    finetune: runs_mixed_hook/old_run/weights/best.pt
    """

    # Old/simple format
    if isinstance(model_cfg, str):
        model_string = model_cfg.strip()

        if looks_like_local_path(model_string):
            model_path = resolve_repo_path(model_string)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            return str(model_path)

        return model_string

    if not isinstance(model_cfg, dict):
        raise ValueError(f"Invalid model config: {model_cfg}")

    source = model_cfg.get("source")
    path = model_cfg.get("path")
    model_dir = model_cfg.get("dir")
    download_if_missing = bool(model_cfg.get("download_if_missing", False))

    # Explicit local path, mostly for finetune.
    if path:
        model_path = resolve_repo_path(path)

        if model_path.exists():
            return str(model_path)

        if download_if_missing:
            if not source:
                raise ValueError(
                    "download_if_missing: true requires a source model name."
                )

            downloaded_path = download_ultralytics_model(
                source_model=str(source),
                target_path=model_path,
            )
            return str(downloaded_path)

        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Directory + source, mostly for pretrained.
    if model_dir and source:
        model_path = resolve_repo_path(model_dir) / Path(str(source)).name

        if model_path.exists():
            return str(model_path)

        if download_if_missing:
            downloaded_path = download_ultralytics_model(
                source_model=str(source),
                target_path=model_path,
            )
            return str(downloaded_path)

        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Only source, for official Ultralytics names like yolo11n-obb.pt.
    if source:
        return str(source)

    raise ValueError(f"Invalid model config: {model_cfg}")

def load_training_paths(experiment_key: str, config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty training config: {config_path}")

    if experiment_key not in cfg["models"]:
        available = ", ".join(cfg["models"].keys())
        raise KeyError(
            f"Unknown experiment key '{experiment_key}'. Available: {available}"
        )

    data_yaml = resolve_repo_path(cfg["data"]["yaml"])
    project = resolve_repo_path(cfg["runs"]["project"])
    model = resolve_model_value(cfg["models"][experiment_key])
    name = str(cfg["names"][experiment_key])

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    print("[INFO] Experiment:", experiment_key)
    print("[INFO] Model:    ", model)
    print("[INFO] Data YAML:", data_yaml)
    print("[INFO] Project:  ", project)
    print("[INFO] Name:     ", name)

    return {
        "model": model,
        "data": str(data_yaml),
        "project": str(project),
        "name": name,
    }