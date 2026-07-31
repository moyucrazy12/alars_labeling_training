#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO

from train_common import load_training_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training_pipeline/config/train_combined_paths.yaml"),
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_training_paths("pretrained", config_path=args.config)

    model = YOLO(cfg["model"])

    results = model.train(
        data=cfg["data"],
        task="obb",

        epochs=220,
        patience=45,

        imgsz=1024,
        rect=False,
        batch=8,

        degrees=8,
        translate=0.08,
        scale=0.30,
        shear=1.5,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        hsv_h=0.005,
        hsv_s=0.28,
        hsv_v=0.32,

        mosaic=0.45,
        mixup=0.0,
        close_mosaic=50,

        label_smoothing=0.0,

        optimizer="AdamW",
        lr0=0.0015,
        warmup_epochs=4,
        weight_decay=0.005,
        cos_lr=True,

        project=cfg["project"],
        name=cfg["name"],
    )

    print(results)

if __name__ == "__main__":
    main()