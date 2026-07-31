# Training Pipeline

## Overview

The training pipeline prepares the final train/val/test dataset, trains YOLO OBB models, tests trained models, and exports selected models into `trained_models/`.

It is designed to support repeated dataset updates without changing old split assignments.

## Structure

```text
training_pipeline/
├── config/
│   ├── merge_split_dataset.yaml
│   ├── split_registry.csv
│   ├── alars_combined_data.yaml
│   ├── train_combined_params.yaml
│   ├── test_model_params.yaml
│   └── export_model_params.yaml
│
├── scripts/
│   ├── merge_split_dataset.py
│   ├── train_common.py
│   ├── train_combined_finetune.py
│   ├── train_combined_pretrained.py
│   ├── test_model.py
│   └── export_best_model.py
│
└── dataset_combined/
    ├── images/
    └── labels/
```

## Dataset merge/split

The merge/split step combines old fixed data and new labeled data into one YOLO dataset.

Config:

```text
training_pipeline/config/merge_split_dataset.yaml
```

Run:

```bash
docker compose run --rm merge_split_dataset
```

Output:

```text
training_pipeline/dataset_combined/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

It also generates:

```text
training_pipeline/config/alars_combined_data.yaml
```

This file is used by Ultralytics training and testing.

## Split registry

The split registry is a folder-level CSV:

```text
training_pipeline/config/split_registry.csv
```

Format:

```csv
source,folder,split
old,video_01_boat_background_2,train
old,video_04_boat_sam_under,val
new,bag_01,test
```

Behavior:

```text
- Folders already in split_registry.csv keep their split.
- Old already_split folders are registered from images/train, images/val, images/test.
- Only new unseen folders are assigned.
```

This avoids split drift when new data is added later.

## Training config

Training paths and run names are defined in:

```text
training_pipeline/config/train_combined_paths.yaml
```

Example:

```yaml
data:
  yaml: training_pipeline/config/alars_combined_data.yaml

runs:
  project: runs_mixed_hook

models:
  finetune:
    path: trained_models/current_best.pt

  pretrained:
    source: yolo11n-obb.pt
    dir: training_pipeline/base_models
    download_if_missing: true

names:
  finetune: M_finetune_1024_sam_buoy_hook_combined
  pretrained: M_pretrained_1024_sam_buoy_hook_combined
```

## Fine-tuning

Fine-tuning starts from an existing trained model.

Run:

```bash
docker compose run --rm train_combined_finetune
```

Use this when you want to continue improving an existing ALARS model with new data.

## Training from YOLO pretrained weights

This starts from the official YOLO11 OBB pretrained model.

Run:

```bash
docker compose run --rm train_combined_pretrained
```

If the base model is missing, the script can download it and save it under:

```text
training_pipeline/base_models/
```

This option is useful when you want to compare fine-tuning against a fresh YOLO baseline or when the dataset has changed enough that starting from the official pretrained weights may be cleaner.

## Testing

Testing uses one generic script. The only thing that changes is the model path in the config.

Config:

```text
training_pipeline/config/test_combined_model.yaml
```

Run:

```bash
docker compose run --rm test_model
```

The test script performs:

```text
1. Quantitative evaluation using model.val(split="test")
2. Optional qualitative prediction visualization on images/test
```

Outputs are saved under the configured test project folder, for example:

```text
runs_yolo_training/test_results/
```

## Exporting best model

After training, export the selected `best.pt` from the raw Ultralytics run folder into `trained_models/`.

Config:

```text
training_pipeline/config/export_best_model.yaml
```

Run:

```bash
docker compose run --rm export_best_model
```

Example output:

```text
trained_models/yolo11n_obb_pretrained_combined_best.pt
```

The export step copies the model. It does not delete or modify the original file inside `runs_yolo_training/`.

## Recommended workflow

```bash
# 1. Merge old and new datasets
docker compose run --rm merge_split_dataset

# 2. Train
docker compose run --rm train_combined_finetune

# or
docker compose run --rm train_combined_pretrained

# 3. Test
docker compose run --rm test_model

# 4. Export selected best model
docker compose run --rm export_best_model
```

## Notes

- Do not manually edit `training_pipeline/dataset_combined/`; it is generated.
- Keep `split_registry.csv` under version control if stable experiment splits must be preserved.
- Keep final reusable models in `trained_models/`.
- Keep raw experiment outputs in `runs_yolo_training/`.