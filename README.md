# ALARS Labeling and Training

## Overview

This repository provides the complete pipeline for  dataset preparation, dataset labeling, YOLO OBB training, testing, and trained-model export for the ALARS perception system.

In addition, the repository is structured as a ROS 2 resource package, which means other ROS 2 packages can find and load the trained YOLO models from the package share path, instead of relying on manually defined model paths.

The pipeline is fully Docker-based, which separates the environments required by [SAM 3](https://github.com/facebookresearch/sam3), [SAM 2](https://github.com/facebookresearch/sam2), ROS 2 rosbag tools, and YOLO training, thereby reducing dependency conflicts and making the workflow easier to reproduce across machines.

## Main components

The repository is organized into four main parts:

- `docker/`: Dockerfiles for the different pipeline environments.
- `labeling_pipeline/`: frame selection, automatic labeling, and manual correction tools.
- `training_pipeline/`: dataset merge/split, YOLO training, testing, and model export tools.
- `trained_models/`: exported YOLO models used by the ALARS perception system.

## Main trained models

The repository currently includes two main YOLO OBB models in `trained_models/`:

| Model | Classes | Status |
|---|---|---|
| `yolo_model_2cls_may.pt` | `sam`, `buoy` | Two-class model used for the Djuro demonstration and tested in Askö. |
| `yolo_model_4cls_july.pt` | `sam`, `buoy`, `hook`, `landing_pad` | Extended four-class model. It has been tested in simulation so far. |

These models are intended to be loaded by the ALARS perception system through the ROS 2 package share path.

## Docker requirements

Install:

1. Docker Engine
2. Docker Compose
3. NVIDIA GPU driver
4. NVIDIA Container Toolkit

Use the official installation guides:

- [Docker Engine installation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

After installation, verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Environment variables

The `.env` file defines the user created inside the containers:

```env
USER_UID=1000
USER_GID=999
USER_NAME=smarc
```

These values are used by Docker Compose as build arguments. They help avoid permission problems when the container creates files inside the mounted repository.

## Build containers

From the repository root:

```bash
docker compose build
```

Or build a specific service:

```bash
docker compose build part1
docker compose build part2
docker compose build training-stage1
```

## Main workflow

A typical full workflow is:

```bash
# 1. Optional: extract/select frames from ROS 2 rosbags
docker compose run --rm frame_selection_rosbag

# 2. Optional: further filter the extracted frames from an image folder
# This can be useful if you want to use a stronger DINOv2 model or apply extra filtering.
docker compose run --rm frame_selection_folder

# 3. Automatic labeling
docker compose run --rm part1

# 4. Manual correction
docker compose run --rm part2

# 5. Merge old and new datasets into a stable train/val/test dataset
docker compose run --rm merge_split_dataset

# 6. Train from a previous model
docker compose run --rm train_combined_finetune

# 7. Or train from official YOLO11 OBB pretrained weights
docker compose run --rm train_combined_pretrained

# 8. Test any trained model
docker compose run --rm test_model

# 9. Export the selected best model to trained_models/
docker compose run --rm export_best_model
```

`frame_selection_folder` can be used either directly on an existing image folder or after `frame_selection_rosbag` if the extracted frames still need additional filtering.

## Docker services

| Service | Purpose |
|---|---|
| `frame_selection_rosbag` | Extract/select representative images from ROS 2 rosbags. |
| `frame_selection_folder` | Select representative frames from an existing image folder using [DINOv2](https://github.com/facebookresearch/dinov2) + [HDBSCAN](https://joss.theoj.org/papers/10.21105/joss.00205). It can also be used after rosbag frame extraction for additional filtering. |
| `part1` | Automatic labeling using [SAM 3](https://github.com/facebookresearch/sam3) and [YOLO](https://docs.ultralytics.com/models/yolo11/). |
| `part2` | Manual correction using the [SAM 2](https://github.com/facebookresearch/sam2) interactive UI. |
| `merge_split_dataset` | Merge old/new datasets using a stable folder-level split registry. |
| `train_combined_finetune` | Fine-tune from an existing trained model. |
| `train_combined_pretrained` | Train from official [YOLO11 OBB](https://docs.ultralytics.com/tasks/obb/) pretrained weights. |
| `test_model` | Evaluate and optionally visualize predictions for a trained model. |
| `export_best_model` | Copy the best model from `runs/` into `trained_models/` with a clean name. |

## ROS 2 usage: models only

This repository can also be used as a ROS 2 resource package to expose trained models to other ROS 2 packages.

If the repository is used as a submodule inside a ROS 2 workspace:

```bash
colcon build --symlink-install --packages-select alars_labeling_training
source install/setup.bash
```

Other ROS 2 packages can then locate the package share path and load models from `trained_models/`.

## Important folders

| Folder | Description |
|---|---|
| `data/rosbags/` | Input folder for ROS 2 rosbags used by the frame-selection pipeline. |
| `data/old_dataset/` | Previous dataset, already divided into `train`, `val`, and `test` if available. This is used by the merge/split step and the split registry. |
| `labeling_pipeline/dataset_to_label/` | Current dataset being labeled. |
| `training_pipeline/dataset_combined/` | Generated combined train/val/test dataset. It can be recreated. |
| `training_pipeline/config/split_registry.csv` | Stable folder-level train/val/test registry. |
| `runs_yolo_training/` | Raw training outputs from Ultralytics. |
| `trained_models/` | Clean exported model files for reuse. |

## Notes

- Do not manually edit `training_pipeline/dataset_combined/`; it is generated.
- Keep raw datasets and labeled datasets as source data.
- Keep `data/old_dataset/` if you want to preserve and reuse an existing train/val/test split.
- Keep the split registry to preserve stable train/val/test assignments when adding new datasets.
- Export final models into `trained_models/` instead of manually copying from `runs/`.

## Maintainer

Cristhian Mallqui Castro  
ckmc@kth.se