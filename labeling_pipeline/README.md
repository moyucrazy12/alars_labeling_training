# Labeling Pipeline

## Overview

The labeling pipeline prepares image datasets for YOLO OBB training.

It contains:

```text
1. Frame selection
2. Automatic labeling
3. Manual correction
```

The goal is to reduce repeated frames, generate initial labels automatically, and then correct labels manually when needed.

## Structure

```text
labeling_pipeline/
├── config/
│   ├── frame_selection_folder.yaml
│   ├── frame_selection_rosbag.yaml
│   ├── part1_parameters.yaml
│   └── part2_parameters.yaml
│
├── scripts/
│   ├── frame_selection_folder.py
│   ├── frame_selection_rosbag.py
│   ├── part1_sam3_yolo_auto_labeling.py
│   └── part2_sam2_manual_labeling.py
│
├── models/
│   ├── sam2/
│   ├── sam3/
│   └── yolo_models/
│
└── dataset_to_label/
    ├── images/
    └── labels/
```
## Frame selection

Frame selection is an optional preprocessing step used to reduce repeated or highly similar images before labeling. This is useful when the dataset comes from videos or rosbags, where many consecutive frames can look almost identical.

The frame-selection tools use [DINOv2](https://github.com/facebookresearch/dinov2) to extract visual embeddings from each image. These embeddings describe the visual content of the image better than raw pixel differences, making them more useful for identifying semantically similar frames.

After extracting the embeddings, [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) is used to group visually similar frames. HDBSCAN is useful here because it does not require manually choosing the number of clusters, and it can also mark unusual frames as noise/outliers, this helps keep representative images while removing redundant ones.

This step helps to:

- reduce labeling time
- avoid training with many repeated frames
- keep more diverse images for YOLO training
- reduce the risk of overfitting to nearly identical samples

---

## Frame selection from image folder

This service selects representative images from an existing image dataset.

Config:

```text
labeling_pipeline/config/frame_selection_folder.yaml
```

Run:

```bash
docker compose run --rm frame_selection_folder
```

This service can be used directly on an existing image folder. It can also be used after `frame_selection_rosbag` if the extracted rosbag frames still contain too many similar images or if a stronger DINOv2 model should be used for additional filtering.

---

## Frame selection from ROS 2 rosbags

This service reads ROS 2 bags, extracts image messages, and keeps representative frames.

Config:

```text
labeling_pipeline/config/frame_selection_rosbag.yaml
```

Run:

```bash
docker compose run --rm frame_selection_rosbag
```

Expected input:

```text
data/rosbags/
```

The output is organized by rosbag name, expected:

```text
labeling_pipeline/dataset_to_label/
├── images/
│   └── rosbag_name/
└── labels/
    └── rosbag_name/
```

---

## Important frame-selection parameters

Both `frame_selection_folder.yaml` and `frame_selection_rosbag.yaml` use mostly the same frame-selection parameters. The values can be different depending on the use case. For example, rosbag processing can take longer because frames must be extracted and processed from the bag, so it may be better to start with faster or less strict settings there.

| Parameter | Purpose |
|---|---|
| `model_name` | DINOv2 model used for visual embeddings. Larger models usually give better features but are slower. |
| `image_size` | Image size used before computing embeddings. Larger values can improve quality but increase memory usage. |
| `embedding_mode` | Defines whether embeddings are computed from the whole image or from image tiles. Tiled embeddings are useful when small objects matter. |
| `tile_grid` | Number of tiles used when tiled/global-tiled features are enabled. More tiles can improve small-object sensitivity but are slower. |
| `min_cluster_size` | Main HDBSCAN parameter. Smaller values keep more fine-grained groups; larger values remove more redundancy. |
| `min_samples` | Controls how conservative HDBSCAN is. Higher values usually mark more samples as outliers. |
| `keep_noise` | Whether to keep HDBSCAN outliers. Keeping them is usually useful because outliers may contain rare or important cases. |
| `save_reports` | Saves CSV/contact-sheet reports for debugging the selected frames. |

For a first pass, the default DINOv2 model is usually enough. For a cleaner or more selective dataset, run `frame_selection_folder` afterwards with a stronger DINOv2 model or stricter clustering parameters.

## Part 1: Automatic labeling

It is intended to provide a first set of annotations before manual review, so the user does not need to label every object from scratch.

This stage combines:

- [SAM 3](https://github.com/facebookresearch/sam3) prompt-based segmentation to generate object masks.
- [YOLO segmentation/detection](https://docs.ultralytics.com/tasks/segment) models to find candidate objects in the image.
- Optional custom [YOLO11 OBB](https://docs.ultralytics.com/tasks/obb/) models to include classes that are specific to the ALARS dataset.

The output is a set of YOLO OBB label files, together with optional visualizations that can be checked before continuing to manual correction in Part 2.

Config:

```text
labeling_pipeline/config/part1_parameters.yaml
```

Run:

```bash
docker compose run --rm part1
```

Typical input:

```text
labeling_pipeline/dataset_to_label/images/
```

Typical output:

```text
labeling_pipeline/dataset_to_label/labels/
labeling_pipeline/dataset_to_label/visualizations_part1/
```

## Part 2: Manual correction

Part 2 opens an interactive UI to review, correct, and add labels manually. It is used after automatic labeling when some detections are missing, inaccurate, or need small adjustments.

Through the UI, the user can add positive and negative points for [SAM 2](https://github.com/facebookresearch/sam2) segmentation, create new OBB labels, delete incorrect annotations, adjust existing labels, and save the corrected label files.

This step is useful for improving label quality before the dataset is used for YOLO OBB training.
Config:

```text
labeling_pipeline/config/part2_parameters.yaml
```

Before running the UI, allow Docker to use the display:

```bash
export DISPLAY=:1
xhost +local:
docker compose run --rm part2
xhost -local:
```

The display value may change depending on the machine.

## Manual labeling controls

| Control | Action |
|---|---|
| Left click | Select object if clicking on an existing OBB; otherwise add positive SAM 2 point. |
| Right click | Select object if clicking on an existing OBB; otherwise add negative SAM 2 point. |
| Middle click | Delete object if clicking on an existing OBB. |
| `0..9` | Select class. |
| `Space` | Run SAM 2 and add object. |
| `m` | Enter manual OBB mode. |
| `Backspace` | Remove last manual OBB point. |
| `Esc` | Cancel manual OBB mode. |
| `o` | Orientation/alignment mode. |
| Arrow keys | Rotate/refit selected OBB. |
| `x` | Delete selected object. |
| `c` | Clear current SAM 2 clicks. |
| `u` | Undo last object. |
| `s` | Save current image. |
| `a / d` | Previous / next image. |
| `w` | Save and move to next image. |
| `q` | Quit. |

## Notes

- The pipeline uses separate Docker images for frame selection, Part 1, Part 2, and training because each step has different dependencies.
- The frame-selection image includes the tools for DINOv2, HDBSCAN, and ROS 2 rosbag reading.
- Keep class IDs consistent between Part 1, Part 2, and training.
- Check the generated labels visually before training.