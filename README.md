# ALARS Labeling and Training

## Overview
This repository contains the labeling and training pipeline used for the ALARS perception system.

It is designed for three main purposes:

1. **Dataset labeling**, combining automatic and manual annotation tools.
2. **YOLO model training**, using the labeled datasets for object detection with oriented bounding boxes (OBB).
3. **Model documentation**, including the trained models obtained through the pipeline, together with their evaluation results and the datasets used for training.

The labeling pipeline is divided into two stages:

- **Part 1:** Automatic labeling using **SAM 3** and YOLO-based segmentation/detection.
- **Part 2:** Manual correction and refinement using **SAM 2** through an interactive UI.

This separation is useful because the two segmentation models require different environments and dependencies.

The trained YOLO models generated with this repository can later be used in the ROS 2 perception package: [alars_auv_perception](https://github.com/moyucrazy12/alars_auv_perception.git)


---

## Installation

Since this pipeline uses both **SAM 3** and **SAM 2**, it is recommended to install them in **separate Conda environments** to avoid dependency conflicts.

---

## Install SAM 3 Environment
The first model to install is **SAM 3**, based on the original repository: [facebookresearch/sam3](https://github.com/facebookresearch/sam3.git)

From the repository root:

```bash
cd labeling_pipeline
cd models
```

### 1. Create a new Conda environment
```bash
conda create -n part1_labeling_sam3 python=3.12
conda deactivate
conda activate part1_labeling_sam3
```

### 2. Install PyTorch with CUDA support
```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Clone the repository and install the package
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install ultralytics
```

> **Important**  
> Before using SAM 3, you must request access to the checkpoints from the SAM 3 Hugging Face repository:  
> [facebook/sam3](https://huggingface.co/facebook/sam3)
>
> Once access is granted, authenticate with Hugging Face so the checkpoints can be downloaded. For example:
>
> ```bash
> hf auth login
> ```

---

## Install SAM 2 Environment
The second model to install is **SAM 2**, based on the original repository: [facebookresearch/sam2](https://github.com/facebookresearch/sam2.git)

The code requires:
- `python >= 3.10`
- `torch >= 2.5.1`
- `torchvision >= 0.20.1`

From the repository root:

```bash
cd labeling_pipeline
cd models
```

### 1. Create a new Conda environment
```bash
conda create -n part2_labeling_sam2 python=3.10
conda deactivate
conda activate part2_labeling_sam2
```

### 2. Clone the repository and install the package
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 3. Download the checkpoints
All checkpoints can be downloaded with:

```bash
cd checkpoints && ./download_ckpts.sh && cd ..
```

Or downloaded individually:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

These checkpoints correspond to **SAM 2.1**, the improved version of SAM 2.

After installation, you can deactivate the environment if needed:

```bash
conda deactivate
```

---

## Labeling Pipeline

## Part 1: Automatic Labeling with SAM 3 + YOLO
The first stage of the labeling pipeline performs **automatic labeling** using:

- **SAM 3** for prompt-based segmentation
- **YOLO segmentation models** for general or newly introduced classes
- A **custom YOLO OBB model** trained to identify:
  - SAM
  - buoy
  - lolo
  - catamaran

This stage generates labels automatically and saves the corresponding **oriented bounding boxes (OBB)** for each image in the dataset.

### Before running Part 1
Make sure that:

- The required models are stored in the correct folders, especially:
  - `models/yolo_models/`
- Your custom YOLO models are available there.
- The images to label are located in:
  - `dataset_to_label/images/`
- The desired classes and confidence thresholds are configured in:
  - `part1_parameters.yaml`

### Run Part 1
```bash
conda deactivate
conda activate part1_labeling_sam3
cd labeling_pipeline
python3 scripts/part1_sam3_yolo.py
```

---

## Part 2: Manual Labeling and Correction with SAM 2
The second stage is intended for **manual correction** when the automatically generated labels are not accurate enough.

This tool provides an interactive UI where the user can:

- click on an object to segment it with SAM 2
- generate an OBB from the selected region
- manually define an OBB using four corner points if segmentation is not accurate
- remove, redo, or adjust existing annotations

This makes manual labeling much faster than drawing all boxes from scratch.

### Run Part 2
```bash
conda deactivate
conda activate part2_labeling_sam2
cd labeling_pipeline
python3 scripts/part2_sam2_manual_labeling.py
```

---

## Manual Labeling UI Controls

| Control | Action |
|---|---|
| Left click | Select object if clicking on an existing OBB; otherwise add a positive SAM 2 point |
| Right click | Select object if clicking on an existing OBB; otherwise add a negative SAM 2 point |
| Middle click | Delete object if clicking on an existing OBB |
| `0..5` | Select class |
| `Space` | Run SAM 2 and add object |
| `m` | Enter manual OBB mode (click 4 corners) |
| `Backspace` | Remove last manual OBB point |
| `Esc` | Cancel manual OBB mode |
| `x` | Delete selected object |
| `c` | Clear current SAM 2 clicks |
| `u` | Undo last object |
| `s` | Save current image |
| `a / d` | Previous / next image |
| `w` | Save and move to next image |
| `r` | Rerun AMG fallback for current image |
| `q` | Quit |

---

## YOLO Models
Inside the `trained_models/` folder you can find the currently available YOLO models trained from `yolo11n-obb.pt`.

That folder should also include a more detailed description of:

- the datasets used for training
- the classes included in each model
- the corresponding evaluation results on a separate test set

A general summary is provided below.

### Available Models

| Model | Classes | Training Data | Description |
|---|---|---|---|
| `yolo_model_2cls_fisheye` | SAM, buoy | Real images with fisheye distortion | Model specialized for real fisheye data, focused only on SAM and buoy detection. |
| `yolo_model_2cls_mixed` | SAM, buoy | Real and simulated images, with and without fisheye distortion | More general 2-class model for SAM and buoy detection across mixed domains. |
| `yolo_model_5cls` | SAM, buoy, lolo, catamaran, boats | Real and simulated images, with and without fisheye distortion | Multi-class model for the main marine objects used in the perception pipeline. |
| `yolo_model_6cls` | SAM, buoy, lolo, catamaran, boats, people | Real and simulated images, with and without fisheye distortion | Extended model including people in addition to the marine object classes. |

These models can be used directly in the ROS 2 perception pipeline:  
[alars_auv_perception](https://github.com/moyucrazy12/alars_auv_perception.git)

---

## Training Process

For training, the dataset must follow the structure expected by:

```bash
training_pipeline/data.yaml
```

At minimum, you should have two main folders:

- `images/`
- `labels/`

Each of them should contain the standard split:

- `train/`
- `val/`
- `test/`

A typical structure is:

```bash
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Training Strategy
The training process is divided into **two stages**:

1. **Stage 1:**  
   Train for more epochs at a lower image resolution.  
   This helps the model learn the general structure of the task efficiently.

2. **Stage 2:**  
   Fine-tune for fewer epochs at a higher image resolution.  
   This helps improve performance, especially for **small-object detections**.

This two-stage strategy was chosen to balance training cost and final detection quality.

### Training Environment
The training environment is the same as the one described in the perception repository:  
[alars_auv_perception](https://github.com/moyucrazy12/alars_auv_perception.git)

### Run Training
Start with Stage 1:

```bash
python3 training_pipeline/train_stage1.py
```

After Stage 1 finishes, continue with Stage 2:

```bash
python3 training_pipeline/train_stage2.py
```

---

## Notes
- Keep the SAM 3 and SAM 2 environments separate to avoid dependency conflicts.
- Verify all model paths before running the labeling scripts.
- Verify the dataset structure before starting training.
- Keep `part1_parameters.yaml` updated with the correct classes and thresholds for automatic labeling.

---

## Maintainer
**Cristhian Mallqui Castro**  
ckmc@kth.se