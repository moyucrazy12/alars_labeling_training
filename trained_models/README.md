# Trained Models

This document summarizes the trained YOLO models available in this repository, including their datasets, performance, and recommended usage.

---

## Overview

Model selection depends on your application:

- **2-class models (sam, buoy)** → best performance and robustness  
- **Multi-class models (5–6 classes)** → more semantic understanding but slight drop in core class performance  

---

## 1. yolo_model_2cls_fisheye.pt

### Description
- Classes: `sam`, `buoy`
- Trained only on fisheye data

### Dataset
https://kth-my.sharepoint.com/:f:/r/personal/ckmc_ug_kth_se/Documents/training_datasets/only_fisheye_dataset

### Performance
| Class | AP |
|------|----|
| sam  | 0.98 |
| buoy | 0.98 |

### Recommendation
Best for **fisheye-camera setups** with maximum accuracy on core classes.

---

## 2. yolo_model_2cls_mixed.pt

### Description
- Classes: `sam`, `buoy`
- Trained on **real + simulated data**, from normal and fisheye cameras

### Dataset
https://kth-my.sharepoint.com/:f:/r/personal/ckmc_ug_kth_se/Documents/training_datasets/real_sim_dataset

### Performance (Mixed Test Set)

| Metric | Value |
|--------|------|
| mAP@0.5 | 0.922 |
| mAP@0.5:0.95 | 0.771 |

| Class | AP@0.5 | AP@0.5:0.95 |
|------|--------|-------------|
| sam  | 0.984 | 0.890 |
| buoy | 0.860 | 0.652 |

### Recommendation
Best for **domain generalization** (real + sim environments).

---

## 3. yolo_model_5cls.pt

### Description
Classes:
- sam, buoy, lolo, catamaran, boat

### Dataset
https://kth-my.sharepoint.com/:f:/r/personal/ckmc_ug_kth_se/Documents/training_datasets/all_classes_dataset

### Performance

| Metric | Value |
|--------|------|
| mAP@0.5 | 0.945 |
| mAP@0.5:0.95 | 0.814 |

| Class | AP@0.5 | AP@0.5:0.95 |
|------|--------|-------------|
| sam | 0.951 | 0.802 |
| buoy | 0.829 | 0.592 |
| lolo | 0.995 | 0.902 |
| catamaran | 0.962 | 0.846 |
| boat | 0.989 | 0.931 |

### Recommendation
Best **trade-off between detection quality and semantic richness**.

---

## 4. yolo_model_6cls.pt

### Description
Classes:
- sam, buoy, lolo, catamaran, boat, person

### Dataset
https://kth-my.sharepoint.com/:f:/r/personal/ckmc_ug_kth_se/Documents/training_datasets/all_classes_dataset

### Performance

| Metric | Value |
|--------|------|
| mAP@0.5 | 0.918 |
| mAP@0.5:0.95 | 0.762 |

| Class | AP@0.5 | AP@0.5:0.95 |
|------|--------|-------------|
| sam | 0.957 | 0.801 |
| buoy | 0.709 | 0.485 |
| lolo | 0.995 | 0.951 |
| catamaran | 0.995 | 0.792 |
| boat | 0.992 | 0.948 |
| person | 0.863 | 0.593 |

### Recommendation
Use when **full scene understanding (including humans)** is required.

---

## Model Selection Guide

| Use Case | Recommended Model |
|---------|------------------|
| Max accuracy (sam + buoy) | 2cls_fisheye |
| Real + Sim generalization | 2cls_mixed |
| Maritime multi-object detection | 5cls |
| Full semantic scene (incl. people) | 6cls |

---

## Notes

- Increasing number of classes may reduce performance on **sam and buoy**
- Choose model based on:
  - environment (real vs sim)
  - required classes
  - inference constraints

---

## Tip

If your pipeline only needs:
- navigation targets → use **2-class**
- semantic reasoning → use **5/6-class**
