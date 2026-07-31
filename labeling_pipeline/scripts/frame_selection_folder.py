#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import hdbscan
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from torchvision import transforms

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

class PadToSquare:
    """
    Pads a rectangular image to square without stretching it.

    This is better for 1920x1080 images than resizing directly to square DINO inputs,
    because direct resizing changes the aspect ratio.
    """

    def __init__(self, fill=(0, 0, 0)):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size

        if width == height:
            return image

        size = max(width, height)
        padded = Image.new("RGB", (size, size), self.fill)

        left = (size - width) // 2
        top = (size - height) // 2

        padded.paste(image, (left, top))
        return padded

def natural_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]

def load_yaml_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return data

def add_bool_flag(argv, name, value):
    if value:
        argv.append(f"--{name}")

def config_to_argv(cfg: dict):
    argv = []

    paths = cfg.get("paths", {})
    dino = cfg.get("dino", {})
    hdbscan_cfg = cfg.get("hdbscan", {})
    selection = cfg.get("selection", {})
    output = cfg.get("output", {})
    runtime = cfg.get("runtime", {})

    simple_args = {
        "dataset-root": paths.get("dataset_root"),

        "model": dino.get("model"),
        "embedding-mode": dino.get("embedding_mode"),
        "tile-cols": dino.get("tile_cols"),
        "tile-rows": dino.get("tile_rows"),
        "tile-overlap": dino.get("tile_overlap"),
        "global-weight": dino.get("global_weight"),
        "tile-weight": dino.get("tile_weight"),
        "image-size": dino.get("image_size"),

        "pca-dim": hdbscan_cfg.get("pca_dim"),
        "min-cluster-size": hdbscan_cfg.get("min_cluster_size"),
        "min-samples": hdbscan_cfg.get("min_samples"),

        "small-cluster-keep-all": selection.get("small_cluster_keep_all"),
        "min-keep-per-cluster": selection.get("min_keep_per_cluster"),
        "keep-sqrt-factor": selection.get("keep_sqrt_factor"),
        "max-keep-per-cluster": selection.get("max_keep_per_cluster"),

        "mode": output.get("mode"),
        "missing-label-policy": output.get("missing_label_policy"),
        "max-contact-sheet-images": output.get("max_contact_sheet_images"),

        "device": runtime.get("device"),
        "batch-size": runtime.get("batch_size"),
    }

    for key, value in simple_args.items():
        if value is not None:
            argv.extend([f"--{key}", str(value)])

    add_bool_flag(argv, "backup-original", bool(output.get("backup_original", True)))
    add_bool_flag(argv, "copy-compressed-out", bool(output.get("copy_compressed_out", False)))
    add_bool_flag(argv, "reuse-embeddings", bool(runtime.get("reuse_embeddings", False)))

    if output.get("save_reports") is False:
        argv.append("--disable-reports")

    return argv

def list_images(images_root: Path):
    images = [
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_key)

def copy_or_symlink(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_label_path_for_image(image_path: Path, images_root: Path, labels_root: Path):
    rel_path = image_path.relative_to(images_root)
    return labels_root / rel_path.with_suffix(".txt")

def copy_label(label_path: Path, output_label_path: Path, missing_label_policy: str):
    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    if label_path.exists():
        shutil.copy2(label_path, output_label_path)
        return "label_copied"

    if missing_label_policy == "create_empty":
        output_label_path.write_text("")
        return "empty_label_created"

    if missing_label_policy == "skip":
        return "missing_label_skipped"

    if missing_label_policy == "error":
        raise FileNotFoundError(f"Missing label file: {label_path}")

    raise ValueError(f"Unknown missing label policy: {missing_label_policy}")

def load_dino_model(model_name: str, device: str):
    print(f"[INFO] Loading DINOv2 model: {model_name}")
    print("[INFO] First run may download weights using torch.hub")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)

    return model

def build_transform(image_size: int):
    return transforms.Compose([
        PadToSquare(fill=(0, 0, 0)),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

def extract_feature_from_model(model, batch):
    if hasattr(model, "forward_features"):
        output = model.forward_features(batch)

        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                feat = output["x_norm_clstoken"]
            elif "x_prenorm" in output:
                feat = output["x_prenorm"]
                if feat.ndim == 3:
                    feat = feat[:, 0]
            elif "x_norm_patchtokens" in output:
                feat = output["x_norm_patchtokens"].mean(dim=1)
            else:
                tensor_values = [v for v in output.values() if torch.is_tensor(v)]
                if not tensor_values:
                    raise RuntimeError("Could not find tensor feature in model output.")
                feat = tensor_values[0]
                if feat.ndim == 3:
                    feat = feat[:, 0]
        else:
            feat = output
    else:
        feat = model(batch)

    if feat.ndim == 3:
        feat = feat[:, 0]

    return feat

def extract_features_for_tensors(model, tensors, device, batch_size):
    all_features = []

    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch_tensors = tensors[start:start + batch_size]
            batch = torch.stack(batch_tensors, dim=0).to(device)

            features = extract_feature_from_model(model, batch)
            features = features.detach().cpu().numpy().astype(np.float32)

            all_features.append(features)

    features = np.concatenate(all_features, axis=0)
    features = normalize(features, norm="l2", axis=1)

    return features

def generate_tile_boxes(width, height, tile_cols, tile_rows, tile_overlap):
    boxes = []

    cell_w = width / tile_cols
    cell_h = height / tile_rows

    for gy in range(tile_rows):
        for gx in range(tile_cols):
            x0 = int(round(gx * cell_w))
            y0 = int(round(gy * cell_h))
            x1 = int(round((gx + 1) * cell_w))
            y1 = int(round((gy + 1) * cell_h))

            pad_x = int(round((x1 - x0) * tile_overlap))
            pad_y = int(round((y1 - y0) * tile_overlap))

            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(width, x1 + pad_x)
            y1 = min(height, y1 + pad_y)

            boxes.append((x0, y0, x1, y1))

    return boxes

def compute_one_image_embedding(
    image_path,
    model,
    transform,
    device,
    batch_size,
    embedding_mode,
    tile_cols,
    tile_rows,
    tile_overlap,
    global_weight,
    tile_weight,
):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    tensors = []

    if embedding_mode in ["global", "global_tiles"]:
        tensors.append(transform(pil_image))

    if embedding_mode in ["tiles", "global_tiles"]:
        tile_boxes = generate_tile_boxes(
            width=width,
            height=height,
            tile_cols=tile_cols,
            tile_rows=tile_rows,
            tile_overlap=tile_overlap,
        )

        for box in tile_boxes:
            tile = pil_image.crop(box)
            tensors.append(transform(tile))

    features = extract_features_for_tensors(
        model=model,
        tensors=tensors,
        device=device,
        batch_size=batch_size,
    )

    parts = []
    current = 0

    if embedding_mode in ["global", "global_tiles"]:
        global_feat = features[current]
        current += 1
        parts.append(global_weight * global_feat)

    if embedding_mode in ["tiles", "global_tiles"]:
        tile_features = features[current:]
        num_tiles = tile_features.shape[0]
        tile_block = tile_features.reshape(-1)
        tile_block = (tile_weight / math.sqrt(num_tiles)) * tile_block
        parts.append(tile_block)

    final_embedding = np.concatenate(parts, axis=0).astype(np.float32)
    final_embedding = normalize(final_embedding.reshape(1, -1), norm="l2", axis=1)[0]

    return final_embedding

def build_cache_metadata(args):
    return {
        "model": args.model,
        "embedding_mode": args.embedding_mode,
        "image_size": args.image_size,
        "tile_cols": args.tile_cols,
        "tile_rows": args.tile_rows,
        "tile_overlap": args.tile_overlap,
        "global_weight": args.global_weight,
        "tile_weight": args.tile_weight,
    }

def cache_matches(data, image_paths, metadata):
    cached_paths = list(data["paths"])
    current_paths = [str(p) for p in image_paths]

    if cached_paths != current_paths:
        return False

    if "metadata_json" not in data:
        return False

    cached_metadata = json.loads(str(data["metadata_json"].item()))
    return cached_metadata == metadata

def compute_dino_embeddings(image_paths, model, transform, device, batch_size, cache_path, reuse_cache, args):
    metadata = build_cache_metadata(args)

    if reuse_cache and cache_path is not None and cache_path.exists():
        print(f"[INFO] Loading cached embeddings: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        if cache_matches(data, image_paths, metadata):
            embeddings = data["embeddings"]
            print("[INFO] Cache matches current settings. Reusing embeddings.")
            return embeddings

        print("[WARN] Cache exists but image list/settings changed. Recomputing embeddings.")

    all_embeddings = []

    for image_path in tqdm(image_paths, desc="Extracting DINO global/tile embeddings"):
        emb = compute_one_image_embedding(
            image_path=image_path,
            model=model,
            transform=transform,
            device=device,
            batch_size=batch_size,
            embedding_mode=args.embedding_mode,
            tile_cols=args.tile_cols,
            tile_rows=args.tile_rows,
            tile_overlap=args.tile_overlap,
            global_weight=args.global_weight,
            tile_weight=args.tile_weight,
        )
        all_embeddings.append(emb)

    embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
    embeddings = normalize(embeddings, norm="l2", axis=1)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            paths=np.array([str(p) for p in image_paths]),
            embeddings=embeddings,
            metadata_json=np.array(json.dumps(metadata)),
        )
        print(f"[INFO] Saved embeddings cache: {cache_path}")

    return embeddings

def reduce_embeddings_for_hdbscan(embeddings, pca_dim):
    n_samples, n_features = embeddings.shape

    if pca_dim <= 0:
        print("[INFO] PCA disabled")
        return embeddings

    max_dim = min(pca_dim, n_samples - 1, n_features)

    if max_dim < 2:
        print("[WARN] Not enough samples for PCA. Using original embeddings.")
        return embeddings

    print(f"[INFO] Reducing embeddings with PCA: {n_features} -> {max_dim}")

    pca = PCA(n_components=max_dim, random_state=0)
    reduced = pca.fit_transform(embeddings)
    reduced = normalize(reduced, norm="l2", axis=1)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] PCA explained variance: {explained:.3f}")

    return reduced.astype(np.float32)

def run_hdbscan(features, min_cluster_size, min_samples):
    print("[INFO] Running HDBSCAN")
    print(f"[INFO] min_cluster_size: {min_cluster_size}")
    print(f"[INFO] min_samples: {min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(features)

    probabilities = getattr(clusterer, "probabilities_", np.zeros(len(labels)))
    outlier_scores = getattr(clusterer, "outlier_scores_", np.zeros(len(labels)))

    return labels, probabilities, outlier_scores

def select_from_cluster(sorted_indices, args):
    n = len(sorted_indices)

    if n <= args.small_cluster_keep_all:
        return sorted_indices

    keep_n = int(math.ceil(math.sqrt(n) * args.keep_sqrt_factor))
    keep_n = max(args.min_keep_per_cluster, keep_n)
    keep_n = min(keep_n, n)

    if args.max_keep_per_cluster > 0:
        keep_n = min(keep_n, args.max_keep_per_cluster)

    if keep_n >= n:
        return sorted_indices

    positions = np.linspace(0, n - 1, keep_n)
    positions = np.round(positions).astype(int)
    positions = sorted(set(positions.tolist()))

    return [sorted_indices[pos] for pos in positions]

def select_indices_with_hdbscan(embeddings, args):
    cluster_features = reduce_embeddings_for_hdbscan(embeddings, pca_dim=args.pca_dim)

    labels, probabilities, outlier_scores = run_hdbscan(
        cluster_features,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    unique_labels = sorted(set(labels.tolist()))
    cluster_labels = [x for x in unique_labels if x >= 0]
    noise_count = int(np.sum(labels == -1))

    print()
    print("[INFO] HDBSCAN result")
    print(f"  clusters found: {len(cluster_labels)}")
    print(f"  noise/unique frames: {noise_count}")

    selected_indices = set()
    reasons = {}

    for idx in np.where(labels == -1)[0].tolist():
        selected_indices.add(idx)
        reasons[idx] = "unique_noise_kept"

    for cluster_label in cluster_labels:
        cluster_indices = np.where(labels == cluster_label)[0].tolist()
        cluster_indices = sorted(cluster_indices)

        selected_from_cluster = select_from_cluster(cluster_indices, args)

        for idx in selected_from_cluster:
            selected_indices.add(idx)
            reasons[idx] = "cluster_representative_kept"

        for idx in cluster_indices:
            if idx not in selected_from_cluster:
                reasons[idx] = "compressed_from_dense_cluster"

    return selected_indices, reasons, labels, probabilities, outlier_scores, cluster_labels

def make_contact_sheet(records, out_path: Path, selected_value=True, max_images=120):
    chosen = [r for r in records if r["selected"] == selected_value]

    if len(chosen) == 0:
        return

    if len(chosen) > max_images:
        indices = np.linspace(0, len(chosen) - 1, max_images).astype(int)
        chosen = [chosen[i] for i in indices]

    thumb_w = 180
    thumb_h = 120
    label_h = 42
    cols = 5
    rows = int(np.ceil(len(chosen) / cols))

    sheet = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 255, dtype=np.uint8)

    for idx, record in enumerate(chosen):
        path = Path(record["source_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(thumb_w / w, thumb_h / h)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        row = idx // cols
        col = idx % cols

        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        x_img = x0 + (thumb_w - new_w) // 2
        y_img = y0 + (thumb_h - new_h) // 2

        sheet[y_img:y_img + new_h, x_img:x_img + new_w] = resized

        label1 = f"i:{record['frame_index']} c:{record['cluster_label']}"
        label2 = record["reason"][:24]
        label3 = str(record["relative_path"])[:26]

        cv2.putText(sheet, label1, (x0 + 5, y0 + thumb_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(sheet, label2, (x0 + 5, y0 + thumb_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(sheet, label3, (x0 + 5, y0 + thumb_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)

def make_cluster_contact_sheets(records, reports_dir: Path, max_clusters=20, max_images_per_cluster=80):
    cluster_dir = reports_dir / "cluster_contact_sheets"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    clustered = df[df["cluster_label"] >= 0]

    if clustered.empty:
        return

    cluster_sizes = clustered.groupby("cluster_label").size().sort_values(ascending=False)

    for cluster_label, _ in cluster_sizes.head(max_clusters).items():
        cluster_records = [r for r in records if r["cluster_label"] == cluster_label]
        selected_records = [r for r in cluster_records if r["selected"]]
        rejected_records = [r for r in cluster_records if not r["selected"]]

        make_contact_sheet(
            selected_records,
            cluster_dir / f"cluster_{cluster_label:04d}_selected.jpg",
            selected_value=True,
            max_images=max_images_per_cluster,
        )

        if rejected_records:
            make_contact_sheet(
                rejected_records,
                cluster_dir / f"cluster_{cluster_label:04d}_compressed_out.jpg",
                selected_value=False,
                max_images=max_images_per_cluster,
            )

def replace_dataset_in_place(dataset_root: Path, tmp_root: Path, backup_original: bool):
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    tmp_images_root = tmp_root / "images"
    tmp_labels_root = tmp_root / "labels"

    if not tmp_images_root.exists():
        raise RuntimeError(f"Temporary images folder does not exist: {tmp_images_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dataset_root / f".frame_selection_backup_{timestamp}"

    if backup_original:
        backup_root.mkdir(parents=True, exist_ok=True)

        if images_root.exists():
            shutil.move(str(images_root), str(backup_root / "images"))

        if labels_root.exists():
            shutil.move(str(labels_root), str(backup_root / "labels"))

        print(f"[INFO] Original images/labels backed up to: {backup_root}")
    else:
        if images_root.exists():
            shutil.rmtree(images_root)

        if labels_root.exists():
            shutil.rmtree(labels_root)

    shutil.move(str(tmp_images_root), str(images_root))

    if tmp_labels_root.exists():
        shutil.move(str(tmp_labels_root), str(labels_root))
    else:
        labels_root.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[INFO] Dataset updated in place: {dataset_root}")

def build_records_and_selected_dataset(
    image_paths,
    images_root,
    labels_root,
    tmp_root,
    compressed_dataset_root,
    selected_indices,
    reasons,
    labels,
    probabilities,
    outlier_scores,
    args,
):
    tmp_images_root = tmp_root / "images"
    tmp_labels_root = tmp_root / "labels"
    compressed_images_root = compressed_dataset_root / "images"
    compressed_labels_root = compressed_dataset_root / "labels"

    records = []

    selected_count = 0
    compressed_count = 0
    skipped_missing_label_count = 0
    empty_labels_created_count = 0

    for idx, image_path in enumerate(image_paths):
        selected = idx in selected_indices
        cluster_label = int(labels[idx])
        reason = reasons.get(idx, "unknown")

        rel_path = image_path.relative_to(images_root)
        source_label_path = get_label_path_for_image(image_path, images_root, labels_root)

        output_image_path = ""
        output_label_path = ""
        label_status = ""

        if selected:
            if not source_label_path.exists() and args.missing_label_policy == "skip":
                selected = False
                reason = "skipped_missing_label"
                skipped_missing_label_count += 1
            else:
                selected_count += 1

                dst_image_path = tmp_images_root / rel_path
                dst_label_path = tmp_labels_root / rel_path.with_suffix(".txt")

                copy_or_symlink(image_path, dst_image_path, args.mode)

                label_status = copy_label(
                    label_path=source_label_path,
                    output_label_path=dst_label_path,
                    missing_label_policy=args.missing_label_policy,
                )

                if label_status == "empty_label_created":
                    empty_labels_created_count += 1

                output_image_path = str(dst_image_path)
                output_label_path = str(dst_label_path)

        if not selected:
            compressed_count += 1

            if args.copy_compressed_out and reason != "skipped_missing_label":
                dst_image_path = compressed_images_root / rel_path
                dst_label_path = compressed_labels_root / rel_path.with_suffix(".txt")

                copy_or_symlink(image_path, dst_image_path, args.mode)

                label_status = copy_label(
                    label_path=source_label_path,
                    output_label_path=dst_label_path,
                    missing_label_policy="create_empty",
                )

                output_image_path = str(dst_image_path)
                output_label_path = str(dst_label_path)

        records.append({
            "frame_index": idx,
            "relative_path": str(rel_path),
            "source_path": str(image_path),
            "source_label_path": str(source_label_path),
            "output_image_path": output_image_path,
            "output_label_path": output_label_path,
            "selected": bool(selected),
            "reason": reason,
            "label_exists": bool(source_label_path.exists()),
            "label_status": label_status,
            "cluster_label": cluster_label,
            "cluster_probability": float(probabilities[idx]),
            "outlier_score": float(outlier_scores[idx]),
        })

    counts = {
        "selected_count": selected_count,
        "compressed_count": compressed_count,
        "skipped_missing_label_count": skipped_missing_label_count,
        "empty_labels_created_count": empty_labels_created_count,
    }

    return records, counts

def write_reports(records, cluster_labels, reports_dir: Path, args):
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "dino_hdbscan_selection_report.csv"
    df = pd.DataFrame(records)
    df.to_csv(report_path, index=False)

    summary_rows = []
    for cluster_label in cluster_labels:
        cluster_df = df[df["cluster_label"] == cluster_label]
        summary_rows.append({
            "cluster_label": int(cluster_label),
            "total_images": int(len(cluster_df)),
            "selected_images": int(cluster_df["selected"].sum()),
            "compressed_out_images": int((~cluster_df["selected"]).sum()),
            "mean_probability": float(cluster_df["cluster_probability"].mean()),
        })

    summary_path = reports_dir / "cluster_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    make_contact_sheet(records, reports_dir / "selected_contact_sheet.jpg", selected_value=True, max_images=args.max_contact_sheet_images)
    make_contact_sheet(records, reports_dir / "compressed_out_contact_sheet.jpg", selected_value=False, max_images=args.max_contact_sheet_images)

    make_cluster_contact_sheets(records, reports_dir=reports_dir, max_clusters=20, max_images_per_cluster=80)

    return report_path, summary_path

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Curate an existing dataset_to_label folder in place using "
            "DINOv2 global/tile embeddings + HDBSCAN."
        )
    )

    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--dataset-root", required=False, type=Path, default=None, help="Dataset root containing images/ and labels/.")
    parser.add_argument("--model", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"], help="DINOv2 model.")
    parser.add_argument("--embedding-mode", default="global_tiles", choices=["global", "tiles", "global_tiles"], help="Embedding mode.")
    parser.add_argument("--tile-cols", type=int, default=4, help="Number of tile columns.")
    parser.add_argument("--tile-rows", type=int, default=3, help="Number of tile rows.")
    parser.add_argument("--tile-grid", type=int, default=None, help="Backward-compatible square grid override.")
    parser.add_argument("--tile-overlap", type=float, default=0.15, help="Tile overlap ratio.")
    parser.add_argument("--global-weight", type=float, default=1.0, help="Global embedding weight.")
    parser.add_argument("--tile-weight", type=float, default=1.0, help="Tile embedding block weight.")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=16, help="DINO crop/tile batch size.")
    parser.add_argument("--image-size", type=int, default=518, help="DINO input size.")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before HDBSCAN. 0 disables PCA.")
    parser.add_argument("--min-cluster-size", type=int, default=8, help="Minimum dense repeated group size.")
    parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN conservativeness.")
    parser.add_argument("--small-cluster-keep-all", type=int, default=5, help="Keep all images in clusters up to this size.")
    parser.add_argument("--min-keep-per-cluster", type=int, default=3, help="Minimum representatives per cluster.")
    parser.add_argument("--keep-sqrt-factor", type=float, default=1.0, help="Representatives per cluster factor.")
    parser.add_argument("--max-keep-per-cluster", type=int, default=0, help="Max representatives per cluster. 0 means no cap.")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="Copy selected images or create symlinks in temp dataset.")
    parser.add_argument("--missing-label-policy", choices=["create_empty", "skip", "error"], default="create_empty", help="What to do when matching label file is missing.")
    parser.add_argument("--backup-original", action="store_true", help="Backup original images/labels before in-place replacement.")
    parser.add_argument("--copy-compressed-out", action="store_true", help="Also copy removed images/labels for inspection.")
    parser.add_argument("--reuse-embeddings", action="store_true", help="Reuse cached DINO embeddings if image list/settings are unchanged.")
    parser.add_argument("--disable-reports", action="store_true", help="Disable CSV reports and contact sheets.")
    parser.add_argument("--max-contact-sheet-images", type=int, default=120, help="Max images in contact sheets.")

    return parser

def main():
    parser = build_parser()

    pre_args, remaining_argv = parser.parse_known_args()

    if pre_args.config is not None:
        cfg = load_yaml_config(pre_args.config)
        config_argv = config_to_argv(cfg)
        args = parser.parse_args(config_argv + remaining_argv)
    else:
        args = parser.parse_args()

    if args.tile_grid is not None:
        args.tile_cols = args.tile_grid
        args.tile_rows = args.tile_grid

    if args.dataset_root is None:
        raise ValueError("Missing paths.dataset_root in YAML or --dataset-root on CLI.")

    if args.tile_cols < 1:
        raise ValueError("tile_cols must be >= 1")

    if args.tile_rows < 1:
        raise ValueError("tile_rows must be >= 1")

    if args.tile_overlap < 0.0:
        raise ValueError("tile_overlap must be >= 0.0")

    if args.mode == "symlink":
        raise ValueError("mode: symlink is not allowed for in-place curation. Use mode: copy.")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    dataset_root = args.dataset_root.resolve()
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"Images folder does not exist: {images_root}")

    labels_root.mkdir(parents=True, exist_ok=True)

    reports_enabled = not args.disable_reports

    if reports_enabled:
        reports_dir = dataset_root / "reports" / "folder_selection"
        cache_dir = reports_dir
        compressed_dataset_root = reports_dir / "compressed_out_dataset"
    else:
        reports_dir = None
        cache_dir = dataset_root / ".frame_selection_cache" / "folder_selection"
        compressed_dataset_root = dataset_root / "compressed_out_dataset"

    cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = dataset_root / ".frame_selection_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    image_paths = list_images(images_root)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {images_root}")

    cache_name = (
        f"dino_cache_{args.model}_"
        f"{args.embedding_mode}_"
        f"cols{args.tile_cols}_rows{args.tile_rows}_"
        f"img{args.image_size}.npz"
    )
    cache_path = cache_dir / cache_name

    print(f"[INFO] Dataset root:     {dataset_root}")
    print(f"[INFO] Images root:      {images_root}")
    print(f"[INFO] Labels root:      {labels_root}")
    print(f"[INFO] Reports enabled:  {reports_enabled}")
    print(f"[INFO] Cache folder:     {cache_dir}")
    print(f"[INFO] Found images:     {len(image_paths)}")
    print(f"[INFO] Embedding mode:   {args.embedding_mode}")
    print(f"[INFO] Tile layout:      {args.tile_cols} cols x {args.tile_rows} rows")
    print(f"[INFO] Tile overlap:     {args.tile_overlap}")
    print(f"[INFO] DINO image size:  {args.image_size}")
    print(f"[INFO] Model:            {args.model}")

    label_exists_count = 0
    missing_label_count = 0

    for image_path in image_paths:
        label_path = get_label_path_for_image(image_path, images_root, labels_root)
        if label_path.exists():
            label_exists_count += 1
        else:
            missing_label_count += 1

    print(f"[INFO] Matching label files found: {label_exists_count}")
    print(f"[INFO] Missing label files:        {missing_label_count}")
    print(f"[INFO] Missing label policy:       {args.missing_label_policy}")

    if missing_label_count > 0 and args.missing_label_policy == "error":
        raise FileNotFoundError(
            f"{missing_label_count} images are missing labels. "
            f"Use missing_label_policy: create_empty if these are negative images."
        )

    transform = build_transform(args.image_size)
    model = load_dino_model(args.model, args.device)

    embeddings = compute_dino_embeddings(
        image_paths=image_paths,
        model=model,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
        cache_path=cache_path,
        reuse_cache=args.reuse_embeddings,
        args=args,
    )

    print(f"[INFO] Final embedding shape: {embeddings.shape}")

    selected_indices, reasons, labels, probabilities, outlier_scores, cluster_labels = select_indices_with_hdbscan(
        embeddings=embeddings,
        args=args,
    )

    records, counts = build_records_and_selected_dataset(
        image_paths=image_paths,
        images_root=images_root,
        labels_root=labels_root,
        tmp_root=tmp_root,
        compressed_dataset_root=compressed_dataset_root,
        selected_indices=selected_indices,
        reasons=reasons,
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
        args=args,
    )

    report_path = None
    summary_path = None

    if reports_enabled:
        report_path, summary_path = write_reports(records, cluster_labels, reports_dir, args)
    else:
        print("[INFO] Reports disabled. CSV reports and contact sheets were not written.")

    replace_dataset_in_place(
        dataset_root=dataset_root,
        tmp_root=tmp_root,
        backup_original=args.backup_original,
    )

    print()
    print("[DONE]")
    print(f"Total images before:        {len(image_paths)}")
    print(f"Selected images after:      {counts['selected_count']}")
    print(f"Compressed/skipped images:  {counts['compressed_count']}")
    print(f"Empty labels created:       {counts['empty_labels_created_count']}")
    print(f"Skipped missing labels:     {counts['skipped_missing_label_count']}")
    print(f"Updated dataset root:       {dataset_root}")
    print(f"Updated images:             {dataset_root / 'images'}")
    print(f"Updated labels:             {dataset_root / 'labels'}")
    print(f"Embedding cache:            {cache_path}")

    if reports_enabled:
        print(f"Report:                     {report_path}")
        print(f"Cluster summary:            {summary_path}")
        print(f"Selected sheet:             {reports_dir / 'selected_contact_sheet.jpg'}")
        print(f"Compressed-out sheet:       {reports_dir / 'compressed_out_contact_sheet.jpg'}")
        print(f"Cluster sheets folder:      {reports_dir / 'cluster_contact_sheets'}")
    else:
        print("Reports:                    disabled")

    if args.copy_compressed_out:
        print(f"Compressed-out dataset:     {compressed_dataset_root}")

if __name__ == "__main__":
    main()
