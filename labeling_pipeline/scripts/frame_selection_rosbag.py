#!/usr/bin/env python3
import argparse
import copy
import json
import math
import re
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

import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DEFAULT_TOPIC = "/M350/gimbal_camera/camera/image_raw"

SUPPORTED_IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
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

def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("_")
    return name if name else "bag"

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
    """
    Convert YAML config into CLI args. CLI values after --config still override YAML.
    """
    argv = []

    rosbag_cfg = cfg.get("rosbag", {})
    paths = cfg.get("paths", {})
    dino = cfg.get("dino", {})
    hdbscan_cfg = cfg.get("hdbscan", {})
    selection = cfg.get("selection", {})
    output = cfg.get("output", {})
    runtime = cfg.get("runtime", {})

    simple_args = {
        "bags-root": rosbag_cfg.get("bags_root"),
        "topic": rosbag_cfg.get("topic"),
        "storage-id": rosbag_cfg.get("storage_id"),
        "every-n": rosbag_cfg.get("every_n"),
        "start-sec": rosbag_cfg.get("start_sec"),
        "end-sec": rosbag_cfg.get("end_sec"),
        "max-candidates": rosbag_cfg.get("max_candidates"),

        "output": paths.get("output"),

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

        "image-format": output.get("image_format"),
        "jpg-quality": output.get("jpg_quality"),
        "max-contact-sheet-images": output.get("max_contact_sheet_images"),

        "device": runtime.get("device"),
        "batch-size": runtime.get("batch_size"),
    }

    for key, value in simple_args.items():
        if value is not None:
            argv.extend([f"--{key}", str(value)])

    add_bool_flag(argv, "create-empty-labels", bool(output.get("create_empty_labels", False)))
    add_bool_flag(argv, "copy-compressed-out", bool(output.get("copy_compressed_out", False)))
    add_bool_flag(argv, "reuse-embeddings", bool(runtime.get("reuse_embeddings", False)))

    if output.get("save_reports") is False:
        argv.append("--disable-reports")

    return argv

def is_rosbag_folder(path: Path):
    if not path.is_dir():
        return False

    if (path / "metadata.yaml").exists():
        return True

    if any(path.glob("*.db3")):
        return True

    if any(path.glob("*.mcap")):
        return True

    return False

def discover_bag_folders(bags_root: Path):
    """
    Discover rosbag folders from bags_root.

    If bags_root itself is a rosbag folder, one bag is returned.
    Otherwise, each direct child that looks like a rosbag folder is returned.
    """
    bags_root = bags_root.resolve()

    if not bags_root.exists():
        raise FileNotFoundError(f"bags_root does not exist: {bags_root}")

    if is_rosbag_folder(bags_root):
        bags = [bags_root]
    else:
        bags = [p for p in bags_root.iterdir() if is_rosbag_folder(p)]

    bags = sorted(bags, key=natural_key)

    if not bags:
        raise RuntimeError(f"No ROS2 bag folders found inside: {bags_root}")

    return bags

def open_bag_reader(bag_path: Path, storage_id: str):
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    return reader

def get_topic_type(reader, topic_name: str):
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topics_and_types}

    if topic_name not in topic_type_map:
        print()
        print(f"[ERROR] Topic not found in bag: {topic_name}")
        print()
        print("[INFO] Available topics:")
        for name, typ in sorted(topic_type_map.items()):
            print(f"  {name}  [{typ}]")
        raise RuntimeError(f"Topic not found: {topic_name}")

    topic_type = topic_type_map[topic_name]

    if topic_type not in SUPPORTED_IMAGE_TYPES:
        raise RuntimeError(
            f"Topic {topic_name} has type {topic_type}, but this script only supports: "
            f"{sorted(SUPPORTED_IMAGE_TYPES)}"
        )

    return topic_type

def image_msg_to_cv2(msg, topic_type: str, bridge: CvBridge):
    if topic_type == "sensor_msgs/msg/Image":
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    if topic_type == "sensor_msgs/msg/CompressedImage":
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise RuntimeError("Could not decode compressed image.")

        return image_bgr

    raise RuntimeError(f"Unsupported image type: {topic_type}")

def cv2_bgr_to_pil_rgb(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def get_header_stamp_ns(msg):
    if not hasattr(msg, "header"):
        return None

    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

def should_keep_by_time(bag_time_ns, first_topic_time_ns, start_sec, end_sec):
    relative_sec = (bag_time_ns - first_topic_time_ns) / 1e9

    if start_sec is not None and relative_sec < start_sec:
        return False

    if end_sec is not None and relative_sec > end_sec:
        return False

    return True

def save_image(image_bgr, output_path: Path, image_format: str, jpg_quality: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_format.lower() in ["jpg", "jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    elif image_format.lower() == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = []

    ok = cv2.imwrite(str(output_path), image_bgr, params)

    if not ok:
        raise RuntimeError(f"Could not write image: {output_path}")

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
    pil_image,
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
        "bag": str(args.bag.resolve()),
        "topic": args.topic,
        "storage_id": args.storage_id,
        "every_n": args.every_n,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "max_candidates": args.max_candidates,
        "model": args.model,
        "embedding_mode": args.embedding_mode,
        "image_size": args.image_size,
        "tile_cols": args.tile_cols,
        "tile_rows": args.tile_rows,
        "tile_overlap": args.tile_overlap,
        "global_weight": args.global_weight,
        "tile_weight": args.tile_weight,
    }

def cache_matches(data, metadata):
    if "metadata_json" not in data:
        return False

    cached_metadata = json.loads(str(data["metadata_json"].item()))
    return cached_metadata == metadata

def first_pass_compute_embeddings(args, model, transform, cache_path):
    metadata = build_cache_metadata(args)

    if args.reuse_embeddings and cache_path.exists():
        print(f"[INFO] Loading cached embeddings: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        if cache_matches(data, metadata):
            embeddings = data["embeddings"]
            frame_rows = json.loads(str(data["frame_rows_json"].item()))
            print("[INFO] Cache matches current bag/settings. Reusing embeddings.")
            return embeddings, frame_rows

        print("[WARN] Cache exists but bag/settings changed. Recomputing embeddings.")

    reader = open_bag_reader(args.bag.resolve(), args.storage_id)
    topic_type = get_topic_type(reader, args.topic)
    msg_type = get_message(topic_type)

    bridge = CvBridge()

    topic_msg_count = 0
    candidate_count = 0
    skipped_stride_count = 0
    skipped_time_count = 0
    failed_count = 0
    first_topic_time_ns = None

    embeddings = []
    frame_rows = []

    print("[INFO] First pass: reading bag and computing embeddings")

    pbar = tqdm(desc="Candidate frames embedded")

    while reader.has_next():
        topic_name, data, bag_time_ns = reader.read_next()

        if topic_name != args.topic:
            continue

        if first_topic_time_ns is None:
            first_topic_time_ns = bag_time_ns

        topic_msg_count += 1

        if not should_keep_by_time(
            bag_time_ns=bag_time_ns,
            first_topic_time_ns=first_topic_time_ns,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        ):
            skipped_time_count += 1
            continue

        if (topic_msg_count - 1) % args.every_n != 0:
            skipped_stride_count += 1
            continue

        if args.max_candidates > 0 and candidate_count >= args.max_candidates:
            break

        try:
            msg = deserialize_message(data, msg_type)
            image_bgr = image_msg_to_cv2(msg, topic_type, bridge)
            pil_image = cv2_bgr_to_pil_rgb(image_bgr)

            emb = compute_one_image_embedding(
                pil_image=pil_image,
                model=model,
                transform=transform,
                device=args.device,
                batch_size=args.batch_size,
                embedding_mode=args.embedding_mode,
                tile_cols=args.tile_cols,
                tile_rows=args.tile_rows,
                tile_overlap=args.tile_overlap,
                global_weight=args.global_weight,
                tile_weight=args.tile_weight,
            )

            header_stamp_ns = get_header_stamp_ns(msg)
            relative_sec = (bag_time_ns - first_topic_time_ns) / 1e9

            frame_rows.append({
                "candidate_index": candidate_count,
                "topic_message_index": topic_msg_count - 1,
                "bag_time_ns": int(bag_time_ns),
                "header_stamp_ns": header_stamp_ns,
                "relative_sec": relative_sec,
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
            })

            embeddings.append(emb)

            candidate_count += 1
            pbar.update(1)

        except Exception as exc:
            failed_count += 1
            print(f"[WARN] Failed to process topic message {topic_msg_count - 1}: {exc}")

    pbar.close()

    if len(embeddings) == 0:
        raise RuntimeError("No candidate frames were embedded from the bag.")

    embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    embeddings = normalize(embeddings, norm="l2", axis=1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        metadata_json=np.array(json.dumps(metadata)),
        frame_rows_json=np.array(json.dumps(frame_rows)),
    )

    print()
    print("[INFO] First pass summary")
    print(f"  topic messages seen:       {topic_msg_count}")
    print(f"  candidate frames embedded: {candidate_count}")
    print(f"  skipped by time window:    {skipped_time_count}")
    print(f"  skipped by every-n:        {skipped_stride_count}")
    print(f"  failed messages:           {failed_count}")
    print(f"  embeddings cache:          {cache_path}")

    return embeddings, frame_rows

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

def select_indices_with_hdbscan(embeddings, frame_rows, args):
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
    print(f"  clusters found:       {len(cluster_labels)}")
    print(f"  noise/unique frames:  {noise_count}")

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

    records = []

    for idx, row in enumerate(frame_rows):
        selected = idx in selected_indices

        records.append({
            **row,
            "selected": bool(selected),
            "reason": reasons.get(idx, "unknown"),
            "cluster_label": int(labels[idx]),
            "cluster_probability": float(probabilities[idx]),
            "outlier_score": float(outlier_scores[idx]),
            "output_image_path": "",
            "output_label_path": "",
            "compressed_image_path": "",
        })

    return selected_indices, records, labels

def second_pass_save_selected(args, records, selected_indices, output_images_dir, output_labels_dir, compressed_images_dir):
    selected_by_candidate = set(selected_indices)

    reader = open_bag_reader(args.bag.resolve(), args.storage_id)
    topic_type = get_topic_type(reader, args.topic)
    msg_type = get_message(topic_type)

    bridge = CvBridge()

    topic_msg_count = 0
    candidate_count = 0
    saved_count = 0
    compressed_saved_count = 0
    first_topic_time_ns = None

    record_by_candidate = {int(r["candidate_index"]): r for r in records}

    print("[INFO] Second pass: saving selected frames")

    pbar = tqdm(total=len(selected_by_candidate), desc="Selected frames saved")

    while reader.has_next():
        topic_name, data, bag_time_ns = reader.read_next()

        if topic_name != args.topic:
            continue

        if first_topic_time_ns is None:
            first_topic_time_ns = bag_time_ns

        topic_msg_count += 1

        if not should_keep_by_time(
            bag_time_ns=bag_time_ns,
            first_topic_time_ns=first_topic_time_ns,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        ):
            continue

        if (topic_msg_count - 1) % args.every_n != 0:
            continue

        if args.max_candidates > 0 and candidate_count >= args.max_candidates:
            break

        should_save_selected = candidate_count in selected_by_candidate
        should_save_compressed = args.copy_compressed_out and candidate_count not in selected_by_candidate

        if should_save_selected or should_save_compressed:
            msg = deserialize_message(data, msg_type)
            image_bgr = image_msg_to_cv2(msg, topic_type, bridge)

            file_stem = f"frame_{candidate_count:06d}"
            image_name = f"{file_stem}.{args.image_format}"
            label_name = f"{file_stem}.txt"

            record = record_by_candidate[candidate_count]

            if should_save_selected:
                image_path = output_images_dir / image_name
                label_path = output_labels_dir / label_name

                save_image(
                    image_bgr=image_bgr,
                    output_path=image_path,
                    image_format=args.image_format,
                    jpg_quality=args.jpg_quality,
                )

                if args.create_empty_labels:
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("")

                record["output_image_path"] = str(image_path)
                record["output_label_path"] = str(label_path) if args.create_empty_labels else ""

                saved_count += 1
                pbar.update(1)

            elif should_save_compressed:
                compressed_path = compressed_images_dir / image_name

                save_image(
                    image_bgr=image_bgr,
                    output_path=compressed_path,
                    image_format=args.image_format,
                    jpg_quality=args.jpg_quality,
                )

                record["compressed_image_path"] = str(compressed_path)
                compressed_saved_count += 1

        candidate_count += 1

    pbar.close()

    return saved_count, compressed_saved_count

def make_contact_sheet(records, out_path: Path, selected_value=True, max_images=120):
    if selected_value:
        chosen = [r for r in records if r["selected"] and r.get("output_image_path")]
        path_key = "output_image_path"
    else:
        chosen = [r for r in records if not r["selected"] and r.get("compressed_image_path")]
        path_key = "compressed_image_path"

    if len(chosen) == 0:
        return

    if len(chosen) > max_images:
        indices = np.linspace(0, len(chosen) - 1, max_images).astype(int)
        chosen = [chosen[i] for i in indices]

    thumb_w = 180
    thumb_h = 120
    label_h = 36
    cols = 5
    rows = int(np.ceil(len(chosen) / cols))

    sheet = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 255, dtype=np.uint8)

    for idx, record in enumerate(chosen):
        path = Path(record[path_key])
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

        label1 = f"cand:{record['candidate_index']} c:{record['cluster_label']}"
        label2 = record["reason"][:24]

        cv2.putText(sheet, label1, (x0 + 5, y0 + thumb_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(sheet, label2, (x0 + 5, y0 + thumb_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)

def write_reports(records, reports_dir: Path):
    report_path = reports_dir / "rosbag_dino_hdbscan_selection_report.csv"

    df = pd.DataFrame(records)
    df.to_csv(report_path, index=False)

    cluster_df = df[df["cluster_label"] >= 0]
    summary_rows = []

    if not cluster_df.empty:
        for cluster_label, group in cluster_df.groupby("cluster_label"):
            summary_rows.append({
                "cluster_label": int(cluster_label),
                "total_images": int(len(group)),
                "selected_images": int(group["selected"].sum()),
                "compressed_out_images": int((~group["selected"]).sum()),
                "mean_probability": float(group["cluster_probability"].mean()),
            })

    summary_path = reports_dir / "cluster_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return report_path, summary_path

def run_one_bag(args, model, transform, bag_index: int, total_bags: int):
    bag_path = args.bag.resolve()
    sequence_name = sanitize_name(bag_path.name)

    output_root = args.output.resolve()
    output_images_dir = output_root / "images" / sequence_name
    output_labels_dir = output_root / "labels" / sequence_name

    reports_enabled = not args.disable_reports

    if reports_enabled:
        reports_dir = output_root / "reports" / sequence_name
        cache_dir = reports_dir
        compressed_images_dir = reports_dir / "compressed_out_images"
    else:
        reports_dir = None
        cache_dir = output_root / ".frame_selection_cache" / sequence_name
        compressed_images_dir = output_root / "compressed_out_images" / sequence_name

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if reports_enabled:
        reports_dir.mkdir(parents=True, exist_ok=True)

    cache_name = (
        f"dino_cache_{args.model}_"
        f"{args.embedding_mode}_"
        f"cols{args.tile_cols}_rows{args.tile_rows}_"
        f"img{args.image_size}.npz"
    )
    cache_path = cache_dir / cache_name

    print()
    print("=" * 80)
    print(f"[INFO] Processing bag {bag_index}/{total_bags}: {bag_path.name}")
    print("=" * 80)
    print(f"[INFO] Bag:             {bag_path}")
    print(f"[INFO] Topic:           {args.topic}")
    print(f"[INFO] Output images:   {output_images_dir}")
    print(f"[INFO] Output labels:   {output_labels_dir}")
    print(f"[INFO] Reports enabled: {reports_enabled}")
    print(f"[INFO] Cache folder:    {cache_dir}")
    print(f"[INFO] Device:          {args.device}")
    print(f"[INFO] Embedding mode:  {args.embedding_mode}")
    print(f"[INFO] Tile layout:     {args.tile_cols} cols x {args.tile_rows} rows")
    print(f"[INFO] Tile overlap:    {args.tile_overlap}")
    print(f"[INFO] DINO image size: {args.image_size}")

    embeddings, frame_rows = first_pass_compute_embeddings(args, model, transform, cache_path)

    print(f"[INFO] Final embedding shape: {embeddings.shape}")

    selected_indices, records, _ = select_indices_with_hdbscan(
        embeddings=embeddings,
        frame_rows=frame_rows,
        args=args,
    )

    saved_count, compressed_saved_count = second_pass_save_selected(
        args=args,
        records=records,
        selected_indices=selected_indices,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        compressed_images_dir=compressed_images_dir,
    )

    report_path = None
    summary_path = None
    selected_sheet_path = None
    compressed_sheet_path = None

    if reports_enabled:
        report_path, summary_path = write_reports(records, reports_dir)

        selected_sheet_path = reports_dir / "selected_contact_sheet.jpg"
        compressed_sheet_path = reports_dir / "compressed_out_contact_sheet.jpg"

        make_contact_sheet(records, selected_sheet_path, selected_value=True, max_images=args.max_contact_sheet_images)
        make_contact_sheet(records, compressed_sheet_path, selected_value=False, max_images=args.max_contact_sheet_images)
    else:
        print("[INFO] Reports disabled. CSV reports and contact sheets were not written.")

    total_candidates = len(records)
    selected_count = int(sum(1 for r in records if r["selected"]))
    compressed_count = total_candidates - selected_count

    print()
    print("[DONE]")
    print(f"Bag:                     {bag_path.name}")
    print(f"Candidate frames:        {total_candidates}")
    print(f"Selected frames:         {selected_count}")
    print(f"Saved selected frames:   {saved_count}")
    print(f"Compressed out frames:   {compressed_count}")
    print(f"Saved compressed frames: {compressed_saved_count}")
    print(f"Output images folder:    {output_images_dir}")
    print(f"Output labels folder:    {output_labels_dir}")
    print(f"Embedding cache:         {cache_path}")

    if reports_enabled:
        print(f"Report:                  {report_path}")
        print(f"Cluster summary:         {summary_path}")
        print(f"Selected contact sheet:  {selected_sheet_path}")
        if args.copy_compressed_out:
            print(f"Compressed contact sheet:{compressed_sheet_path}")
            print(f"Compressed images:       {compressed_images_dir}")
    else:
        print("Reports:                 disabled")
        if args.copy_compressed_out:
            print(f"Compressed images:       {compressed_images_dir}")

    if not args.create_empty_labels:
        print()
        print("[NOTE] Empty labels were not created.")
        print("       Set create_empty_labels: true in the YAML if you want matching YOLO txt files.")

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read one or more ROS2 bags from a folder, select useful frames using "
            "DINOv2 global/tile embeddings + HDBSCAN, and save selected frames."
        )
    )

    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--bags-root", type=Path, default=None, help="Folder containing one or more ROS2 bag folders.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Image topic. Default: {DEFAULT_TOPIC}")
    parser.add_argument("--output", type=Path, default=None, help="Output dataset root. Creates images/, labels/, reports/.")
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"], help="ROS bag storage backend.")
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "png"], help="Output image format.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality.")
    parser.add_argument("--create-empty-labels", action="store_true", help="Create empty YOLO txt file for each selected frame.")
    parser.add_argument("--every-n", type=int, default=1, help="Only consider every Nth image message before DINO selection.")
    parser.add_argument("--start-sec", type=float, default=None, help="Start time in seconds relative to first topic message.")
    parser.add_argument("--end-sec", type=float, default=None, help="End time in seconds relative to first topic message.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Max candidate frames. 0 means no limit.")
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
    parser.add_argument("--reuse-embeddings", action="store_true", help="Reuse cached embeddings if settings match.")
    parser.add_argument("--copy-compressed-out", action="store_true", help="Also save rejected frames for inspection.")
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

    if args.bags_root is None:
        raise ValueError("Missing rosbag.bags_root in YAML or --bags-root on CLI.")

    if args.output is None:
        raise ValueError("Missing paths.output in YAML or --output on CLI.")

    if args.every_n < 1:
        raise ValueError("every_n must be >= 1")

    if args.tile_cols < 1:
        raise ValueError("tile_cols must be >= 1")

    if args.tile_rows < 1:
        raise ValueError("tile_rows must be >= 1")

    if args.tile_overlap < 0.0:
        raise ValueError("tile_overlap must be >= 0.0")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    bag_paths = discover_bag_folders(args.bags_root)

    print()
    print("[INFO] Bags to process:")
    for bag_path in bag_paths:
        print(f"  - {bag_path}")

    print()
    print("[INFO] Loading DINO model once and reusing it for all bags.")
    transform = build_transform(args.image_size)
    model = load_dino_model(args.model, args.device)

    for idx, bag_path in enumerate(bag_paths, start=1):
        bag_args = copy.copy(args)
        bag_args.bag = bag_path

        run_one_bag(
            args=bag_args,
            model=model,
            transform=transform,
            bag_index=idx,
            total_bags=len(bag_paths),
        )

    print()
    print("=" * 80)
    print("[ALL DONE]")
    print(f"Processed bags: {len(bag_paths)}")
    print(f"Output root:    {args.output.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
