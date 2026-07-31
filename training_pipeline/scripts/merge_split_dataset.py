#!/usr/bin/env python3
import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}

VALID_SPLITS = {"train", "val", "test"}
ROOT_FOLDER_NAME = "_root"

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge YOLO datasets using a folder-level split registry. "
            "Existing folders keep their split; only unseen folders are assigned."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to merge/split YAML config.",
    )

    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty YAML file: {path}")

    return data

def resolve_repo_path(path_value) -> Path:
    path = Path(str(path_value).strip()).expanduser()

    if path.is_absolute():
        return path

    return (Path.cwd() / path).resolve()

def natural_key(path: Path):
    import re

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]

def find_images_root(dataset_root: Path) -> Path:
    images_root = dataset_root / "images"

    if images_root.exists():
        return images_root

    return dataset_root

def find_labels_root(dataset_root: Path) -> Path:
    labels_root = dataset_root / "labels"

    if labels_root.exists():
        return labels_root

    return dataset_root

def list_images(images_root: Path):
    if not images_root.exists():
        return []

    images = [
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images, key=natural_key)

def normalize_split(split_value: str) -> str:
    split = str(split_value).strip().lower()
    split = SPLIT_ALIASES.get(split, split)

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split_value}'. Use train, val, or test.")

    return split

def folder_and_relative_inside(rel_path: Path):
    parts = rel_path.parts

    if len(parts) <= 1:
        return ROOT_FOLDER_NAME, Path(rel_path.name)

    folder = parts[0]
    rel_inside = Path(*parts[1:])

    return folder, rel_inside

def read_split_registry(registry_path: Path):
    registry = {}

    if not registry_path.exists():
        print(f"[INFO] Split registry does not exist yet: {registry_path}")
        print("[INFO] It will be created.")
        return registry

    with open(registry_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"source", "folder", "split"}

        if reader.fieldnames is None:
            return registry

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Registry CSV is missing columns: {sorted(missing)}\n"
                f"Expected columns: source,folder,split"
            )

        for row in reader:
            source = str(row["source"]).strip()
            folder = str(row["folder"]).strip()
            split = normalize_split(row["split"])

            if not source or not folder:
                continue

            key = (source, folder)

            if key in registry and registry[key] != split:
                raise ValueError(
                    f"Conflicting registry entry for {source},{folder}: "
                    f"{registry[key]} vs {split}"
                )

            registry[key] = split

    print(f"[INFO] Loaded registry entries: {len(registry)}")
    return registry

def write_split_registry(registry_path: Path, registry: dict):
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (source, folder), split in sorted(registry.items()):
        rows.append({
            "source": source,
            "folder": folder,
            "split": split,
        })

    with open(registry_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "folder", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Updated split registry: {registry_path}")
    print(f"[INFO] Registry rows: {len(rows)}")

def add_image_to_group(
    groups_by_key,
    source_name: str,
    folder: str,
    image_path: Path,
    label_path: Path,
    rel_inside: Path,
    existing_split,
    already_split: bool,
):
    key = (source_name, folder)

    if key not in groups_by_key:
        groups_by_key[key] = {
            "source_name": source_name,
            "folder": folder,
            "already_split": already_split,
            "existing_split": existing_split,
            "assigned_split": None,
            "items": [],
        }

    group = groups_by_key[key]

    if existing_split is not None:
        if group["existing_split"] is not None and group["existing_split"] != existing_split:
            raise ValueError(
                f"Folder appears in multiple splits:\n"
                f"  source: {source_name}\n"
                f"  folder: {folder}\n"
                f"  splits: {group['existing_split']} and {existing_split}"
            )

        group["existing_split"] = existing_split

    group["items"].append({
        "image_path": image_path,
        "label_path": label_path,
        "rel_inside": rel_inside,
    })

def collect_unsplit_source(
    source_name: str,
    images_root: Path,
    labels_root: Path,
    groups_by_key,
):
    image_paths = list_images(images_root)

    for image_path in image_paths:
        rel_path = image_path.relative_to(images_root)
        folder, rel_inside = folder_and_relative_inside(rel_path)

        label_path = labels_root / rel_path.with_suffix(".txt")

        add_image_to_group(
            groups_by_key=groups_by_key,
            source_name=source_name,
            folder=folder,
            image_path=image_path,
            label_path=label_path,
            rel_inside=rel_inside,
            existing_split=None,
            already_split=False,
        )

    return len(image_paths)

def collect_already_split_source(
    source_name: str,
    images_root: Path,
    labels_root: Path,
    groups_by_key,
):
    total_images = 0

    for split_name in ["train", "val", "test"]:
        split_images_root = images_root / split_name
        split_labels_root = labels_root / split_name

        if not split_images_root.exists():
            print(f"[WARN] Missing split folder: {split_images_root}")
            continue

        image_paths = list_images(split_images_root)

        for image_path in image_paths:
            rel_path = image_path.relative_to(split_images_root)
            folder, rel_inside = folder_and_relative_inside(rel_path)

            label_path = split_labels_root / rel_path.with_suffix(".txt")

            add_image_to_group(
                groups_by_key=groups_by_key,
                source_name=source_name,
                folder=folder,
                image_path=image_path,
                label_path=label_path,
                rel_inside=rel_inside,
                existing_split=split_name,
                already_split=True,
            )

        total_images += len(image_paths)

    return total_images

def collect_folder_groups(cfg: dict):
    groups_by_key = {}

    for source in cfg["sources"]:
        source_name = str(source["name"]).strip()
        dataset_root = resolve_repo_path(source["root"])
        already_split = bool(source.get("already_split", False))

        if not source_name:
            raise ValueError("Every source needs a non-empty name.")

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        images_root = find_images_root(dataset_root)
        labels_root = find_labels_root(dataset_root)

        print()
        print(f"[INFO] Source:        {source_name}")
        print(f"[INFO] Root:          {dataset_root}")
        print(f"[INFO] Images root:   {images_root}")
        print(f"[INFO] Labels root:   {labels_root}")
        print(f"[INFO] Already split: {already_split}")

        if already_split:
            image_count = collect_already_split_source(
                source_name=source_name,
                images_root=images_root,
                labels_root=labels_root,
                groups_by_key=groups_by_key,
            )
        else:
            image_count = collect_unsplit_source(
                source_name=source_name,
                images_root=images_root,
                labels_root=labels_root,
                groups_by_key=groups_by_key,
            )

        print(f"[INFO] Images found:  {image_count}")

    groups = list(groups_by_key.values())

    if not groups:
        raise RuntimeError("No image folders were collected from any source.")

    print()
    print(f"[INFO] Folder groups found: {len(groups)}")

    return groups

def validate_split_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    for name, value in {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }.items():
        if value < 0.0:
            raise ValueError(f"Split ratio for {name} must be >= 0.0")

def assign_splits_from_registry(groups, registry, cfg):
    split_cfg = cfg["split"]

    mode = str(split_cfg.get("mode", "registry")).strip().lower()
    if mode != "registry":
        raise ValueError("This script version expects split.mode: registry")

    train_ratio = float(split_cfg.get("train", 0.80))
    val_ratio = float(split_cfg.get("val", 0.10))
    test_ratio = float(split_cfg.get("test", 0.10))
    seed = int(split_cfg.get("seed", 42))

    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    assigned_counts = {"train": 0, "val": 0, "test": 0}
    unknown_groups = []

    registered_existing = 0
    registered_from_split_folders = 0

    for group in groups:
        source = group["source_name"]
        folder = group["folder"]
        key = (source, folder)

        if key in registry:
            split = registry[key]
            group["assigned_split"] = split
            assigned_counts[split] += len(group["items"])
            registered_existing += 1
            continue

        if group["existing_split"] in VALID_SPLITS:
            split = group["existing_split"]
            group["assigned_split"] = split
            registry[key] = split
            assigned_counts[split] += len(group["items"])
            registered_from_split_folders += 1
            continue

        unknown_groups.append(group)

    print()
    print("[INFO] Registry assignment")
    print(f"  already in registry:       {registered_existing}")
    print(f"  initialized from old split: {registered_from_split_folders}")
    print(f"  new folders to assign:      {len(unknown_groups)}")

    if not unknown_groups:
        return groups, registry

    rng = random.Random(seed)

    unknown_groups = sorted(
        unknown_groups,
        key=lambda group: (group["source_name"], group["folder"]),
    )
    rng.shuffle(unknown_groups)

    unknown_images = sum(len(group["items"]) for group in unknown_groups)
    final_total_images = sum(assigned_counts.values()) + unknown_images

    target_counts = {
        split_name: final_total_images * ratio
        for split_name, ratio in ratios.items()
    }

    print()
    print("[INFO] Assigning only new folders")
    print(f"  new images: {unknown_images}")
    print(f"  target final train images: {target_counts['train']:.1f}")
    print(f"  target final val images:   {target_counts['val']:.1f}")
    print(f"  target final test images:  {target_counts['test']:.1f}")

    split_priority = ["train", "val", "test"]

    for group in unknown_groups:
        deficits = {
            split_name: target_counts[split_name] - assigned_counts[split_name]
            for split_name in ["train", "val", "test"]
        }

        split_name = max(
            split_priority,
            key=lambda s: (deficits[s], ratios[s], -split_priority.index(s)),
        )

        group["assigned_split"] = split_name
        registry[(group["source_name"], group["folder"])] = split_name
        assigned_counts[split_name] += len(group["items"])

        print(
            f"  {group['source_name']},{group['folder']} -> {split_name} "
            f"({len(group['items'])} images)"
        )

    return groups, registry

def print_balance_report(groups, cfg):
    split_cfg = cfg["split"]

    target_ratios = {
        "train": float(split_cfg.get("train", 0.80)),
        "val": float(split_cfg.get("val", 0.10)),
        "test": float(split_cfg.get("test", 0.10)),
    }

    warning_threshold = float(
        split_cfg.get("balance_warning_threshold_percent", 5.0)
    )

    image_counts = {split: 0 for split in ["train", "val", "test"]}
    folder_counts = {split: 0 for split in ["train", "val", "test"]}

    for group in groups:
        split_name = group.get("assigned_split")

        if split_name not in VALID_SPLITS:
            continue

        image_counts[split_name] += len(group["items"])
        folder_counts[split_name] += 1

    total_images = sum(image_counts.values())
    total_folders = sum(folder_counts.values())

    if total_images == 0:
        print("[WARN] No images found for split balance report.")
        return

    print()
    print("[INFO] Split balance report")
    print("[INFO] Target ratios:")
    print(
        f"  train={target_ratios['train'] * 100:.1f}% "
        f"val={target_ratios['val'] * 100:.1f}% "
        f"test={target_ratios['test'] * 100:.1f}%"
    )

    imbalance_messages = []

    print()
    print("  By images:")
    for split_name in ["train", "val", "test"]:
        count = image_counts[split_name]
        current_pct = 100.0 * count / total_images
        target_pct = 100.0 * target_ratios[split_name]
        delta_pct = current_pct - target_pct
        target_count = total_images * target_ratios[split_name]
        delta_images = count - target_count

        print(
            f"    {split_name:5s}: "
            f"{count:6d} images "
            f"({current_pct:6.2f}%) | "
            f"target {target_pct:6.2f}% | "
            f"delta {delta_pct:+6.2f} pp "
            f"({delta_images:+.1f} images)"
        )

        if abs(delta_pct) > warning_threshold:
            if delta_pct < 0:
                imbalance_messages.append(
                    f"{split_name} is under target by about "
                    f"{abs(delta_pct):.2f} percentage points "
                    f"(roughly {-delta_images:.0f} images)."
                )
            else:
                imbalance_messages.append(
                    f"{split_name} is over target by about "
                    f"{delta_pct:.2f} percentage points "
                    f"(roughly {delta_images:.0f} images)."
                )

    if total_folders > 0:
        print()
        print("  By folders:")
        for split_name in ["train", "val", "test"]:
            count = folder_counts[split_name]
            current_pct = 100.0 * count / total_folders
            target_pct = 100.0 * target_ratios[split_name]
            delta_pct = current_pct - target_pct

            print(
                f"    {split_name:5s}: "
                f"{count:6d} folders "
                f"({current_pct:6.2f}%) | "
                f"target {target_pct:6.2f}% | "
                f"delta {delta_pct:+6.2f} pp"
            )

    print()
    if imbalance_messages:
        print("[SUGGESTION] Split balance could be improved:")
        for message in imbalance_messages:
            print(f"  - {message}")

        under_splits = []
        over_splits = []

        for split_name in ["train", "val", "test"]:
            current_pct = 100.0 * image_counts[split_name] / total_images
            target_pct = 100.0 * target_ratios[split_name]
            delta_pct = current_pct - target_pct

            if delta_pct < -warning_threshold:
                under_splits.append(split_name)
            elif delta_pct > warning_threshold:
                over_splits.append(split_name)

        if under_splits:
            print(
                f"  Add future folders preferably to: {', '.join(under_splits)}"
            )

        if over_splits:
            print(
                f"  Avoid adding many future folders to: {', '.join(over_splits)}"
            )

        print(
            "  Since the split is folder-level, do not move individual images. "
            "For manual balancing, edit split_registry.csv by moving complete "
            "folders from overrepresented splits to underrepresented ones."
        )
    else:
        print(
            "[SUGGESTION] Split balance looks acceptable based on the configured "
            f"threshold of {warning_threshold:.1f} percentage points."
        )

def ensure_safe_output(output_root: Path, source_roots, overwrite: bool):
    output_root = output_root.resolve()

    for source_root in source_roots:
        source_root = source_root.resolve()

        if output_root == source_root:
            raise ValueError(
                f"Refusing to write output into source dataset root: {output_root}"
            )

        try:
            output_root.relative_to(source_root)
            raise ValueError(
                f"Refusing to write output inside a source dataset root:\n"
                f"  output: {output_root}\n"
                f"  source: {source_root}"
            )
        except ValueError as exc:
            if "Refusing" in str(exc):
                raise

    if output_root.exists() and overwrite:
        print(f"[INFO] Removing existing output dataset: {output_root}")
        shutil.rmtree(output_root)

    if output_root.exists() and not overwrite:
        images_root = output_root / "images"
        labels_root = output_root / "labels"

        if images_root.exists() or labels_root.exists():
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                f"Set output.overwrite: true to replace it."
            )

    output_root.mkdir(parents=True, exist_ok=True)

def copy_or_symlink(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unknown output.mode: {mode}")

def copy_label(label_path: Path, dst_label_path: Path, missing_label_policy: str):
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    if label_path.exists():
        shutil.copy2(label_path, dst_label_path)
        return "copied"

    if missing_label_policy == "create_empty":
        dst_label_path.write_text("")
        return "empty_created"

    if missing_label_policy == "skip":
        return "skipped_missing_label"

    if missing_label_policy == "error":
        raise FileNotFoundError(f"Missing label file: {label_path}")

    raise ValueError(f"Unknown missing_label_policy: {missing_label_policy}")

def output_relative_path(group, item):
    source_name = group["source_name"]
    folder = group["folder"]
    rel_inside = item["rel_inside"]

    if folder == ROOT_FOLDER_NAME:
        return Path(source_name) / ROOT_FOLDER_NAME / rel_inside

    return Path(source_name) / folder / rel_inside

def write_dataset(groups, output_root: Path, mode: str, missing_label_policy: str):
    copied_images = 0
    copied_labels = 0
    empty_labels = 0
    skipped_images = 0

    split_counts = defaultdict(lambda: defaultdict(int))
    folder_counts = defaultdict(lambda: defaultdict(int))

    for group in groups:
        split_name = group["assigned_split"]

        if split_name not in VALID_SPLITS:
            raise ValueError(
                f"Group has no valid assigned split:\n"
                f"  source: {group['source_name']}\n"
                f"  folder: {group['folder']}\n"
                f"  split: {split_name}"
            )

        split_counts[split_name][group["source_name"]] += len(group["items"])
        folder_counts[split_name][group["source_name"]] += 1

        for item in group["items"]:
            out_rel = output_relative_path(group, item)

            dst_image_path = output_root / "images" / split_name / out_rel
            dst_label_path = output_root / "labels" / split_name / out_rel.with_suffix(".txt")

            label_status = copy_label(
                label_path=item["label_path"],
                dst_label_path=dst_label_path,
                missing_label_policy=missing_label_policy,
            )

            if label_status == "skipped_missing_label":
                skipped_images += 1
                continue

            copy_or_symlink(
                src=item["image_path"],
                dst=dst_image_path,
                mode=mode,
            )

            copied_images += 1

            if label_status == "copied":
                copied_labels += 1
            elif label_status == "empty_created":
                empty_labels += 1

    print()
    print("[INFO] Copy summary")
    print(f"  copied images:        {copied_images}")
    print(f"  copied labels:        {copied_labels}")
    print(f"  empty labels created: {empty_labels}")
    print(f"  skipped images:       {skipped_images}")

    print()
    print("[INFO] Final split summary")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}:")

        total_images = sum(split_counts[split_name].values())
        total_folders = sum(folder_counts[split_name].values())

        print(f"    images:  {total_images}")
        print(f"    folders: {total_folders}")

        for source_name in sorted(split_counts[split_name].keys()):
            print(
                f"    {source_name}: "
                f"{split_counts[split_name][source_name]} images, "
                f"{folder_counts[split_name][source_name]} folders"
            )

def write_data_yaml(path: Path, output_root: Path, classes_cfg: dict):
    id_to_name = classes_cfg["id_to_name"]

    names = {
        int(class_id): str(class_name)
        for class_id, class_name in id_to_name.items()
    }

    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "nc": len(names),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    output_root = resolve_repo_path(cfg["paths"]["output_root"])
    data_yaml_path = resolve_repo_path(cfg["paths"]["data_yaml"])
    registry_path = resolve_repo_path(cfg["registry"]["path"])

    output_cfg = cfg.get("output", {})
    mode = str(output_cfg.get("mode", "copy")).strip().lower()
    overwrite = bool(output_cfg.get("overwrite", False))
    missing_label_policy = str(
        output_cfg.get("missing_label_policy", "create_empty")
    ).strip().lower()

    if mode not in {"copy", "symlink"}:
        raise ValueError("output.mode must be copy or symlink")

    if missing_label_policy not in {"create_empty", "skip", "error"}:
        raise ValueError(
            "output.missing_label_policy must be create_empty, skip, or error"
        )

    source_roots = [
        resolve_repo_path(source["root"])
        for source in cfg["sources"]
    ]

    ensure_safe_output(
        output_root=output_root,
        source_roots=source_roots,
        overwrite=overwrite,
    )

    registry = read_split_registry(registry_path)

    groups = collect_folder_groups(cfg)

    groups, registry = assign_splits_from_registry(
        groups=groups,
        registry=registry,
        cfg=cfg,
    )

    print_balance_report(groups, cfg)

    write_dataset(
        groups=groups,
        output_root=output_root,
        mode=mode,
        missing_label_policy=missing_label_policy,
    )

    write_split_registry(registry_path, registry)

    write_data_yaml(
        path=data_yaml_path,
        output_root=output_root,
        classes_cfg=cfg["classes"],
    )

    print()
    print("[DONE]")
    print(f"Combined dataset: {output_root}")
    print(f"Data YAML:        {data_yaml_path}")
    print(f"Split registry:   {registry_path}")

if __name__ == "__main__":
    main()