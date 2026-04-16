#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 3D MRI volumes from 2D image dataset.

Expected input layout (auto-detected):
    <root>/Training/<class_name>/*.jpg|png|tif
    <root>/Testing/<class_name>/*.jpg|png|tif

This script stacks 2D slices into 3D volumes using a sliding window and saves
them as .npy arrays for the next-stage 3D analysis/models.
"""

import argparse
import csv
import os
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def resolve_dataset_paths(project_dir: str) -> Tuple[str, str]:
    """Locate Training/Testing directories from common layouts."""
    candidates = [
        project_dir,
        os.path.join(project_dir, "Brain MRI Dataset", "brain mri dataset"),
        os.path.join(project_dir, "brain mri dataset"),
    ]

    for base in candidates:
        train_dir = os.path.join(base, "Training")
        test_dir = os.path.join(base, "Testing")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            print(f"[INFO] Using dataset base: {base}")
            return train_dir, test_dir

    # Fallback: walk under project and parent folder.
    search_roots = {project_dir, os.path.dirname(project_dir)}
    for root in search_roots:
        for current_dir, dirs, _ in os.walk(root):
            dirset = set(dirs)
            if "Training" in dirset and "Testing" in dirset:
                base = current_dir
                train_dir = os.path.join(base, "Training")
                test_dir = os.path.join(base, "Testing")
                print(f"[INFO] Auto-detected dataset base: {base}")
                return train_dir, test_dir

    raise FileNotFoundError(
        "Could not find Training/Testing directories. "
        "Place your dataset inside the project or pass --dataset-root explicitly."
    )


def list_class_images(split_dir: str) -> Dict[str, List[str]]:
    """Return image paths grouped by class for one split."""
    grouped = {}
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        images = [
            os.path.join(class_dir, fname)
            for fname in sorted(os.listdir(class_dir))
            if fname.lower().endswith(VALID_EXTENSIONS)
        ]
        if images:
            grouped[class_name] = images
    return grouped


def load_slice(path: str, image_size: int) -> np.ndarray:
    """Load a single 2D image as normalized grayscale slice."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def build_volumes_for_class(
    image_paths: List[str], depth: int, stride: int, image_size: int, allow_padding: bool
) -> List[np.ndarray]:
    """Build 3D volumes from sorted slice list."""
    slices = []
    for p in image_paths:
        try:
            slices.append(load_slice(p, image_size=image_size))
        except Exception as exc:
            print(f"[WARN] Skipping unreadable image: {p} ({exc})")

    if not slices:
        return []

    # If class has very few images, optionally pad to make one volume.
    if len(slices) < depth:
        if not allow_padding:
            return []
        last = slices[-1]
        while len(slices) < depth:
            slices.append(last.copy())

    volumes = []
    start = 0
    max_start = len(slices) - depth
    while start <= max_start:
        chunk = slices[start : start + depth]
        # Shape: (D, H, W, 1) -> suitable for many 3D CNN pipelines
        volume = np.stack(chunk, axis=0)[..., np.newaxis]
        volumes.append(volume)
        start += stride

    return volumes


def save_volumes(
    grouped_paths: Dict[str, List[str]],
    split_name: str,
    output_dir: str,
    depth: int,
    stride: int,
    image_size: int,
    allow_padding: bool,
    metadata_rows: List[List[str]],
) -> int:
    """Build and save volumes for one split."""
    total = 0
    split_out = os.path.join(output_dir, split_name)
    os.makedirs(split_out, exist_ok=True)

    for class_name, paths in grouped_paths.items():
        class_out = os.path.join(split_out, class_name)
        os.makedirs(class_out, exist_ok=True)

        volumes = build_volumes_for_class(
            image_paths=paths,
            depth=depth,
            stride=stride,
            image_size=image_size,
            allow_padding=allow_padding,
        )

        print(
            f"[INFO] {split_name}/{class_name}: "
            f"{len(paths)} slices -> {len(volumes)} volumes"
        )

        for idx, vol in enumerate(volumes):
            fname = f"vol_{idx:05d}.npy"
            out_path = os.path.join(class_out, fname)
            np.save(out_path, vol)
            metadata_rows.append(
                [
                    split_name,
                    class_name,
                    out_path,
                    str(vol.shape[0]),
                    str(vol.shape[1]),
                    str(vol.shape[2]),
                    str(vol.shape[3]),
                ]
            )
            total += 1

    return total


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D array to uint8 [0,255] safely."""
    arr = np.asarray(slice_2d, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _create_montage(volume_3d: np.ndarray) -> np.ndarray:
    """Create a tiled montage image from a 3D volume (D,H,W)."""
    depth, height, width = volume_3d.shape
    cols = int(math.ceil(math.sqrt(depth)))
    rows = int(math.ceil(depth / cols))
    canvas = np.zeros((rows * height, cols * width), dtype=np.uint8)

    for i in range(depth):
        r = i // cols
        c = i % cols
        y0 = r * height
        y1 = y0 + height
        x0 = c * width
        x1 = x0 + width
        canvas[y0:y1, x0:x1] = _normalize_to_uint8(volume_3d[i])

    return canvas


def export_npy_volumes_as_images(
    metadata_rows: List[List[str]],
    output_dir: str,
    image_format: str = "png",
    preview_mode: str = "middle",
) -> int:
    """Export each .npy volume as a preview image (middle slice or montage)."""
    image_format = image_format.lower()
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("--image-format must be one of: png, jpg, jpeg")
    if preview_mode not in {"middle", "montage"}:
        raise ValueError("--preview-mode must be one of: middle, montage")

    image_ext = "jpg" if image_format == "jpeg" else image_format
    preview_root = os.path.join(output_dir, "preview_images")
    os.makedirs(preview_root, exist_ok=True)

    saved = 0
    # Skip header row
    for row in metadata_rows[1:]:
        split, label, volume_path = row[0], row[1], row[2]
        rel_dir = os.path.join(preview_root, split, label)
        os.makedirs(rel_dir, exist_ok=True)

        try:
            vol = np.load(volume_path)
            # Support (D,H,W,1) and (D,H,W)
            if vol.ndim == 4 and vol.shape[-1] == 1:
                vol = vol[..., 0]
            if vol.ndim != 3:
                print(f"[WARN] Unexpected volume shape, skipping: {volume_path}")
                continue

            if preview_mode == "middle":
                mid = vol.shape[0] // 2
                preview = _normalize_to_uint8(vol[mid])
            else:
                preview = _create_montage(vol)

            base_name = os.path.splitext(os.path.basename(volume_path))[0]
            out_file = os.path.join(rel_dir, f"{base_name}.{image_ext}")
            success = cv2.imwrite(out_file, preview)
            if not success:
                print(f"[WARN] Failed to write image: {out_file}")
                continue
            saved += 1
        except Exception as exc:
            print(f"[WARN] Failed to export preview for {volume_path}: {exc}")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D MRI class folders into stacked 3D volumes."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Optional explicit dataset root containing Training/Testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mri_3d_volumes",
        help="Directory to save generated .npy volumes.",
    )
    parser.add_argument("--depth", type=int, default=16, help="Slices per volume.")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride.")
    parser.add_argument(
        "--image-size", type=int, default=128, help="Resize slices to square size."
    )
    parser.add_argument(
        "--allow-padding",
        action="store_true",
        help="Pad short classes by repeating last slice to create at least one volume.",
    )
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="Also export each .npy volume into a viewable PNG/JPG preview image.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        help="Preview image format: png or jpg.",
    )
    parser.add_argument(
        "--preview-mode",
        type=str,
        default="middle",
        help="Preview mode: 'middle' (single central slice) or 'montage' (all slices tiled).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset_root:
        base = os.path.abspath(args.dataset_root)
        train_dir = os.path.join(base, "Training")
        test_dir = os.path.join(base, "Testing")
        if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
            raise FileNotFoundError(
                f"Provided --dataset-root does not contain Training/Testing: {base}"
            )
        print(f"[INFO] Using user dataset root: {base}")
    else:
        train_dir, test_dir = resolve_dataset_paths(script_dir)

    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(script_dir, args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    train_grouped = list_class_images(train_dir)
    test_grouped = list_class_images(test_dir)

    metadata_rows = [
        ["split", "label", "volume_path", "depth", "height", "width", "channels"]
    ]

    total_train = save_volumes(
        grouped_paths=train_grouped,
        split_name="Training",
        output_dir=output_dir,
        depth=args.depth,
        stride=args.stride,
        image_size=args.image_size,
        allow_padding=args.allow_padding,
        metadata_rows=metadata_rows,
    )
    total_test = save_volumes(
        grouped_paths=test_grouped,
        split_name="Testing",
        output_dir=output_dir,
        depth=args.depth,
        stride=args.stride,
        image_size=args.image_size,
        allow_padding=args.allow_padding,
        metadata_rows=metadata_rows,
    )

    metadata_path = os.path.join(output_dir, "volumes_metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(metadata_rows)

    print("\n[OK] 3D volume generation completed.")
    print(f"[OK] Training volumes: {total_train}")
    print(f"[OK] Testing volumes: {total_test}")
    print(f"[OK] Output directory: {output_dir}")
    print(f"[OK] Metadata CSV: {metadata_path}")

    if args.export_images:
        print("[INFO] Exporting .npy previews as images...")
        exported_count = export_npy_volumes_as_images(
            metadata_rows=metadata_rows,
            output_dir=output_dir,
            image_format=args.image_format,
            preview_mode=args.preview_mode,
        )
        preview_dir = os.path.join(output_dir, "preview_images")
        print(f"[OK] Exported preview images: {exported_count}")
        print(f"[OK] Preview directory: {preview_dir}")


if __name__ == "__main__":
    main()
