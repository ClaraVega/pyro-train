#!/usr/bin/env python3
"""
Split a grouped exported dataset into YOLO folder structure.

Input expected:
root_dir/
  <group_id_1>/
    images/  (contains .jpg/.png/...)
    labels/  (contains .txt with same stem as image)
  <group_id_2>/
    images/
    labels/
  ...

Output created:
output_dir/
  images/train, images/val
  labels/train, labels/val

By default, files are COPIED. You can use --move to move instead.

Example:
python split_exported_dataset.py \
  --root-dir "Projet/dataset_exported_20260114_211415/wildfire" \
  --output-dir "Projet/yolo_ready" \
  --val-ratio 0.2 \
  --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs(out: Path) -> None:
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_pairs(root_dir: Path):
    """
    Returns list of dicts: {group_id, img_path, label_path_or_None}
    """
    items = []
    # Each direct child is a group folder with images/ and labels/
    for group_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        images_dir = group_dir / "images"
        labels_dir = group_dir / "labels"
        if not images_dir.exists():
            continue

        # Collect images in this group (non-recursive; change to rglob if needed)
        for img_path in sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]):
            label_path = None
            if labels_dir.exists():
                candidate = labels_dir / f"{img_path.stem}.txt"
                if candidate.exists():
                    label_path = candidate

            items.append(
                {
                    "group_id": group_dir.name,
                    "img": img_path,
                    "label": label_path,
                }
            )
    return items


def safe_copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def unique_name(group_id: str, stem: str, suffix: str) -> str:
    # Prefix with group_id to avoid collisions if different groups have same filenames
    return f"{group_id}__{stem}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", type=Path, required=True, help="Path to .../wildfire (contains group subfolders).")
    ap.add_argument("--output-dir", type=Path, required=True, help="Where to write YOLO train/val structure.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation set (0..1).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument(
        "--require-labels",
        action="store_true",
        help="If set, only keep images that have a matching label .txt.",
    )
    ap.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    args = ap.parse_args()

    if not args.root_dir.exists():
        raise SystemExit(f"root-dir not found: {args.root_dir}")

    if not (0.0 <= args.val_ratio <= 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1")

    items = collect_pairs(args.root_dir)
    if args.require_labels:
        items = [it for it in items if it["label"] is not None]

    if len(items) == 0:
        raise SystemExit("No images found (check your root-dir and structure).")

    ensure_dirs(args.output_dir)

    rng = random.Random(args.seed)
    rng.shuffle(items)

    n_total = len(items)
    n_val = int(round(n_total * args.val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]

    def write_split(split_name: str, split_items):
        for it in split_items:
            group_id = it["group_id"]
            img_path: Path = it["img"]
            lbl_path: Path | None = it["label"]

            out_img_name = unique_name(group_id, img_path.stem, img_path.suffix.lower())
            out_img = args.output_dir / "images" / split_name / out_img_name
            safe_copy_or_move(img_path, out_img, move=args.move)

            if lbl_path is not None:
                out_lbl_name = unique_name(group_id, lbl_path.stem, ".txt")
                out_lbl = args.output_dir / "labels" / split_name / out_lbl_name
                safe_copy_or_move(lbl_path, out_lbl, move=args.move)

    write_split("train", train_items)
    write_split("val", val_items)

    # Small summary
    n_train = len(train_items)
    n_val = len(val_items)
    n_with_labels = sum(1 for it in items if it["label"] is not None)
    print("Done.")
    print(f"Total images considered: {n_total}")
    print(f"Images with labels found: {n_with_labels}")
    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()