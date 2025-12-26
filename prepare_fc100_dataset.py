#!/usr/bin/env python3
"""
Prepare an FC100-style folder dataset from CIFAR-100 python pickles.

This repo's episodic dataloader (`data.py`) expects images arranged on disk as:

  datasets/<dataset_name>/{train,val,test}/<class_name>/<image>.png

and uses the directory names to infer both the split ("train"/"val"/"test") and
the class label.

This script:
  - Reads CIFAR-100 python-format pickles (train/test/meta)
  - Merges CIFAR-100 train+test to use all 600 images per class
  - Splits fine classes into FC100 meta-train/meta-val/meta-test by coarse label:
        coarse 0-11  -> train (60 classes)
        coarse 12-15 -> val   (20 classes)
        coarse 16-19 -> test  (20 classes)
  - Writes PNGs into `datasets/fc100/{train,val,test}/<fine_label_name>/...`
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="bytes")


def _decode_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _coarse_to_split(coarse_label: int) -> str:
    if 0 <= coarse_label <= 11:
        return "train"
    if 12 <= coarse_label <= 15:
        return "val"
    if 16 <= coarse_label <= 19:
        return "test"
    raise ValueError(f"Unexpected coarse label {coarse_label} (expected 0..19).")


def _iter_cifar_samples(split_name: str, d):
    data = np.asarray(d[b"data"])
    fine = d[b"fine_labels"]
    coarse = d[b"coarse_labels"]
    filenames = d.get(b"filenames", None)

    if data.ndim != 2 or data.shape[1] != 3072:
        raise ValueError(f"Unexpected CIFAR-100 data shape: {data.shape}")

    if not (len(fine) == len(coarse) == data.shape[0]):
        raise ValueError("CIFAR-100 pickle fields have inconsistent lengths.")

    for i in range(data.shape[0]):
        filename = None
        if filenames is not None:
            filename = _decode_str(filenames[i])
        yield {
            "split": split_name,
            "index": i,
            "flat": data[i],
            "fine": int(fine[i]),
            "coarse": int(coarse[i]),
            "filename": filename,
        }


def _flat_to_rgb_image(flat: np.ndarray) -> Image.Image:
    arr = np.asarray(flat, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
    return Image.fromarray(arr, mode="RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cifar-root",
        type=str,
        default="datasets/cifar-100-python",
        help="Path to CIFAR-100 python pickles directory (contains train/test/meta).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="datasets/fc100",
        help="Output directory for FC100 folder dataset.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents (dangerous).",
    )
    args = ap.parse_args()

    cifar_root = Path(args.cifar_root)
    out_dir = Path(args.out_dir)

    train_p = cifar_root / "train"
    test_p = cifar_root / "test"
    meta_p = cifar_root / "meta"
    for p in (train_p, test_p, meta_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing CIFAR-100 file: {p}")

    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError(f"--out-dir exists but is not a directory: {out_dir}")
        if any(out_dir.iterdir()) and not args.overwrite:
            raise RuntimeError(
                f"--out-dir is not empty: {out_dir}. Pass --overwrite to continue."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    meta = _load_pickle(meta_p)
    fine_names = [_decode_str(x) for x in meta[b"fine_label_names"]]

    train = _load_pickle(train_p)
    test = _load_pickle(test_p)

    total = int(np.asarray(train[b"data"]).shape[0] + np.asarray(test[b"data"]).shape[0])
    written = 0
    per_fine_count = {i: 0 for i in range(100)}
    fine_to_split = {}

    for sample in _iter_cifar_samples("train", train):
        split = _coarse_to_split(sample["coarse"])
        fine_to_split.setdefault(sample["fine"], split)
    for sample in _iter_cifar_samples("test", test):
        split = _coarse_to_split(sample["coarse"])
        fine_to_split.setdefault(sample["fine"], split)

    # Sanity: ensure each fine class maps to a single split.
    for split_name, d in (("train", train), ("test", test)):
        for i, sample in enumerate(_iter_cifar_samples(split_name, d)):
            if i >= 1000:
                break
            expected = fine_to_split[sample["fine"]]
            actual = _coarse_to_split(sample["coarse"])
            if expected != actual:
                raise RuntimeError(
                    f"Fine label {sample['fine']} maps to multiple coarse splits: {expected} vs {actual}"
                )

    def write_sample(sample):
        nonlocal written
        fine = sample["fine"]
        split = fine_to_split[fine]
        fine_name = fine_names[fine]
        class_dir = out_dir / split / fine_name
        class_dir.mkdir(parents=True, exist_ok=True)

        orig = sample["filename"] or f"{sample['split']}_{sample['index']:05d}.png"
        orig = os.path.basename(orig)
        stem = f"{sample['split']}_{sample['index']:05d}_{orig}"
        stem = stem.replace("/", "_")
        if not stem.lower().endswith(".png"):
            stem = f"{stem}.png"
        img_path = class_dir / stem

        img = _flat_to_rgb_image(sample["flat"])
        img.save(img_path)

        per_fine_count[fine] += 1
        written += 1
        if written % 5000 == 0 or written == total:
            print(f"[fc100] wrote {written}/{total} images")

    for sample in _iter_cifar_samples("train", train):
        write_sample(sample)
    for sample in _iter_cifar_samples("test", test):
        write_sample(sample)

    # Final sanity: CIFAR-100 has exactly 600 images per fine label across train+test.
    bad = {k: v for k, v in per_fine_count.items() if v != 600}
    if bad:
        raise RuntimeError(
            f"Expected 600 images per fine label, but got mismatches for: {bad}"
        )

    split_counts = {"train": 0, "val": 0, "test": 0}
    for fine, split in fine_to_split.items():
        split_counts[split] += 1
    print(f"[fc100] fine-class split counts: {split_counts} (expected 60/20/20)")

    print(f"[fc100] done. Dataset written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
