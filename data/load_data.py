# data/load_data.py

import os
from pathlib import Path
from typing import Tuple

from datasets import load_from_disk, load_dataset, Dataset, DatasetDict, ClassLabel

# Path where you saved the QA dataset built earlier with create_qa_dataset.py
# (the one that has question, answer, video_id, subset, video_url, etc.)
QA_DATASET_LOCAL = "egotextvqa_video_qa"  # folder created by qa_ds.save_to_disk(...)

# Metadata-only dataset on Hub (no video column)
QA_DATASET_META_HF = "petkopetkov/EgoTextVQA"

# Root folder where your processed videos live, e.g. fps6 + low-res:
#   PROCESSED_VIDEO_ROOT / <subset> / <video_id>.mp4
PROCESSED_VIDEO_ROOT = "./data/egotextvqa_fps6_lowres"

# Where to save the final train/val/test splits for reuse (training + prediction)
SPLIT_SAVE_DIR = "egotextvqa_video_qa_splits"


def _load_raw_qa_dataset(use_local: bool = True) -> Dataset:
    """
    Load the raw QA dataset (no splits), either from a local Arrow folder or from the Hub.
    """
    if use_local and os.path.exists(QA_DATASET_LOCAL):
        print(f"[INFO] Loading QA dataset from disk: {QA_DATASET_LOCAL}")
        ds = load_from_disk(QA_DATASET_LOCAL)
    else:
        print(f"[INFO] Loading QA dataset from Hub: {QA_DATASET_META_HF}")
        ds = load_dataset(QA_DATASET_META_HF, split="train")
    print(ds)
    return ds


def _normalize_subset(example, subset_feature=None):
    """
    Make sure subset is stored as a string: "EgoTextVQA-Indoor"/"EgoTextVQA-Outdoor".
    (Safe even if it's already a string.)
    """
    raw = example["subset"]
    if subset_feature is not None and isinstance(subset_feature, ClassLabel):
        example["subset"] = subset_feature.int2str(raw)
    else:
        example["subset"] = str(raw)
    return example


def _add_local_video_path(example, processed_root: str):
    """
    Add a 'local_video_path' field pointing to the pre-processed video file.

    Expected layout:
        processed_root / subset / video_id.mp4

    We defensively cast both subset and video_id to strings in case they are ints.
    """
    subset_name = "EgoTextVQA-Indoor" if example["subset"] == 0 else "EgoTextVQA-Outdoor"
    video_id = str(example["video_id"])
    # use absolute path
    example["local_video_path"] = os.path.abspath(os.path.join(processed_root, subset_name, f"{video_id}.mp4"))
    print("video path: ", example["local_video_path"])
    return example


def _filter_existing_videos(example):
    """
    Filter out examples whose local video file does NOT exist.
    """
    path = example["local_video_path"]
    return os.path.exists(path)


def split_by_video_id(
    dataset: Dataset,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test by unique video_id to avoid leakage across splits.
    """
    import random

    video_ids = list(set(dataset["video_id"]))
    random.Random(seed).shuffle(video_ids)

    n = len(video_ids)
    train_cut = int(train_ratio * n)
    val_cut = int((train_ratio + val_ratio) * n)

    train_ids = set(video_ids[:train_cut])
    val_ids = set(video_ids[train_cut:val_cut])
    test_ids = set(video_ids[val_cut:])

    print(f"[INFO] #video_ids train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_ds = dataset.filter(lambda ex: ex["video_id"] in train_ids)
    val_ds = dataset.filter(lambda ex: ex["video_id"] in val_ids)
    test_ds = dataset.filter(lambda ex: ex["video_id"] in test_ids)

    return train_ds, val_ds, test_ds


def load_data(
    processed_video_root: str = PROCESSED_VIDEO_ROOT,
    use_local_dataset: bool = True,
    save_splits: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the QA dataset, normalize subset, attach local_video_path, filter missing videos,
    split into train/val/test by video_id, optionally save the splits to disk.

    Returns:
        train_ds, val_ds, test_ds
    """
    ds = _load_raw_qa_dataset(use_local=use_local_dataset)

    # Normalize subset to string (handles ClassLabel or already-string)
    subset_feature = ds.features.get("subset", None)
    ds = ds.map(
        lambda ex: _normalize_subset(ex, subset_feature=subset_feature),
        desc="Normalizing subset as string",
    )

    # Add local_video_path (robust str-casting inside)
    ds = ds.map(
        lambda ex: _add_local_video_path(ex, processed_root=processed_video_root),
        desc="Adding local_video_path",
    )
    
    # apply _add_local_video_path to all examples
    for example in ds:
        _add_local_video_path(example, processed_root=processed_video_root)

    # Filter out rows where the processed video file doesn't exist
    ds = ds.filter(_filter_existing_videos, desc="Filtering examples with missing local_video_path")
    print(f"[INFO] After filtering: {len(ds)} examples remain.")

    # Split
    train_ds, val_ds, test_ds = split_by_video_id(ds)

    if save_splits:
        os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
        splits = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
        print(f"[INFO] Saving splits to disk at: {SPLIT_SAVE_DIR}")
        splits.save_to_disk(SPLIT_SAVE_DIR)

    return train_ds, val_ds, test_ds


def load_splits() -> DatasetDict:
    """
    Load previously saved splits (train/validation/test) from disk.
    """
    if not os.path.exists(SPLIT_SAVE_DIR):
        raise FileNotFoundError(
            f"Split folder '{SPLIT_SAVE_DIR}' not found. "
            f"Run a training script that calls load_data(..., save_splits=True) first."
        )
    print(f"[INFO] Loading splits from disk: {SPLIT_SAVE_DIR}")
    return load_from_disk(SPLIT_SAVE_DIR)
