import os
from typing import Tuple
import random
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict, ClassLabel

QA_DATASET_META_HF = "petkopetkov/EgoTextVQA"

PROCESSED_VIDEO_ROOT = "./data/EgoTextVQA_fps6_lowres"

SPLIT_SAVE_DIR = "EgoTextVQA_video_qa_splits"

def _load_raw_qa_dataset() -> Dataset:
    print(f"[INFO] Loading QA dataset from Hub: {QA_DATASET_META_HF}")
    ds = load_dataset(QA_DATASET_META_HF, split="train")
    print(ds)
    return ds


def _normalize_subset(example, subset_feature=None):
    raw = example["subset"]
    if subset_feature is not None and isinstance(subset_feature, ClassLabel):
        example["subset"] = subset_feature.int2str(raw)
    else:
        example["subset"] = str(raw)
    return example


def _add_local_video_path(example, processed_root: str):
    subset_name = "EgoTextVQA-Indoor" if example["subset"] == 0 else "EgoTextVQA-Outdoor"
    video_id = str(example["video_id"])
    example["local_video_path"] = os.path.abspath(os.path.join(processed_root, subset_name, f"{video_id}.mp4"))
    return example

def _filter_existing_videos(example):
    path = example["local_video_path"]
    return os.path.exists(path)


def split_by_video_id(
    dataset: Dataset,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Dataset]:
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
    save_splits: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    ds = _load_raw_qa_dataset()

    subset_feature = ds.features.get("subset", None)
    ds = ds.map(
        lambda ex: _normalize_subset(ex, subset_feature=subset_feature),
        desc="Normalizing subset as string",
    )

    ds = ds.map(
        lambda ex: _add_local_video_path(ex, processed_root=processed_video_root),
        desc="Adding local_video_path",
    )
    
    for example in ds:
        _add_local_video_path(example, processed_root=processed_video_root)

    ds = ds.filter(_filter_existing_videos, desc="Filtering examples with missing local_video_path")
    print(f"[INFO] After filtering: {len(ds)} examples remain.")

    train_ds, val_ds, test_ds = split_by_video_id(ds)

    if save_splits:
        os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
        splits = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
        print(f"[INFO] Saving splits to disk at: {SPLIT_SAVE_DIR}")
        splits.save_to_disk(SPLIT_SAVE_DIR)

    return train_ds, val_ds, test_ds


def load_splits() -> DatasetDict:
    if not os.path.exists(SPLIT_SAVE_DIR):
        raise FileNotFoundError(
            f"Split folder '{SPLIT_SAVE_DIR}' not found. "
            f"Run a training script that calls load_data(..., save_splits=True) first."
        )
    print(f"[INFO] Loading splits from disk: {SPLIT_SAVE_DIR}")
    return load_from_disk(SPLIT_SAVE_DIR)
