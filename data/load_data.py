import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)

QA_DATASET_META_HF = "petkopetkov/EgoTextVQA"

PROCESSED_VIDEO_ROOT = "./data/EgoTextVQA_fps6_lowres"
EGOTEMPO_JSON_PATH = "./data/egotempo_openQA.json"
EGOTEMPO_VIDEO_ROOT = "./data/trimmed_clips"

SPLIT_SAVE_DIR = "EgoTextVQA_video_qa_splits"
EGOTEMPO_SPLIT_SAVE_DIR = "EgoTempo_video_qa_splits_TEST_ONLY"

CORRUPTED_VIDEOS = ["0e0d6704-1c6c-4a62-bc97-cc55658cf8ac_3443.2712334842913_3446.397589849042", "04041aaa-d309-42db-b65c-dcaf86b9f96c_141.06121484275377_145.05766515724622"]

COMMON_COLUMNS = [
    "question_id",
    "video_id",
    "question",
    "answer",
    "question_type",
    "timestamp",
    "local_video_path",
    "dataset",
    "subset",
]


@dataclass
class DatasetSpec:
    name: str
    weight: float = 1.0
    max_samples: Optional[int] = None


def _normalize_name(name: str) -> str:
    return name.lower()


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
    subset_value = example["subset"]
    is_indoor = subset_value in [0, "0", "EgoTextVQA-Indoor"]
    subset_name = "EgoTextVQA-Indoor" if is_indoor else "EgoTextVQA-Outdoor"
    video_id = str(example["video_id"])
    example["local_video_path"] = os.path.abspath(os.path.join(processed_root, subset_name, f"{video_id}.mp4"))
    return example


def _filter_existing_videos(example):
    path = example["local_video_path"]
    return os.path.exists(path)


def _split_by_id(
    dataset: Dataset,
    id_key: str = "video_id",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Dataset]:
    ids = list(set(dataset[id_key]))
    random.Random(seed).shuffle(ids)

    n = len(ids)
    train_cut = int(train_ratio * n)
    val_cut = int((train_ratio + val_ratio) * n)

    train_ids = set(ids[:train_cut])
    val_ids = set(ids[train_cut:val_cut])
    test_ids = set(ids[val_cut:])

    print(f"[INFO] #{id_key} train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_ds = dataset.filter(lambda ex: ex[id_key] in train_ids)
    val_ds = dataset.filter(lambda ex: ex[id_key] in val_ids)
    test_ds = dataset.filter(lambda ex: ex[id_key] in test_ids)

    return train_ds, val_ds, test_ds


def _standardize_schema(ds: Dataset, dataset_name: str) -> Dataset:
    columns_to_remove = [c for c in ds.column_names if c not in COMMON_COLUMNS]

    def _convert(example):
        return {
            "question_id": str(example.get("question_id", "")),
            "video_id": str(example.get("video_id", "")),
            "question": example.get("question", ""),
            "answer": example.get("answer", ""),
            "question_type": example.get("question_type", ""),
            "timestamp": float(example.get("timestamp", 0.0) or 0.0),
            "local_video_path": example["local_video_path"],
            "dataset": dataset_name,
            "subset": example.get("subset", dataset_name),
        }

    return ds.map(
        _convert,
        remove_columns=columns_to_remove,
        desc=f"Standardizing schema for {dataset_name}",
    )


def _load_egotext_dataset(processed_video_root: str) -> Dataset:
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

    ds = ds.filter(_filter_existing_videos, desc="Filtering examples with missing local_video_path")
    print(f"[INFO] After filtering: {len(ds)} EgoTextVQA examples remain.")

    ds = _standardize_schema(ds, dataset_name="EgoTextVQA")
    return ds


def _load_egotempo_dataset(json_path: str, video_root: str) -> Dataset:
    json_path = os.path.abspath(json_path)
    video_root = os.path.abspath(video_root)

    with open(json_path, "r") as f:
        payload = json.load(f)

    annotations = payload["annotations"] if "annotations" in payload else payload

    examples: List[Dict] = []
    for row in annotations:
        clip_id = str(row["clip_id"]).replace(".mp4", "")
        local_video_path = os.path.abspath(os.path.join(video_root, f"{clip_id}.mp4"))
        print("local video path: ", local_video_path)
        print("row: ", row)
            
        examples.append(
            {
                "question_id": str(row.get("question_id", clip_id)),
                "video_id": clip_id,
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row.get("question_type", ""),
                "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                "local_video_path": local_video_path,
                "dataset": "EgoTempo",
                "subset": "EgoTempo",
            }
        )

    ds = Dataset.from_list(examples)
    ds = ds.filter(_filter_existing_videos, desc="Filtering EgoTempo examples with missing local_video_path")
    print(f"[INFO] Loaded EgoTempo with {len(ds)} examples after filtering.")
    ds = _standardize_schema(ds, dataset_name="EgoTempo")
    return ds


def _load_dataset_splits(
    dataset_name: str,
    processed_video_root: str,
    egotempo_json_path: str,
    egotempo_video_root: str,
    save_splits: bool,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> DatasetDict:
    dataset_name = _normalize_name(dataset_name)
    
    print("dataset name after normalization: ", dataset_name)

    if dataset_name == "egotextvqa":
        split_dir = SPLIT_SAVE_DIR
        base_loader = lambda: _load_egotext_dataset(processed_video_root)
        id_key = "video_id"
        label_name = "EgoTextVQA"
    elif dataset_name == "egotempo":
        split_dir = EGOTEMPO_SPLIT_SAVE_DIR
        base_loader = lambda: _load_egotempo_dataset(
            json_path=egotempo_json_path,
            video_root=egotempo_video_root,
        )
        id_key = "video_id"
        label_name = "EgoTempo"
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    if os.path.exists(split_dir):
        print(f"[INFO] Loading cached splits for {dataset_name} from {split_dir}")
        splits = load_from_disk(split_dir)
        return DatasetDict(
            {
                "train": _standardize_schema(splits["train"], dataset_name=label_name),
                "validation": _standardize_schema(splits["validation"], dataset_name=label_name),
                "test": _standardize_schema(splits["test"], dataset_name=label_name),
            }
        )

    ds = base_loader()
    train_ds, val_ds, test_ds = _split_by_id(
        ds,
        id_key=id_key,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    splits = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    if save_splits:
        os.makedirs(split_dir, exist_ok=True)
        print(f"[INFO] Saving {dataset_name} splits to disk at: {split_dir}")
        splits.save_to_disk(split_dir)

    return splits


def _as_dataset_specs(names: Sequence[str], weights: Optional[Sequence[float]]) -> List[DatasetSpec]:
    weights = list(weights) if weights is not None else []
    specs: List[DatasetSpec] = []
    for idx, name in enumerate(names):
        weight = weights[idx] if idx < len(weights) else 1.0
        specs.append(DatasetSpec(name=_normalize_name(name), weight=weight))
    return specs


def _build_split_from_specs(
    dataset_specs: List[DatasetSpec],
    dataset_splits: Dict[str, DatasetDict],
    split_name: str,
    seed: int,
) -> Dataset:
    selected = []
    probs = []
    for spec in dataset_specs:
        ds = dataset_splits[spec.name][split_name]
        if spec.max_samples is not None:
            limit = min(spec.max_samples, len(ds))
            ds = ds.shuffle(seed=seed).select(range(limit))
        selected.append(ds)
        probs.append(spec.weight)

    if len(selected) == 1:
        return selected[0]

    if sum(probs) == 0:
        probs = [1.0] * len(selected)
    total = sum(probs)
    probs = [p / total for p in probs]
    return interleave_datasets(selected, probabilities=probs, seed=seed)


def load_data(
    train_datasets: Sequence[str] = ("egotextvqa",),
    val_datasets: Optional[Sequence[str]] = None,
    test_datasets: Optional[Sequence[str]] = None,
    train_weights: Optional[Sequence[float]] = None,
    val_weights: Optional[Sequence[float]] = None,
    test_weights: Optional[Sequence[float]] = None,
    processed_video_root: str = PROCESSED_VIDEO_ROOT,
    egotempo_json_path: str = EGOTEMPO_JSON_PATH,
    egotempo_video_root: str = EGOTEMPO_VIDEO_ROOT,
    save_splits: bool = True,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Dataset]:
    val_datasets = val_datasets or train_datasets
    test_datasets = test_datasets or val_datasets

    train_specs = _as_dataset_specs(train_datasets, train_weights)
    val_specs = _as_dataset_specs(val_datasets, val_weights)
    test_specs = _as_dataset_specs(test_datasets, test_weights)

    dataset_splits: Dict[str, DatasetDict] = {}
    for spec in train_specs + val_specs + test_specs:
        if spec.name not in dataset_splits:
            dataset_splits[spec.name] = _load_dataset_splits(
                dataset_name=spec.name,
                processed_video_root=processed_video_root,
                egotempo_json_path=egotempo_json_path,
                egotempo_video_root=egotempo_video_root,
                save_splits=save_splits,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )

    train_ds = _build_split_from_specs(train_specs, dataset_splits, split_name="train", seed=seed)
    val_ds = _build_split_from_specs(val_specs, dataset_splits, split_name="validation", seed=seed)
    test_ds = _build_split_from_specs(test_specs, dataset_splits, split_name="test", seed=seed)

    print(f"[INFO] Train / Val / Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    return train_ds, val_ds, test_ds


def load_splits(dataset_name: str = "egotextvqa") -> DatasetDict:
    dataset_name = _normalize_name(dataset_name)
    if dataset_name == "egotextvqa":
        split_dir = SPLIT_SAVE_DIR
        label_name = "EgoTextVQA"
    elif dataset_name == "egotempo":
        split_dir = EGOTEMPO_SPLIT_SAVE_DIR
        label_name = "EgoTempo"
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    if not os.path.exists(split_dir):
        raise FileNotFoundError(
            f"Split folder '{split_dir}' not found. "
            f"Run a training script that calls load_data(..., save_splits=True) first."
        )
    print(f"[INFO] Loading splits from disk: {split_dir}")
    splits = load_from_disk(split_dir)
    return DatasetDict(
        {
            "train": _standardize_schema(splits["train"], dataset_name=label_name),
            "validation": _standardize_schema(splits["validation"], dataset_name=label_name),
            "test": _standardize_schema(splits["test"], dataset_name=label_name),
        }
    )
