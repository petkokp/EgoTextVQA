from pathlib import Path

from datasets import (
    load_dataset,
    Dataset,
    Features,
    Value,
    ClassLabel,
    Video,
    concatenate_datasets,
)
from build_video_index import get_subset_from_label

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Base video dataset (2 cols: 'video', 'label')
BASE_DATASET = "ShengZhou97/EgoTextVQA"

# Local annotation JSONL files
INDOOR_JSONL = "egotextvqa_indoor_annotation.jsonl"
OUTDOOR_JSONL = "egotextvqa_outdoor_annotation.jsonl"

# Hub repo where we'll push **metadata only** (no video column)
HF_REPO_ID = "petkopetkov/EgoTextVQA"  # adjust if you want a different name

# Local path to save the full video+QA dataset
LOCAL_SAVE_DIR = "egotextvqa_video_qa"

# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def build_video_index(base):
    """
    Build a mapping: (subset, video_id) -> index in the base dataset.

    We assume that the path of base['video'] looks like:
    .../EgoTextVQA-Indoor/0479bea8-d221-4c6a-8c91-60108e43fe56.mp4
    so video_id is the filename stem.

    IMPORTANT: this expects the 'video' column to be stored with decode=False,
    so each example has:
        ex["video"] == {"bytes": None, "path": ".../xxxx.mp4"}
    """
    id2idx = {}
    label_feature = base.features["label"]

    for i, ex in enumerate(base):
        video_info = ex["video"]

        if isinstance(video_info, dict) and "path" in video_info:
            video_path = video_info["path"]
        else:
            # This means we forgot to cast the column with Video(decode=False)
            raise TypeError(
                "Expected 'video' column to be a dict with a 'path' key. "
                "Make sure you've cast the column with Video(decode=False), e.g.: "
                "base = base.cast_column('video', Video(decode=False)) "
                "before calling build_video_index()."
            )

        video_id = Path(video_path).stem  # "0479bea8-d221-4c6a-8c91-60108e43fe56"
        subset = get_subset_from_label(ex, label_feature)  # "EgoTextVQA-Indoor"/"Outdoor"
        key = (subset, video_id)

        if key in id2idx:
            # If this ever happens, it means duplicated (subset, video_id).
            # We just keep the first, but you could assert here instead.
            continue

        id2idx[key] = i

    return id2idx


def load_annotations():
    """
    Load indoor & outdoor JSONL annotations and add a 'subset' column
    to each so we can match to the base dataset.
    """
    ann_indoor = load_dataset(
        "json",
        data_files=INDOOR_JSONL,
        split="train",
    )
    ann_outdoor = load_dataset(
        "json",
        data_files=OUTDOOR_JSONL,
        split="train",
    )

    ann_indoor = ann_indoor.add_column(
        "subset", ["EgoTextVQA-Indoor"] * len(ann_indoor)
    )
    ann_outdoor = ann_outdoor.add_column(
        "subset", ["EgoTextVQA-Outdoor"] * len(ann_outdoor)
    )

    annotations = concatenate_datasets([ann_indoor, ann_outdoor])
    return annotations


def build_qa_dataset(base, annotations):
    """
    Join base video dataset with QA annotations to produce
    a per-QA dataset with schema suitable for VLM finetuning.

    This returns a dataset with a **Video** column, intended for local training.
    """
    features = Features(
        {
            "video": Video(),  # HF Video feature
            "video_id": Value("string"),
            "subset": ClassLabel(names=["EgoTextVQA-Indoor", "EgoTextVQA-Outdoor"]),
            "question_id": Value("string"),
            "question_type": Value("string"),
            "timestamp": Value("float32"),
            "question": Value("string"),
            "answer": Value("string"),
            # Keep video_url around so the metadata on Hub can point to the file
            "video_url": Value("string"),
        }
    )

    id2idx = build_video_index(base)

    new_examples = []
    missing = 0

    for row in annotations:
        subset = row["subset"]
        video_id = row["video_id"]
        key = (subset, video_id)

        if key not in id2idx:
            missing += 1
            print(f"[WARN] Missing video for {key}")
            continue

        base_idx = id2idx[key]
        base_ex = base[base_idx]

        new_examples.append(
            {
                "video": base_ex["video"],  # keep as Video feature
                "video_id": video_id,
                "subset": subset,
                "question_id": str(row["question_id"]),
                "question_type": row.get("question_type", ""),
                "timestamp": float(row["timestamp"]),
                "question": row["question"],
                "answer": row["correct_answer"],
                "video_url": row.get("video_url", ""),
            }
        )

    if missing > 0:
        print(f"[INFO] Skipped {missing} annotation(s) with missing videos.")

    qa_ds = Dataset.from_list(new_examples, features=features)
    return qa_ds

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    print(f"[INFO] Loading base dataset: {BASE_DATASET}")
    base = load_dataset(BASE_DATASET, split="train")

    print(base)
    print("[INFO] Example base row:", base[0])

    print("[INFO] Casting 'video' column to Video(decode=False) to access file paths...")
    base = base.cast_column("video", Video(decode=False))

    print("[INFO] Loading annotation JSONLs...")
    annotations = load_annotations()
    print(annotations)
    print("[INFO] Example annotation row:", annotations[0])

    print("[INFO] Building per-QA dataset (with video)...")
    qa_ds = build_qa_dataset(base, annotations)
    print(qa_ds)
    print("[INFO] Example QA row:", qa_ds[0])

    # ---- IMPORTANT PART: upload metadata-only dataset to the Hub ----
    print("[INFO] Creating metadata-only dataset (dropping 'video' column) for Hub upload...")
    meta_ds = qa_ds.remove_columns(["video"])
    print(meta_ds)
    print(f"[INFO] Pushing metadata-only dataset to Hub: {HF_REPO_ID}")
    meta_ds.push_to_hub(HF_REPO_ID)
    print("[INFO] Metadata push done.")

    print("[INFO] Done. For reconstruction on another machine, call reconstruct_qa_from_hub().")


if __name__ == "__main__":
    main()
