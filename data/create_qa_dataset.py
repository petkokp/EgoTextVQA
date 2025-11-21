from datasets import (
    load_dataset,
    Dataset,
    Features,
    Value,
    ClassLabel,
    Video,
    concatenate_datasets,
)
from build_video_index import build_video_index

BASE_DATASET = "ShengZhou97/EgoTextVQA"

INDOOR_JSONL = "EgoTextVQA_indoor_annotation.jsonl"
OUTDOOR_JSONL = "EgoTextVQA_outdoor_annotation.jsonl"

HF_REPO_ID = "petkopetkov/EgoTextVQA"

LOCAL_SAVE_DIR = "EgoTextVQA_video_qa"

def load_annotations():
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
    features = Features(
        {
            "video": Video(),
            "video_id": Value("string"),
            "subset": ClassLabel(names=["EgoTextVQA-Indoor", "EgoTextVQA-Outdoor"]),
            "question_id": Value("string"),
            "question_type": Value("string"),
            "timestamp": Value("float32"),
            "question": Value("string"),
            "answer": Value("string"),
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
                "video": base_ex["video"],
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

    print("[INFO] Creating metadata-only dataset (dropping 'video' column) for Hub upload...")
    meta_ds = qa_ds.remove_columns(["video"])
    print(meta_ds)
    print(f"[INFO] Pushing metadata-only dataset to Hub: {HF_REPO_ID}")
    meta_ds.push_to_hub(HF_REPO_ID)
    print("[INFO] Metadata push done.")

    print("[INFO] Done. For reconstruction on another machine, call reconstruct_qa_from_hub().")

if __name__ == "__main__":
    main()
