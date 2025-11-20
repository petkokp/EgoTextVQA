from datasets import (
    load_dataset,
    Dataset,
    Features,
    Value,
    ClassLabel,
    Video,
)
from .build_video_index import build_video_index

def build_qa_dataset_from_metadata(base, metadata):
    """
    Reconstruct a full video+QA dataset from:
      - base: the original video dataset (ShengZhou97/EgoTextVQA)
      - metadata: a dataset WITHOUT video column (like the one on the Hub)

    This is what you'd use in a new environment after doing:

        meta = load_dataset(HF_REPO_ID, split="train")
        base = load_dataset(BASE_DATASET, split="train")

    and calling this function.
    """
    # Ensure base has decode=False so build_video_index works
    base = base.cast_column("video", Video(decode=False))

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

    subset_feature = metadata.features["subset"]
    subset_is_classlabel = isinstance(subset_feature, ClassLabel)

    new_examples = []
    missing = 0

    for row in metadata:
        # Convert subset back to string if it's stored as ClassLabel index
        raw_subset = row["subset"]
        if subset_is_classlabel:
            subset = subset_feature.int2str(raw_subset)
        else:
            subset = str(raw_subset)

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
                "answer": row["answer"],
                "video_url": row.get("video_url", ""),
            }
        )

    if missing > 0:
        print(f"[INFO] Reconstruct: skipped {missing} metadata rows with missing videos.")

    qa_ds = Dataset.from_list(new_examples, features=features)
    return qa_ds