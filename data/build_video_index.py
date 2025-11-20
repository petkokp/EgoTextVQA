from pathlib import Path
from datasets import (
    ClassLabel,
)

def get_subset_from_label(ex, label_feature):
    """
    Robustly decode the subset (EgoTextVQA-Indoor/Outdoor) from the 'label' column.

    - If label is a ClassLabel (int), convert to string via int2str.
    - If label is already a string, just use it.
    """
    label_value = ex["label"]
    if isinstance(label_feature, ClassLabel):
        return label_feature.int2str(label_value)
    return str(label_value)

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