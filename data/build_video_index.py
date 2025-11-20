from pathlib import Path
from datasets import (
    ClassLabel,
)

def get_subset_from_label(ex, label_feature):
    label_value = ex["label"]
    if isinstance(label_feature, ClassLabel):
        return label_feature.int2str(label_value)
    return str(label_value)

def build_video_index(base):
    id2idx = {}
    label_feature = base.features["label"]

    for i, ex in enumerate(base):
        video_info = ex["video"]

        if isinstance(video_info, dict) and "path" in video_info:
            video_path = video_info["path"]
        else:
            raise TypeError(
                "Expected 'video' column to be a dict with a 'path' key. "
                "Make sure you've cast the column with Video(decode=False), e.g.: "
                "base = base.cast_column('video', Video(decode=False)) "
                "before calling build_video_index()."
            )

        video_id = Path(video_path).stem
        subset = get_subset_from_label(ex, label_feature)
        key = (subset, video_id)

        if key in id2idx:
            # if this ever happens, it means duplicated (subset, video_id), we just keep the first
            continue

        id2idx[key] = i

    return id2idx