import cv2
import os
from pathlib import Path

from datasets import load_dataset, Video, ClassLabel

BASE_DATASET = "ShengZhou97/EgoTextVQA"
INPUT_ROOT = "../data/egotextvqa_fps6"
OUTPUT_ROOT = "../data/egotextvqa_fps6_lowres"

SCALE_FACTOR = 0.5

def resize_video(input_video_path, output_video_path, scale_factor=0.5):
    if not os.path.exists(input_video_path):
        print(f"[WARN] Input video does not exist: {input_video_path}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {input_video_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if original_width <= 0 or original_height <= 0:
        print(f"[WARN] Invalid video dimensions for {input_video_path}, skipping.")
        cap.release()
        return

    new_width = max(1, int(original_width * scale_factor))
    new_height = max(1, int(original_height * scale_factor))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"[INFO] Video processing completed: {output_video_path}")

def resize_folder_videos(input_folder, output_folder, scale_factor=0.5):
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder {input_folder} does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    video_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print("[INFO] No video files found in the input folder.")
        return

    for video_filename in video_files:
        input_video_path = os.path.join(input_folder, video_filename)
        output_video_path = os.path.join(output_folder, video_filename)
        resize_video(input_video_path, output_video_path, scale_factor)

def _subset_from_label(label_value, label_feature):
    if isinstance(label_feature, ClassLabel):
        return label_feature.int2str(label_value)
    return str(label_value)


def resize_dataset_videos(
    base_dataset_name=BASE_DATASET,
    split="train",
    input_root=INPUT_ROOT,
    output_root=OUTPUT_ROOT,
    scale_factor=SCALE_FACTOR,
    subset_filter=None,
    overwrite=False,
):
    print(f"[INFO] Loading base dataset: {base_dataset_name} (split={split})")
    base = load_dataset(base_dataset_name, split=split)

    print(base)
    print("[INFO] Example base row:", base[0])
    print("[INFO] Casting 'video' column to Video(decode=False) for path access...")
    base = base.cast_column("video", Video(decode=False))

    label_feature = base.features["label"]

    os.makedirs(output_root, exist_ok=True)

    for idx, ex in enumerate(base):
        video_info = ex["video"]
        if not isinstance(video_info, dict) or "path" not in video_info:
            print(f"[WARN] Example {idx} has invalid video info; skipping.")
            continue

        subset = _subset_from_label(ex["label"], label_feature)
        if subset_filter is not None and subset not in subset_filter:
            continue

        video_id = Path(video_info["path"]).stem

        input_video_path = os.path.join(input_root, subset, f"{video_id}.mp4")
        output_video_path = os.path.join(output_root, subset, f"{video_id}.mp4")

        if not os.path.exists(input_video_path):
            print(f"[WARN] Input video not found for {subset}/{video_id}: {input_video_path}")
            continue

        if not overwrite and os.path.exists(output_video_path):
            print(f"[INFO] Skipping (already exists): {output_video_path}")
            continue

        print(
            f"[{idx + 1}/{len(base)}] Resizing video_id={video_id}, subset={subset}, "
            f"scale_factor={scale_factor}"
        )
        resize_video(input_video_path, output_video_path, scale_factor)

if __name__ == "__main__":
    resize_dataset_videos(
        base_dataset_name=BASE_DATASET,
        split="train",
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        scale_factor=SCALE_FACTOR,
        subset_filter=None,
        overwrite=False,
    )
