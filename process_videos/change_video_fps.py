import cv2
import os
from pathlib import Path
from datasets import load_dataset, Video
from data.build_video_index import get_subset_from_label

BASE_DATASET = "ShengZhou97/EgoTextVQA"

OUTPUT_ROOT = "../data/EgoTextVQA_fps6"

TARGET_FPS = 6

def process_video(input_path, output_path, target_fps=6):
    if not os.path.exists(input_path):
        print(f"Warning: Input video {input_path} does not exist.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video {input_path}.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frame_interval = max(1, int(original_fps / target_fps)) if original_fps > 0 else 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")

def process_video_folder(input_folder, output_folder, target_fps=6):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    video_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    if not video_files:
        print("No video files found in the input folder.")
        return

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        process_video(input_path, output_path, target_fps)

def process_dataset_videos(
    base_dataset_name=BASE_DATASET,
    split="train",
    output_root=OUTPUT_ROOT,
    target_fps=TARGET_FPS,
    overwrite=False,
):
    print(f"[INFO] Loading base dataset: {base_dataset_name} (split={split})")
    base = load_dataset(base_dataset_name, split=split)

    print(base)
    print("[INFO] Example base row:", base[0])
    print("[INFO] Casting 'video' column to Video(decode=False) to access file paths...")
    base = base.cast_column("video", Video(decode=False))

    label_feature = base.features["label"]

    os.makedirs(output_root, exist_ok=True)

    for idx, ex in enumerate(base):
        video_info = ex["video"]
        if not isinstance(video_info, dict) or "path" not in video_info:
            print(f"[WARN] Example {idx} has invalid video info; skipping.")
            continue

        input_path = video_info["path"]
        subset = get_subset_from_label(ex["label"], label_feature)
        video_id = Path(input_path).stem

        subset_folder = os.path.join(output_root, subset)
        os.makedirs(subset_folder, exist_ok=True)

        output_path = os.path.join(subset_folder, f"{video_id}.mp4")

        if not overwrite and os.path.exists(output_path):
            print(f"[INFO] Skipping {output_path} (already exists).")
            continue

        print(
            f"[{idx + 1}/{len(base)}] Processing video_id={video_id}, subset={subset}"
        )
        process_video(input_path, output_path, target_fps)

if __name__ == "__main__":
    process_dataset_videos(
        base_dataset_name=BASE_DATASET,
        split="train",
        output_root=OUTPUT_ROOT,
        target_fps=TARGET_FPS,
        overwrite=False,
    )
