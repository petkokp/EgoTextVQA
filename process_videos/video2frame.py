import cv2
import os
from pathlib import Path
from datasets import load_dataset, Video, ClassLabel

BASE_DATASET = "ShengZhou97/EgoTextVQA"
INPUT_ROOT = "../data/egotextvqa_fps6_lowres"
OUTPUT_ROOT = "../data/egotextvqa_fps6_frames"
MIN_FRAMES = 1080

def extract_frames(video_path, output_folder, min_frames=1080):
    if not os.path.exists(video_path):
        print(f"Warning: Video file {video_path} does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    existing_files = [
        f for f in os.listdir(output_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(existing_files) >= min_frames:
        print(f"{os.path.basename(video_path)} already processed ({len(existing_files)} frames).")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {os.path.basename(video_path)} | Original FPS: {fps}")

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Frames extracted and saved in: {output_folder}")

def process_videos(input_folder, output_folder, min_frames=1080):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist.")
        return

    video_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    if not video_files:
        print("No video files found in the input folder.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_frame_folder = os.path.join(
            output_folder, os.path.splitext(video_file)[0]
        )
        extract_frames(video_path, video_frame_folder, min_frames)

def _subset_from_label(label_value, label_feature):
    """Decode 'label' -> 'EgoTextVQA-Indoor' / 'EgoTextVQA-Outdoor'."""
    if isinstance(label_feature, ClassLabel):
        return label_feature.int2str(label_value)
    return str(label_value)


def extract_frames_from_dataset(
    base_dataset_name=BASE_DATASET,
    split="train",
    input_root=INPUT_ROOT,
    output_root=OUTPUT_ROOT,
    min_frames=MIN_FRAMES,
    subset_filter=None,
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
        if not os.path.exists(input_video_path):
            print(
                f"[WARN] Processed video not found for {subset}/{video_id}: "
                f"{input_video_path}"
            )
            continue

        video_frame_folder = os.path.join(output_root, subset, video_id)

        print(
            f"[{idx + 1}/{len(base)}] Extracting frames from video_id={video_id}, "
            f"subset={subset}"
        )
        extract_frames(input_video_path, video_frame_folder, min_frames=min_frames)

if __name__ == "__main__":
    extract_frames_from_dataset(
        base_dataset_name=BASE_DATASET,
        split="train",
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        min_frames=MIN_FRAMES,
        subset_filter=None,
    )
