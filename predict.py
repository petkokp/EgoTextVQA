import argparse
import json
import os
from typing import Dict, Tuple

import torch
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from transformers import AutoProcessor
from unsloth import FastVisionModel

from data.load_data import (
    EGOTEMPO_JSON_PATH,
    EGOTEMPO_VIDEO_ROOT,
    PROCESSED_VIDEO_ROOT,
    load_data,
)

DEFAULT_MODEL_PATH = "/home/petko/projects/EgoTextVQA/checkpoints/Qwen3-VL-2B-Instruct/checkpoint-345"
DEFAULT_PRED_KEY = "gemini-2.5-flash"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions on EgoTextVQA / EgoTempo splits.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_PATH, help="Model identifier or local checkpoint path.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["egotextvqa"],
        help="Datasets to run inference on (choices: egotextvqa, egotempo).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Probabilities for interleaving the requested datasets.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to run inference on.",
    )
    parser.add_argument("--egotext-root", default=PROCESSED_VIDEO_ROOT, help="Path to processed EgoTextVQA videos.")
    parser.add_argument("--egotempo-json", default=EGOTEMPO_JSON_PATH, help="Path to the EgoTempo QA JSON file.")
    parser.add_argument("--egotempo-root", default=EGOTEMPO_VIDEO_ROOT, help="Path to trimmed EgoTempo clips.")
    parser.add_argument("--train-ratio", type=float, default=0, help="Train split ratio used for cached splits.")
    parser.add_argument("--val-ratio", type=float, default=1, help="Validation split ratio used for cached splits.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the cached train/val/test splits.")
    parser.add_argument("--save-splits", action="store_true", default=True, help="Persist dataset splits if missing.")
    parser.add_argument("--no-save-splits", dest="save_splits", action="store_false", help="Do not persist dataset splits.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens for generation.")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=1.0,
        help="Seconds after the timestamp to keep for EgoTextVQA clips (EgoTempo uses full clips).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to store predictions JSON. Defaults to results/<model>_<datasets>_<split>_predictions.json",
    )
    parser.add_argument(
        "--pred-key",
        default=DEFAULT_PRED_KEY,
        help="Key used to store model predictions inside the output JSON.",
    )
    return parser.parse_args()


def max_new_tokens(model_id: str, override: int = None) -> int:
    if override is not None:
        return override
    return 2048 if "Thinking" in model_id else 1024


def normalize_video_path(path: str, dataset: str) -> str:
    if dataset.lower() == "egotextvqa":
        path = path.replace("egotextvqa", "EgoTextVQA")
        path = path.replace("egocentricvqa", "EgoTextVQA")
    return path


def prepare_clip(sample: Dict, clip_seconds: float) -> Tuple[str, bool]:
    dataset = sample.get("dataset", "")
    video_path = normalize_video_path(sample["local_video_path"], dataset)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    if dataset.lower() != "egotextvqa":
        return video_path, False

    timestamp = float(sample.get("timestamp", 0.0) or 0.0)
    os.makedirs("temp", exist_ok=True)
    temp_video_path = os.path.join("temp", f"{sample['video_id']}_{timestamp:.1f}.mp4")

    clip = VideoFileClip(video_path)
    try:
        end_time = min(clip.duration, max(0.0, timestamp) + clip_seconds)
        subclip = clip.subclip(0, end_time)
        try:
            subclip.write_videofile(temp_video_path, codec="libx264", audio=False, logger=None)
        finally:
            subclip.close()
    finally:
        clip.close()

    return temp_video_path, True


def load_eval_split(args):
    train_ds, val_ds, test_ds = load_data(
        train_datasets=args.datasets,
        val_datasets=args.datasets,
        test_datasets=args.datasets,
        train_weights=args.weights,
        val_weights=args.weights,
        test_weights=args.weights,
        processed_video_root=args.egotext_root,
        egotempo_json_path=args.egotempo_json,
        egotempo_video_root=args.egotempo_root,
        save_splits=args.save_splits,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    splits = {"train": train_ds, "validation": val_ds, "test": test_ds}
    return splits[args.split]


def main():
    args = parse_args()

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_id,
        full_finetuning=True,
    )
    FastVisionModel.for_inference(model)
    processor = AutoProcessor.from_pretrained(args.model_id)

    target_split = load_eval_split(args)
    print(f"[INFO] Loaded {args.split} split with {len(target_split)} examples from {args.datasets}")

    max_tokens = max_new_tokens(args.model_id, args.max_new_tokens)
    print(f"[INFO] Using max_new_tokens={max_tokens}")

    output_path = args.output
    if output_path is None:
        dataset_tag = "-".join(args.datasets)
        model_tag = os.path.basename(args.model_id.rstrip("/"))
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", f"{model_tag}_{dataset_tag}_{args.split}_predictions.json")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    predictions = []
    for sample in tqdm(target_split):
        video_path = None
        should_cleanup = False
        try:
            video_path, should_cleanup = prepare_clip(sample, args.clip_seconds)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": sample["question"]},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]) :]
            pred_answer = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

            predictions.append(
                {
                    "question_id": sample["question_id"],
                    "video_id": sample["video_id"],
                    "question": sample["question"],
                    "question_type": sample["question_type"],
                    "dataset": sample.get("dataset", ""),
                    "subset": sample.get("subset", ""),
                    args.pred_key: pred_answer,
                    "correct_answer": sample["answer"],
                }
            )

        except Exception as e:
            print(f"Error processing {sample.get('question_id', 'N/A')}: {e}")
        finally:
            if should_cleanup and video_path and os.path.exists(video_path):
                os.remove(video_path)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"[INFO] Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
