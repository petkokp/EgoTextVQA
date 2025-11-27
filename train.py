from unsloth import FastVisionModel
import argparse
import math
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, TrainingArguments, Trainer
from qwen_vl_utils import process_vision_info

from data.load_data import (
    EGOTEMPO_JSON_PATH,
    EGOTEMPO_VIDEO_ROOT,
    PROCESSED_VIDEO_ROOT,
    CORRUPTED_VIDEOS,
    load_data,
)

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune VLMs on EgoTextVQA / EgoTempo.")
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help='Model identifier (e.g. "unsloth/Qwen3-VL-2B-Instruct" or "HuggingFaceTB/SmolVLM2-500M-Video-Instruct").',
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=["egotextvqa"],
        help="Datasets to mix for training (choices: egotextvqa, egotempo).",
    )
    parser.add_argument(
        "--val-datasets",
        nargs="+",
        default=None,
        help="Datasets to mix for validation. Defaults to --train-datasets.",
    )
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        default=None,
        help="Datasets to mix for test. Defaults to --val-datasets.",
    )
    parser.add_argument(
        "--train-weights",
        nargs="+",
        type=float,
        default=None,
        help="Probabilities for interleaving the training datasets (same order as --train-datasets).",
    )
    parser.add_argument(
        "--val-weights",
        nargs="+",
        type=float,
        default=None,
        help="Probabilities for interleaving the validation datasets.",
    )
    parser.add_argument(
        "--test-weights",
        nargs="+",
        type=float,
        default=None,
        help="Probabilities for interleaving the test datasets.",
    )
    parser.add_argument(
        "--egotext-root",
        default=PROCESSED_VIDEO_ROOT,
        help="Local path to processed EgoTextVQA videos.",
    )
    parser.add_argument(
        "--egotempo-json",
        default=EGOTEMPO_JSON_PATH,
        help="Path to the EgoTempo QA JSON file.",
    )
    parser.add_argument(
        "--egotempo-root",
        default=EGOTEMPO_VIDEO_ROOT,
        help="Local path to trimmed EgoTempo clips.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad-accumulation", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the default LR for the chosen model.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio within each dataset.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio within each dataset.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for splits and shuffling.")
    parser.add_argument("--save-splits", action="store_true", default=True, help="Persist dataset splits to disk.")
    parser.add_argument("--no-save-splits", dest="save_splits", action="store_false", help="Do not persist dataset splits.")
    parser.add_argument(
        "--video-backend",
        default="torchvision",
        choices=["torchvision", "pyav", "decord", "opencv", "torchcodec"],
        help="Video decoding backend to use for models that rely on a BaseVideoProcessor (SmolVLM, etc.).",
    )
    return parser.parse_args()


def dynamic_steps(
    n_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    target_logs_per_epoch: int = 20,
    target_evals_per_epoch: int = 5,
    min_log_step: int = 1,
    min_eval_step: int = 10,
):
    dataloader_steps_per_epoch = math.ceil(n_examples / per_device_train_batch_size)
    update_steps_per_epoch = math.ceil(dataloader_steps_per_epoch / gradient_accumulation_steps)
    total_update_steps = update_steps_per_epoch * num_train_epochs

    logging_steps = max(min_log_step, max(1, update_steps_per_epoch // target_logs_per_epoch))
    eval_steps = max(min_eval_step, max(1, update_steps_per_epoch // target_evals_per_epoch))
    save_steps = eval_steps  # save whenever we eval
    warmup_steps = max(1, int(0.03 * total_update_steps))  # 3% warmup

    return {
        "steps_per_epoch": update_steps_per_epoch,
        "total_update_steps": total_update_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "warmup_steps": warmup_steps,
    }


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", "EgoTextVQA")

    model_id = args.model_id
    is_qwen_model = "Qwen" in model_id

    processor = AutoProcessor.from_pretrained(model_id)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        full_finetuning=True,
        use_gradient_checkpointing="unsloth",
    )

    FastVisionModel.for_training(model, use_gradient_checkpointing=False)
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"[INFO] The model as is is holding: {peak_mem / 1024**3:.2f} GB of GPU RAM")

    train_ds, val_ds, test_ds = load_data(
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        test_datasets=args.test_datasets,
        train_weights=args.train_weights,
        val_weights=args.val_weights,
        test_weights=args.test_weights,
        processed_video_root=args.egotext_root,
        egotempo_json_path=args.egotempo_json,
        egotempo_video_root=args.egotempo_root,
        save_splits=args.save_splits,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    
    train_ds = train_ds.filter(lambda x: not x['video_id'] in CORRUPTED_VIDEOS)

    if not len(train_ds):
        raise RuntimeError("Training dataset is empty. Check dataset paths and split config.")

    print(f"[INFO] Train / Val / Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    if not is_qwen_model:
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
    else:
        image_token_id = None

    def qwen_collate_fn(examples):
        texts = []
        video_inputs = []

        for example in examples:
            video_path = example["local_video_path"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": example["question"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["answer"]}],
                },
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            _, video_input = process_vision_info(messages)

            texts.append(text)
            video_inputs.append(video_input[0] if video_input else None)

        batch = processor(
            text=texts,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        if "video_token_id" in processor.tokenizer.get_vocab():
            video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            labels[labels == video_token_id] = -100

        batch["labels"] = labels
        return batch

    def smolvlm_collate_fn(examples):
        instances = []
        for example in examples:
            video_path = example["local_video_path"]

            user_content = [{"type": "text", "text": example["question"]}]
            user_content.append({"type": "video", "path": video_path, "fps": 1.0})

            messages = [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"{example['answer']}"}],
                },
            ]

            instance = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda").to(model.dtype)

            instances.append(instance)

        input_ids = pad_sequence(
            [inst["input_ids"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [inst["attention_mask"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [inst["input_ids"].squeeze(0).clone() for inst in instances],
            batch_first=True,
            padding_value=-100,
        )

        if image_token_id is not None:
            labels[labels == image_token_id] = -100  # ignore image tokens in loss

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # pad pixel_values (T, C, H, W)
        pvs = []
        for inst in instances:
            pv = inst.get("pixel_values", None)
            if pv is not None:
                pvs.append(pv.squeeze(0))

        if pvs:
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            # fallback if no videos
            max_frames = 1
            max_h = max_w = processor.video_size.get("longest_edge", 224)

        padded_pixel_values_list = []
        for inst in instances:
            pv = inst.get("pixel_values", None)
            if pv is None:
                padded_pv = torch.zeros(
                    (max_frames, 3, max_h, max_w),
                    dtype=torch.float32,
                    device="cuda",
                )
            else:
                pv = pv.squeeze(0)  # (T, C, H, W)
                f, c, h, w = pv.shape
                padded_pv = torch.zeros(
                    (max_frames, c, max_h, max_w),
                    dtype=pv.dtype,
                    device=pv.device,
                )
                padded_pv[:f, :, :h, :w] = pv
            padded_pixel_values_list.append(padded_pv)

        out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
        return out

    batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_accumulation
    num_train_epochs = args.epochs

    config = dynamic_steps(
        n_examples=len(train_ds),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    learning_rate = args.learning_rate if args.learning_rate is not None else (2e-4 if is_qwen_model else 1e-4)

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
        load_best_model_at_end=True,
        save_total_limit=1,
        bf16=True,
        output_dir=f"./checkpoints/EgoTempo_{model_id.split('/')[-1]}",
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_pin_memory=False,
        optim="paged_adamw_32bit" if is_qwen_model else "adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=qwen_collate_fn if is_qwen_model else smolvlm_collate_fn,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    print("[INFO] Finished training!")


if __name__ == "__main__":
    main()
