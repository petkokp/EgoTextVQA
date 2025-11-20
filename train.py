from unsloth import FastVisionModel
import math
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, TrainingArguments, Trainer

from data.load_data import load_data

os.environ["WANDB_PROJECT"] = "egotextvqa"
SEED = 42


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

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

processor = AutoProcessor.from_pretrained(model_id)

model, tokenizer = FastVisionModel.from_pretrained(
    model_id,
    full_finetuning=True,
)

peak_mem = torch.cuda.max_memory_allocated()
print(f"[INFO] The model as is is holding: {peak_mem / 1024**3:.2f} GB of GPU RAM")

train_ds, val_ds, test_ds = load_data()
print(f"[INFO] Train / Val / Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

def collate_fn(examples):
    instances = []
    for example in examples:
        video_path = example["local_video_path"]

        user_content = [{"type": "text", "text": example["question"]}]
        user_content.append({"type": "video", "path": video_path})

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

model_name = model_id.split("/")[-1]

FastVisionModel.for_training(model, use_gradient_checkpointing=False)

batch_size = 2
gradient_accumulation_steps = 8
num_train_epochs = 1

config = dynamic_steps(
    n_examples=len(train_ds),
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
)

training_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=1e-4,
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
    output_dir=f"./checkpoints/{model_name}",
    remove_unused_columns=False,
    report_to="wandb",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

print("[INFO] Finished training!")
