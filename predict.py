import os
import json
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from unsloth import FastVisionModel
from transformers import AutoProcessor
from data.load_data import load_splits

EVAL_MODEL = "gemini-2.5-flash"

model_name = "unsloth/Qwen3-VL-2B-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    full_finetuning=True,
)
FastVisionModel.for_inference(model)
processor = AutoProcessor.from_pretrained(model_name)

os.makedirs("temp", exist_ok=True)

splits = load_splits()
data = splits["test"]
print(f"[INFO] Loaded test split with {len(data)} examples")

predictions = []
for sample in tqdm(data):
    try:
        video_id = sample["video_id"]
        timestamp = sample["timestamp"]
        question = sample["question"]
        correct_answer = sample["answer"]
        local_video_path = sample["local_video_path"]

        # TODO - fix HF dataset
        local_video_path = local_video_path.replace("egotextvqa", "EgoTextVQA")
        local_video_path = local_video_path.replace("egocentricvqa", "EgoTextVQA")

        # trim video to 0 -> timestamp + 1 sec (for real-time eval)
        clip = VideoFileClip(local_video_path)
        end_time = min(clip.duration, timestamp + 1.0)
        subclip = clip.subclip(0, end_time)
        temp_video_path = f"temp/{video_id}_{timestamp:.1f}.mp4"
        subclip.write_videofile(temp_video_path, codec="libx264", audio=False, logger=None)  # No audio, quiet

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": temp_video_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]) :]
        pred_answer = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

        predictions.append({
            "question_id": sample["question_id"],
            "video_id": sample["video_id"],
            "question": sample["question"],
            "question_type": sample["question_type"],
            EVAL_MODEL: pred_answer,
            "correct_answer": correct_answer
        })

        os.remove(temp_video_path)

    except Exception as e:
        print(f"Error processing {sample['question_id']}: {e}")
        continue

with open(f"{model_name}_predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)