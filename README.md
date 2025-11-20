# EgoTextVQA

## Test set results

Base SmolVLM2-500M-Video-Instruct - 27/685 correct

Finetuned SmolVLM2-500M-Video-Instruct - 4/685 correct

---

## Data preparation

**Goal:** create a question–answer dataset for EgoTextVQA (and optional local copies) of the videos, used to finetune VLMs and perform various experiments.

#### 1. Source datasets

* **Videos:** [ShengZhou97/EgoTextVQA](https://huggingface.co/datasets/ShengZhou97/EgoTextVQA) (also includes the JSONL annotations)
* **Final QA metadata dataset:** [petkopetkov/EgoTextVQA](https://huggingface.co/datasets/petkopetkov/EgoTextVQA)

### 2. Build the QA metadata dataset

This step joins the original videos with the JSONL annotations (creates [petkopetkov/EgoTextVQA](https://huggingface.co/datasets/petkopetkov/EgoTextVQA))

```bash
cd data
python create_qa_dataset.py
```

---

### 3. Preprocess videos locally

If you want faster / cheaper training/inference, you can create local downsampled versions of the videos.

**3.1. Reduce FPS**

By default it reduces the videos to 6 FPS. This can be controlled through the `TARGET_FPS` variable. 

```bash
cd data
python change_video_fps.py
```

Outputs:

```text
./data/egotextvqa_fps6/
  EgoTextVQA-Indoor/<video_id>.mp4
  EgoTextVQA-Outdoor/<video_id>.mp4
```

**3.2. Reduce resolution**

```bash
cd data
python change_video_resolution.py
```

Outputs:

```text
./data/egotextvqa_fps6_lowres/
  EgoTextVQA-Indoor/<video_id>.mp4
  EgoTextVQA-Outdoor/<video_id>.mp4
```

Here’s a compact, README-friendly description of your **training script**.

---

## Training

The training script supports finetuning the [SmolVLM2-500M-Video-Instruct model](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) on the EgoTextVQA dataset using **video + question → answer** supervision.

It assumes that the data was preprocessed locally (FPS and resolution were reduced) and tries to read it from `./data/egotextvqa_fps6_lowres`

```bash
python train.py
```

By default, the checkpoints are saved to `./checkpoints/SmolVLM2-500M-Video-Instruct`