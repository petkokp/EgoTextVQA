# EgoTextVQA

https://github.com/user-attachments/assets/e85f740b-0538-4822-b307-634b9ad1d838

## EgoTempo results (496 samples)

**Models fine-tuned on EgoTextVQA and evaluated on EgoTempo**

| Type                     | Smol-FT Acc | Smol-FT Score | Qwen2B-FT Acc | Qwen2B-FT Score |
|--------------------------|-------------|---------------|---------------|------------------|
| action sequence          | 0.0000      | **0.4792**    | 0.0000        | 0.3125           |
| action-specific object   | **0.0400**  | **0.5400**    | 0.0200        | 0.4000           |
| counting actions         | **0.2653**  | **1.5102**    | **0.2653**    | 1.4694           |
| counting objects         | **0.3000**  | **1.5600**    | 0.2000        | 1.1000           |
| future action prediction | **0.0600**  | **0.7000**    | **0.0600**    | 0.5400           |
| locating object          | **0.0200**  | **0.6400**    | **0.0200**    | 0.4000           |
| object sequence          | 0.0200      | 0.4400        | **0.0600**    | **0.4600**       |
| object-specific action   | 0.0000      | **0.7400**    | **0.0200**    | 0.5600           |
| spatial relationship     | 0.1400      | 1.2200        | **0.2600**    | **1.8200**       |
| temporal event ordering  | 0.0408      | 0.2857        | **0.0816**    | **0.4898**       |
| **Overall**              | 0.0887      | **0.8125**    | **0.0988**    | 0.7560           |


## EgoTextVQA test set results (685 samples)

### Overall results (indoor & outdoor)

| Model                                      | Dataset | Overall Accuracy | Overall Avg. Score |
|--------------------------------------------|---------|------------------|--------------------|
| SmolVLM2-500M-Video-Instruct (base)        | Outdoor | 0.0091           | 0.0457             |
| SmolVLM2-500M-Video-Instruct (base)        | Indoor  | 0.0040           | 0.0321             |
| SmolVLM2-500M-Video-Instruct (finetuned)   | Outdoor | 0.0183           | 0.2283             |
| SmolVLM2-500M-Video-Instruct (finetuned)   | Indoor  | 0.0461           | 0.3687             |
| Qwen3-VL-2B-Instruct                       | Outdoor | **0.0548**       | **0.3744**         |
| Qwen3-VL-2B-Instruct                       | Indoor  | **0.1142**       | **0.6673**         |
| Qwen3-VL-4B-Instruct                       | Outdoor | 0.0137           | 0.1279             |
| Qwen3-VL-4B-Instruct                       | Indoor  | 0.0000           | 0.0040             |
| Qwen3-VL-2B-Thinking                       | Outdoor | 0.0411           | 0.2557             |
| Qwen3-VL-2B-Thinking                       | Indoor  | **0.1142**       | 0.6132             |
| Qwen3-VL-2B-Instruct (finetuned)           | Outdoor | 0.0380           | 0.2911             |
| Qwen3-VL-2B-Instruct (finetuned)           | Indoor  | 0.0353           | 0.3173             |

### Outdoor per-type results

| Type         | Smol Base Acc | Smol Base Score | Smol FT Acc | Smol FT Score | Qwen2B Acc | Qwen2B Score | Qwen4B Acc | Qwen4B Score | Qwen2B-Thinking Acc | Qwen2B-Thinking Score | Qwen2B-FT Acc | Qwen2B-FT Score |
|-------------|---------------|-----------------|-------------|---------------|------------|--------------|------------|--------------|----------------------|------------------------|---------------|------------------|
| hands-on    | 0.0139        | 0.0694          | 0.0000      | 0.1389        | **0.0556** | **0.3472**   | 0.0139     | 0.0694       | 0.0278               | 0.1944                 | 0.0506        | 0.2911           |
| kitchen     | 0.0000        | 0.0000          | **0.0606**  | **0.4848**    | 0.0303     | 0.2121       | 0.0000     | 0.0000       | 0.0303               | 0.2424                 | 0.0526        | 0.3158           |
| shopping    | 0.0000        | **0.1667**      | 0.0000      | 0.0000        | 0.0000     | 0.0000       | 0.0000     | 0.0000       | 0.0000               | 0.0000                 | 0.0000        | 0.1143           |
| gameplay    | 0.0000        | 0.0000          | **0.0541**  | **0.5946**    | 0.0000     | 0.0541       | 0.0270     | 0.2703       | 0.0000               | 0.0541                 | 0.0455        | 0.4091           |
| book-related| 0.0263        | 0.1053          | 0.0000      | 0.0526        | **0.1579** | **0.9474**   | 0.0263     | 0.3421       | **0.1579**           | 0.8158                 | 0.0400        | 0.5600           |
| others      | 0.0000        | 0.0000          | 0.0000      | 0.0000        | **0.0303** | **0.3636**   | 0.0000     | 0.0000       | 0.0000               | 0.0303                 | 0.0263        | 0.1842           |

### Indoor per-type results

| Type                | Smol Base Acc | Smol Base Score | Smol FT Acc | Smol FT Score | Qwen2B Acc | Qwen2B Score | Qwen4B Acc | Qwen4B Score | Qwen2B-Thinking Acc | Qwen2B-Thinking Score | Qwen2B-FT Acc | Qwen2B-FT Score |
|---------------------|---------------|-----------------|-------------|---------------|------------|--------------|------------|--------------|----------------------|------------------------|---------------|------------------|
| location            | 0.0078        | 0.0547          | 0.0312      | 0.4219        | **0.1406** | **0.8281**   | 0.0000     | 0.0000       | 0.1250               | 0.6641                 | 0.0476        | 0.4167           |
| direction           | 0.0000        | 0.0085          | 0.1026      | 0.7009        | 0.1111     | 0.6239       | 0.0000     | 0.0000       | **0.1624**           | **0.8632**             | 0.0510        | 0.4184           |
| description         | 0.0075        | 0.0448          | 0.0299      | 0.2090        | **0.1343** | **0.7388**   | 0.0000     | 0.0000       | 0.1045               | 0.5970                 | 0.0148        | 0.1333           |
| intention reasoning | 0.0000        | 0.0230          | 0.0345      | 0.2299        | 0.0805     | **0.4943**   | 0.0000     | 0.0230       | **0.0920**           | 0.4483                 | 0.0300        | 0.3500           |
| others              | 0.0000        | 0.0000          | 0.0000      | 0.0000        | **0.0303** | **0.3636**   | 0.0000     | 0.0000       | 0.0000               | 0.0303                 | 0.0263        | 0.1842           |

* **Smol Base** = SmolVLM2-500M-Video-Instruct (base)
* **Smol FT** = Finetuned SmolVLM2-500M-Video-Instruct
* **Qwen2B** = Qwen3-VL-2B-Instruct
* **Qwen4B** = Qwen3-VL-4B-Instruct
* **Qwen2B-Thinking** = Qwen3-VL-2B-Thinking

---

The prediction and result files are in the `results` directory (`./results/<model_name>_predictions.json` and `./results/<model_name>_results.json`).  

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
./data/EgoTextVQA_fps6/
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
./data/EgoTextVQA_fps6_lowres/
  EgoTextVQA-Indoor/<video_id>.mp4
  EgoTextVQA-Outdoor/<video_id>.mp4
```

Here’s a compact, README-friendly description of your **training script**.

---

## Training

The training script supports finetuning of the [Qwen3-VL-2B-Instruct model](unsloth/Qwen3-VL-2B-Instruct) and the [SmolVLM2-500M-Video-Instruct model](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) on the EgoTextVQA dataset using **video + question → answer** supervision.

It assumes that the data was preprocessed locally (FPS and resolution were reduced) and tries to read it from `./data/EgoTextVQA_fps6_lowres`

Train only on EgoTempo, validate on EgoTempo:

```bash
python train.py --train-datasets egotempo --val-datasets egotempo --egotempo-root ./data/trimmed_clips --egotempo-json ./data/egotempo_openQA.json --model-id unsloth/Qwen3-VL-2B-Instruct
```

Train on EgoTextVQA, validate on EgoTempo:

```bash
python train.py --train-datasets egotextvqa --val-datasets egotempo
```

Mixed training (70/30) with shared val/test:

```bash
python train.py --train-datasets egotextvqa egotempo --train-weights 0.7 0.3 --val-datasets egotextvqa egotempo --val-weights 0.5 0.5
```

By default, the checkpoints are saved to `./checkpoints/SmolVLM2-500M-Video-Instruct`

## Evaluation

A model can be evaluated on the test set () that was generated during the data preparation process. First, model predictions are generated using:

```
python predict.py
```

`model_name` can be adjusted to the path of the model that is being evaluated. The results are saved to `<model_name>_predictions.json`

After the predictions are generated, a LLM-as-a-judge (by default `gemini-2.5-flash` but can easily be adapted) is used to get the final accuracy (the `GEMINI_API_KEY` environment variable has to be set):

```
python eval.py
```

By default the input (`--pred_path`) is `<model_name>_predictions.json` (from the previous step) and the results are saved to `results.json` (`--output_json`).

---

A part of the code is based on `https://github.com/zhousheng97/EgoTextVQA`.

## Dataset visualization and error analysis

The `app.py` Streamlit app allows to visualize the dataset and analyze the errors from the model's predictions. It can be launched like this:

```
streamlit run app.py
```

A predictions JSON file (like the one from the `results` directory) can be uploaded through the `1. Load Predictions` option in the sidebar control panel. Also the current video sample can be changed through the `2. Select Sample` option.
