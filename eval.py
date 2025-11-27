import argparse
import ast
import json
import os
from typing import Dict, List, Tuple
import time
from openai import OpenAI
from tqdm import tqdm

EGO_TEXT_INDOOR_TYPES = ["hands-on", "kitchen", "shopping", "gameplay", "book-related", "others"]
EGO_TEXT_OUTDOOR_TYPES = ["location", "direction", "description", "intention reasoning", "others"]
EGO_TEMPO_TYPES = [
    "action sequence",
    "action-specific object",
    "counting actions",
    "counting objects",
    "future action prediction",
    "locating object",
    "object sequence",
    "object-specific action",
    "spatial relationship",
    "temporal event ordering",
]

PREDEFINED_TYPES = {
    "EgoTextVQA-Indoor": EGO_TEXT_INDOOR_TYPES,
    "EgoTextVQA-Outdoor": EGO_TEXT_OUTDOOR_TYPES,
    "EgoTempo": EGO_TEMPO_TYPES,
    # fall back to union if subset is unknown
    "EgoTextVQA": sorted(set(EGO_TEXT_INDOOR_TYPES + EGO_TEXT_OUTDOOR_TYPES)),
}


def ensure_json_file_exists(file_path: str):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as json_file:
            json.dump([], json_file)


def read_json_file(file_path: str) -> List[Dict]:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []


def write_json_file(file_path: str, data: List[Dict]):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def remove_special_characters(text: str) -> str:
    return text.replace("\n", "")


def annotate(judge_model: str, question: str, answer: str, prediction: str) -> Dict:
    api_key = os.environ["GEMINI_API_KEY"]
    api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    client = OpenAI(api_key=api_key, base_url=api_base)

    response_dict = {}
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer. "
                    "Please note that not only matches of noun phrases between answers, but also matches of prepositional phrases. "
                    'For example, "at the car wash on your right" does not exactly match "car wash". '
                    '"at the gas station beside the sign \'gas sale\'" does not exactly match "gas station".\n'
                    "- Consider synonyms or paraphrases as valid matches. "
                    "Note that the predicted answer must be consistent with the string type of the correct answer, which may include phone numbers, email addresses, numbers, dates, etc. "
                    'For example, the string types of "www.usps.com" and "visit their website" are inconsistent, '
                    'and the string types of "9849041316" and "advertiser\'s contact number" are inconsistent.\n'
                    "- Evaluate the correctness of the prediction compared to the answer.",
                },
                {
                    "role": "user",
                    "content": "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {prediction}\n\n"
                    "Provide your eval_code only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'pred': 'yes', 'score': 5}, {'pred': 'no', 'score': 1}.",
                },
            ],
            stream=False,
        )
        response_message = response.choices[0].message.content
        response_dict = ast.literal_eval(response_message)
    except Exception as e:
        print("Error processing:", e)
    return response_dict


def sample_key(sample: Dict, default_dataset: str) -> Tuple[str, str, str, str]:
    dataset = sample.get("dataset") or default_dataset
    subset = sample.get("subset") or dataset
    return (
        str(dataset).lower(),
        str(subset).lower(),
        str(sample.get("video_id", "")),
        str(sample.get("question_id", "")),
    )


def resolve_dataset_key(dataset_label: str, subset_label: str) -> str:
    dataset_norm = (dataset_label or "").lower()
    subset_norm = (subset_label or "").lower()

    if "egotempo" in dataset_norm or "egotempo" in subset_norm:
        return "EgoTempo"
    if "outdoor" in subset_norm:
        return "EgoTextVQA-Outdoor"
    if "indoor" in subset_norm:
        return "EgoTextVQA-Indoor"
    if dataset_norm == "egotextvqa":
        return "EgoTextVQA"
    return dataset_label or subset_label or "Unknown"


def init_score_table(dataset_key: str) -> Dict[str, Dict[str, int]]:
    question_types = PREDEFINED_TYPES.get(dataset_key, [])
    return {
        q_type: {"score_sum": 0, "count": 0, "yes_count": 0, "no_count": 0}
        for q_type in question_types
    }


def matches_dataset_filter(dataset_label: str, subset_label: str, allowed: set) -> bool:
    if not allowed:
        return True
    dataset_norm = (dataset_label or "").lower()
    subset_norm = (subset_label or "").lower()
    resolved_norm = resolve_dataset_key(dataset_label, subset_label).lower()
    return any(label in allowed for label in (dataset_norm, subset_norm, resolved_norm))


def first_unprocessed_index(pred_contents, existing_keys, args) -> int:
    """Find the first index that still needs evaluation so we can resume faster."""
    allowed = {d.lower() for d in args.datasets} if args.datasets else None
    for idx, sample in enumerate(pred_contents):
        dataset_label = sample.get("dataset") or args.default_dataset
        subset_label = sample.get("subset") or dataset_label
        if not matches_dataset_filter(dataset_label, subset_label, allowed):
            continue
        if args.model_key not in sample:
            continue
        if sample_key(sample, args.default_dataset) not in existing_keys:
            return idx
    return len(pred_contents)


def print_score(args):
    print(f"Evaluating predictions from {args.pred_path}")
    with open(args.pred_path, "r") as file:
        pred_contents = json.load(file)

    ensure_json_file_exists(args.output_json)
    existing_data = read_json_file(args.output_json)
    existing_keys = {sample_key(item, args.default_dataset) for item in existing_data}
    allowed = {d.lower() for d in args.datasets} if args.datasets else None

    start_idx = first_unprocessed_index(pred_contents, existing_keys, args)
    if start_idx >= len(pred_contents):
        print("[INFO] Nothing new to evaluate; all samples are already annotated.")
        return
    print(f"[INFO] Resuming from sample index {start_idx} (0-based) out of {len(pred_contents)}")

    new_entries = 0
    pending_since_last_save = 0
    for sample in tqdm(pred_contents[start_idx:], desc="Generating scores", initial=start_idx, total=len(pred_contents)):
        dataset_label = sample.get("dataset") or args.default_dataset
        subset_label = sample.get("subset") or dataset_label

        if not matches_dataset_filter(dataset_label, subset_label, allowed):
            continue

        if args.model_key not in sample:
            print(f"Skipping {sample.get('question_id', 'N/A')} - missing field '{args.model_key}'.")
            continue

        key = sample_key(sample, args.default_dataset)
        if key in existing_keys:
            continue

        q = sample["question"]
        a = remove_special_characters(str(sample["correct_answer"])).lower()
        pred_text = remove_special_characters(str(sample[args.model_key])).lower()

        score_dict = annotate(args.judge_model, q, a, pred_text)
        time.sleep(8)
        sample["correct_answer"] = a
        sample[args.model_key] = pred_text
        sample["acc"] = score_dict.get("pred", "no")
        sample["score"] = score_dict.get("score", 0)
        sample["dataset"] = dataset_label
        sample["subset"] = subset_label

        existing_data.append(sample)
        existing_keys.add(key)
        new_entries += 1
        pending_since_last_save += 1

        if pending_since_last_save >= args.save_every:
            write_json_file(args.output_json, existing_data)
            pending_since_last_save = 0

    write_json_file(args.output_json, existing_data)
    print(f"[INFO] Added {new_entries} new annotations to {args.output_json}")


def calculate_score(args):
    with open(args.output_json, "r") as file:
        output_contents = json.load(file)

    allowed = {d.lower() for d in args.datasets} if args.datasets else None
    score_tables: Dict[str, Dict[str, Dict[str, int]]] = {}
    totals = {"score_sum": 0, "count": 0, "yes": 0, "no": 0}

    for sample in tqdm(output_contents, desc="Evaluating predictions"):
        dataset_label = sample.get("dataset") or args.default_dataset
        subset_label = sample.get("subset") or dataset_label

        if not matches_dataset_filter(dataset_label, subset_label, allowed):
            continue

        dataset_key = resolve_dataset_key(dataset_label, subset_label)
        table = score_tables.setdefault(dataset_key, init_score_table(dataset_key))

        question_type = sample.get("question_type", "unknown").lower()
        if question_type not in table:
            table[question_type] = {"score_sum": 0, "count": 0, "yes_count": 0, "no_count": 0}

        pred_flag = str(sample.get("acc", "")).lower()
        if "yes" in pred_flag:
            table[question_type]["yes_count"] += 1
            totals["yes"] += 1
        elif "no" in pred_flag:
            table[question_type]["no_count"] += 1
            totals["no"] += 1

        score = int(sample.get("score", 0) or 0)
        table[question_type]["score_sum"] += score
        table[question_type]["count"] += 1
        totals["score_sum"] += score
        totals["count"] += 1

    for dataset_key, data in score_tables.items():
        print(f"=== {dataset_key} ===")
        for q_type, stats in data.items():
            if (stats["yes_count"] + stats["no_count"]) > 0:
                accuracy = stats["yes_count"] / (stats["yes_count"] + stats["no_count"])
            else:
                accuracy = 0
            average_score = stats["score_sum"] / stats["count"] if stats["count"] > 0 else 0
            print(f"Type: {q_type}")
            print(f"Accuracy: {accuracy}")
            print(f"Average score: {average_score}\n")

    overall_accuracy = (totals["yes"] / (totals["yes"] + totals["no"])) if (totals["yes"] + totals["no"]) else 0
    overall_average_score = (totals["score_sum"] / totals["count"]) if totals["count"] else 0
    print("Overall Results:")
    print("Overall Accuracy:", overall_accuracy)
    print("Overall Average Score:", overall_average_score)


def parse_args():
    parser = argparse.ArgumentParser(description="Question-answer evaluation using gemini-2.5-flash")
    parser.add_argument(
        "--pred_path",
        default="./results/finetuned_SmolVLM2-500M-Video-Instruct_egotempo_test_predictions.json",
        help="Path to the file containing predictions.",
    )
    parser.add_argument(
        "--output_json",
        default="./results.json",
        help="Path to save the annotation JSON file.",
    )
    parser.add_argument(
        "--model-key",
        dest="model_key",
        default="gemini-2.5-flash",
        help="Field name inside the predictions JSON that stores model answers.",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-flash",
        help="Model used for LLM-as-a-judge scoring.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset/subset filters (e.g. egotextvqa, EgoTextVQA-Outdoor, egotempo). Defaults to all.",
    )
    parser.add_argument(
        "--default-dataset",
        default="EgoTextVQA",
        help="Fallback dataset label when predictions are missing the dataset field.",
    )
    parser.add_argument(
        "--save-every",
        dest="save_every",
        type=int,
        default=10,
        help="Persist results to disk after this many new samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_score(args)
    calculate_score(args)
