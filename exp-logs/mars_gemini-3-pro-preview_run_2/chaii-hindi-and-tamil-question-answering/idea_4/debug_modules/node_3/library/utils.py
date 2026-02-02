import os
import random
import numpy as np
import torch
import collections
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.
    Implementation provided in the task description.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0
    return float(len(c)) / denominator


def postprocess_qa_predictions(
    examples, features, predictions, n_best_size=20, max_answer_length=30
):
    """
    Post-processes the raw predictions from the model to generate the final answer text.
    Handles sliding windows by aggregating predictions across features for each example.

    Args:
        examples: The original dataset (e.g., Hugging Face Dataset) containing 'id' and 'context'.
        features: The tokenized features containing 'example_id', 'offset_mapping', and 'token_type_ids'.
        predictions: A tuple (start_logits, end_logits).
        n_best_size: The number of best logits to consider.
        max_answer_length: The maximum length of the answer.

    Returns:
        A dictionary mapping example IDs to the predicted answer text.
    """
    all_start_logits, all_end_logits = predictions

    # Build a map from example_id to feature indices
    # We assume 'id' in examples matches 'example_id' in features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    all_predictions = collections.OrderedDict()

    for example in examples:
        example_id = example["id"]

        # If no features generated for this example, predict empty string
        if example_id not in features_per_example:
            all_predictions[example_id] = ""
            continue

        feature_indices = features_per_example[example_id]
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            offset_mapping = features[feature_index]["offset_mapping"]
            # MuRIL/BERT uses token_type_ids to distinguish Question (0) from Context (1)
            token_type_ids = features[feature_index].get("token_type_ids", None)

            # Get the indices of the top-k start and end logits
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Basic validity checks
                    if start_index >= len(offset_mapping) or end_index >= len(
                        offset_mapping
                    ):
                        continue
                    if start_index > end_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    # Context check: Ensure answer is strictly within the context segment
                    if token_type_ids is not None:
                        # In MuRIL/BERT, context is usually segment 1
                        if (
                            token_type_ids[start_index] != 1
                            or token_type_ids[end_index] != 1
                        ):
                            continue

                    # Check for special tokens (usually mapped to (0, 0))
                    # We skip them unless they are genuinely part of the text (rare for (0,0))
                    if offset_mapping[start_index] == (0, 0) or offset_mapping[
                        end_index
                    ] == (0, 0):
                        continue

                    # Calculate combined score
                    score = start_logits[start_index] + end_logits[end_index]

                    # Map tokens to character positions in the original context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    answer_text = context[start_char:end_char]

                    valid_answers.append({"score": score, "text": answer_text})

        if len(valid_answers) > 0:
            # Select the answer with the highest score across all features for this example
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
            all_predictions[example_id] = best_answer["text"]
        else:
            all_predictions[example_id] = ""

    return all_predictions


def compute_metrics(predictions, ground_truths):
    """
    Computes the average Jaccard score across the dataset.

    Args:
        predictions: Dict {id: prediction_string}
        ground_truths: DataFrame or list of dicts with 'id' and 'answer_text'

    Returns:
        float: Average Jaccard score
    """
    gt_dict = {}

    # Normalize ground_truths to a dictionary
    if isinstance(ground_truths, pd.DataFrame):
        for _, row in ground_truths.iterrows():
            gt_dict[str(row["id"])] = row["answer_text"]
    elif isinstance(ground_truths, (list, tuple)):
        for item in ground_truths:
            gt_dict[str(item["id"])] = item["answer_text"]
    elif isinstance(ground_truths, dict):
        gt_dict = ground_truths

    total_score = 0.0
    count = 0

    for pid, ptext in predictions.items():
        pid = str(pid)
        if pid in gt_dict:
            score = jaccard(gt_dict[pid], ptext)
            total_score += score
            count += 1

    if count == 0:
        return 0.0

    return total_score / count


def save_submission(predictions, output_file):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions: Dict {id: prediction_string}
        output_file: Path to save the CSV
    """
    data = []
    for pid, ptext in predictions.items():
        data.append({"id": pid, "PredictionString": ptext})

    df = pd.DataFrame(data)
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
