import os
import random
import numpy as np
import torch
import collections
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Computes the Jaccard similarity score between two strings.
    Defined as Intersection / Union of the set of unique words.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard score between 0.0 and 1.0.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    union_len = len(a) + len(b) - len(c)
    if union_len == 0:
        return 0.0
    return float(len(c)) / union_len


def postprocess_qa_predictions(
    examples, features, predictions, n_best_size=20, max_answer_length=30
):
    """
    Post-processes the predictions of a Question-Answering model to convert logits to text answers.
    Handles sliding windows by aggregating scores across chunks.

    Args:
        examples: The non-preprocessed dataset (contains the original texts and IDs).
        features: The processed dataset (contains the chunks, offset_mappings, and sequence_ids).
        predictions: A tuple containing two numpy arrays (start_logits, end_logits).
        n_best_size (int): The number of top logits to consider for start and end positions.
        max_answer_length (int): The maximum allowed length (in tokens) for an answer.

    Returns:
        OrderedDict: A dictionary mapping example_id to the predicted answer string.
    """
    all_start_logits, all_end_logits = predictions

    # Build a map from example ID to its corresponding feature indices.
    # Features are chunks of the original examples.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        # Features should map back to examples via 'example_id' or 'overflow_to_sample_mapping'
        if "example_id" in feature:
            features_per_example[feature["example_id"]].append(i)
        elif "overflow_to_sample_mapping" in feature:
            sample_idx = feature["overflow_to_sample_mapping"]
            example_id = examples["id"][sample_idx]
            features_per_example[example_id].append(i)
        else:
            # If mapping is missing, we cannot process this feature
            pass

    # The dictionary to hold the final predictions
    all_predictions = collections.OrderedDict()

    for example in examples:
        example_id = example["id"]

        # If no features generated for this example (e.g., filtered out), predict empty string
        if example_id not in features_per_example:
            all_predictions[example_id] = ""
            continue

        feature_indices = features_per_example[example_id]
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            feature = features[feature_index]

            # offset_mapping maps tokens to character positions in the original context
            offset_mapping = feature["offset_mapping"]

            # sequence_ids distinguishes between question (0) and context (1).
            # None usually denotes special tokens.
            if "sequence_ids" in feature:
                sequence_ids = feature["sequence_ids"]
            else:
                # Without sequence_ids, we cannot safely identify the context span
                continue

            # Identify the top n_best_size start and end indices
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 1. Check indices are within bounds of the feature
                    if start_index >= len(offset_mapping) or end_index >= len(
                        offset_mapping
                    ):
                        continue

                    # 2. Check indices are within the context (sequence_id == 1)
                    if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                        continue

                    # 3. Check that end comes after start
                    if end_index < start_index:
                        continue

                    # 4. Check answer length constraint
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    # 5. Compute score
                    score = start_logits[start_index] + end_logits[end_index]

                    # 6. Extract text using offset_mapping
                    # offset_mapping[i] is a tuple (start_char, end_char)
                    char_start = offset_mapping[start_index][0]
                    char_end = offset_mapping[end_index][1]

                    if char_start is None or char_end is None:
                        continue

                    answer_text = context[char_start:char_end]

                    valid_answers.append({"score": score, "text": answer_text})

        if len(valid_answers) > 0:
            # Select the answer with the highest score across all chunks
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
            all_predictions[example_id] = best_answer["text"]
        else:
            all_predictions[example_id] = ""

    return all_predictions
