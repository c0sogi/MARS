import os
import random
import collections
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Jaccard similarity score between two strings.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def postprocess_qa_predictions(
    examples, features, predictions, n_best_size=20, max_answer_length=30
):
    """
    Post-processes the raw logits from the model to generate final text predictions.

    Args:
        examples: The original dataset containing 'id' and 'context'.
        features: The tokenized features containing 'input_ids', 'offset_mapping', and 'example_id'.
        predictions: A tuple (start_logits, end_logits).
        n_best_size: The number of best logits to consider.
        max_answer_length: The maximum allowed length for an answer.

    Returns:
        A dictionary mapping example IDs to the predicted answer string.
    """
    all_start_logits, all_end_logits = predictions

    # Build a map from example_id to list of feature indices
    # We assume 'features' is a list-like object where each item is a dict or has attribute access
    # matching the keys used below.
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    final_predictions = {}

    for example in examples:
        example_id = example["id"]
        context = example["context"]
        feature_indices = features_per_example.get(example_id, [])

        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            feature = features[feature_index]
            offset_mapping = feature["offset_mapping"]
            input_ids = feature["input_ids"]

            # Determine context bounds within the input_ids
            # XLM-RoBERTa structure: <s> Q </s> </s> C </s>
            # IDs: 0 ... 2 2 ... 2
            context_start_idx = 0
            context_end_idx = len(input_ids)

            # Try to use sequence_ids if available (standard in some HF pipelines)
            seq_ids = feature.get("sequence_ids", None)

            if seq_ids is not None:
                # Usually 1 indicates context
                c_indices = [i for i, s in enumerate(seq_ids) if s == 1]
                if not c_indices:
                    continue
                context_start_idx = c_indices[0]
                context_end_idx = c_indices[-1] + 1
            else:
                # Fallback heuristic for XLM-RoBERTa
                sep_token_id = 2
                sep_indices = [i for i, t in enumerate(input_ids) if t == sep_token_id]

                # Look for the double separator </s> </s> which divides Q and C
                found_sep = False
                for k in range(len(input_ids) - 1):
                    if (
                        input_ids[k] == sep_token_id
                        and input_ids[k + 1] == sep_token_id
                    ):
                        context_start_idx = k + 2
                        found_sep = True
                        break

                if found_sep:
                    # Context ends at the next separator after start
                    for k in range(context_start_idx, len(input_ids)):
                        if input_ids[k] == sep_token_id:
                            context_end_idx = k
                            break
                else:
                    # If we can't find the specific structure, we might skip this feature
                    # or assume the whole thing is relevant (risky).
                    # We'll skip to be safe against noise.
                    continue

            # Get the indices of the best logits
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

                    # Ensure span is strictly within the context part of the input
                    if start_index < context_start_idx or end_index >= context_end_idx:
                        continue

                    # Ensure offsets are valid (not None, which indicates special tokens)
                    if (
                        offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue

                    # Map to character offsets in the original context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    # Verify character indices are within the context string bounds
                    if end_char > len(context):
                        continue

                    text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]

                    valid_answers.append({"score": score, "text": text})

        if valid_answers:
            # Select the answer with the highest combined score
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
            final_predictions[example_id] = best_answer["text"]
        else:
            # Default to empty string if no valid answer found
            final_predictions[example_id] = ""

    return final_predictions
