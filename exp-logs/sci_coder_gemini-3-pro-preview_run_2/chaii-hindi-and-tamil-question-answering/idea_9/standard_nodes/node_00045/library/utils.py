import os
import random
import numpy as np
import torch
import collections


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def post_process_predictions(
    examples, features, predictions, n_best_size=20, max_answer_length=30
):
    """
    Post-processes model logits to generate final text predictions.

    Args:
        examples: The original dataset (e.g., list of dicts or HF Dataset) containing 'id' and 'context'.
        features: The tokenized features (list of dicts or HF Dataset) containing 'example_id' and 'offset_mapping'.
        predictions: A tuple of (start_logits, end_logits) or (start_logits, end_logits, answerability_logits).
        n_best_size: Number of top logits to consider for start and end positions.
        max_answer_length: Maximum allowed length for the predicted answer text.

    Returns:
        A dictionary mapping example IDs to the predicted answer strings.
    """
    # Unpack predictions based on whether answerability logits are included
    if len(predictions) == 3:
        all_start_logits, all_end_logits, all_answerability_logits = predictions
        has_answerability = True
    else:
        all_start_logits, all_end_logits = predictions
        has_answerability = False

    # Build a map from example_id to a list of feature indices
    # This assumes features are in the same order or contain 'example_id'
    example_id_to_feature_indices = collections.defaultdict(list)
    for i in range(len(features)):
        ex_id = features[i]["example_id"]
        example_id_to_feature_indices[ex_id].append(i)

    final_predictions = {}

    for example in examples:
        example_id = example["id"]
        context = example["context"]

        # Retrieve all features (windows) associated with this example
        feature_indices = example_id_to_feature_indices.get(example_id, [])

        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # Calculate answerability penalty if available
            # We add log(sigmoid(logit)) to the score, which is equivalent to multiplying by probability
            answerability_score = 0.0
            if has_answerability:
                cls_logit = all_answerability_logits[feature_index]
                # Handle scalar or array input
                if isinstance(cls_logit, (np.ndarray, list)):
                    val = cls_logit.item() if np.ndim(cls_logit) == 0 else cls_logit[0]
                else:
                    val = cls_logit

                # log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
                # Using numpy for stability
                answerability_score = -np.logaddexp(0, -val)

            # Identify top-k start and end indices
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            offset_mapping = features[feature_index]["offset_mapping"]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 1. Check index bounds
                    if start_index >= len(offset_mapping) or end_index >= len(
                        offset_mapping
                    ):
                        continue

                    # 2. Check for valid offsets (not None)
                    if (
                        offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue

                    # 3. Check for special tokens (usually mapped to (0,0) or similar, depending on tokenizer)
                    # We skip spans that start or end on CLS/SEP tokens if they map to (0,0)
                    # Cite debug_lesson_6: Sanitize Complex Data Types (convert potential numpy array to tuple)
                    if tuple(offset_mapping[start_index]) == (0, 0) or tuple(
                        offset_mapping[end_index]
                    ) == (0, 0):
                        continue

                    # 4. Check relative position
                    if end_index < start_index:
                        continue

                    # 5. Check length constraint
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    # Calculate span score
                    # Score = Start_Logit + End_Logit + Answerability_Log_Prob
                    raw_span_score = start_logits[start_index] + end_logits[end_index]
                    final_score = raw_span_score + answerability_score

                    # Extract text
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = context[start_char:end_char]

                    valid_answers.append({"score": final_score, "text": answer_text})

        # Select the best answer for this example across all features
        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
            final_predictions[example_id] = best_answer["text"]
        else:
            # Fallback if no valid span is found (unlikely with n_best_size=20)
            final_predictions[example_id] = ""

    return final_predictions
