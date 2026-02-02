import os
import collections
import numpy as np
import pandas as pd
from library.config import Config


def sigmoid(x):
    """Computes the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def postprocess_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=50
):
    """
    Post-processes the raw logits from the model to generate final text predictions.

    Args:
        examples (pd.DataFrame): The original dataset (test/val) containing 'context', 'id'.
        features (pd.DataFrame): The tokenized features containing mapping info.
        raw_predictions (dict): Dictionary with 'start_logits', 'end_logits', 'answerability_logits'.
        n_best_size (int): Number of top logits to consider for start/end.
        max_answer_length (int): Maximum allowed length of the answer in tokens.

    Returns:
        pd.DataFrame: Submission dataframe with 'id' and 'PredictionString'.
    """

    # Unpack predictions
    all_start_logits = raw_predictions["start_logits"]
    all_end_logits = raw_predictions["end_logits"]
    all_ans_logits = raw_predictions["answerability_logits"]

    # Pre-fetch feature columns for faster access in the loop
    # features is a DataFrame, accessing .iloc inside a loop is slow
    offset_mappings = features["offset_mapping"].tolist()
    sequence_ids_list = features["sequence_ids"].tolist()

    # Create a map from example_id to list of feature indices
    # We use a dictionary for O(1) access
    example_id_to_feature_indices = collections.defaultdict(list)
    for idx, row in features.iterrows():
        # Ensure ID is string
        example_id_to_feature_indices[str(row["example_id"])].append(idx)

    # Store final predictions
    predictions = {}

    # Iterate over all examples
    # We iterate over examples to ensure we produce a prediction for every ID
    for idx, example in examples.iterrows():
        example_id = str(example["id"])
        context = example["context"]

        # Get features associated with this example
        feature_indices = example_id_to_feature_indices.get(example_id, [])

        valid_answers = []

        for feature_index in feature_indices:
            # Retrieve logits
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            ans_logit = all_ans_logits[feature_index]

            # Calculate answerability probability
            ans_prob = sigmoid(ans_logit)

            # Retrieve feature info
            offset_mapping = offset_mappings[feature_index]
            sequence_ids = sequence_ids_list[feature_index]

            # Get top-k start and end indices
            # argsort sorts in ascending order, so we take the last n_best_size
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

                    # Ensure indices are within context
                    # sequence_ids: None (special), 0 (question), 1 (context)
                    # We only want answers extracted from the context (1)
                    # We handle potential None/NaN values by checking equality to 1
                    if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                        continue

                    # Calculate score based on prompt formula
                    # Score = (StartLogit + EndLogit) * sigma(AnswerabilityLogit)
                    # This effectively suppresses spans from windows deemed irrelevant (low ans_prob)
                    span_score = (
                        start_logits[start_index] + end_logits[end_index]
                    ) * ans_prob

                    valid_answers.append(
                        {
                            "score": span_score,
                            "start_index": start_index,
                            "end_index": end_index,
                            "offset_mapping": offset_mapping,
                        }
                    )

        if len(valid_answers) > 0:
            # Sort by score descending to find the best span across all windows
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]

            # Extract text using offset mapping
            offsets = best_answer["offset_mapping"]
            # offsets is a list of [start_char, end_char]
            start_char = offsets[best_answer["start_index"]][0]
            end_char = offsets[best_answer["end_index"]][1]

            # Extract from original context string
            predicted_text = context[start_char:end_char]
        else:
            # Fallback if no valid span is found
            predicted_text = ""

        predictions[example_id] = predicted_text

    # Create submission DataFrame
    submission_data = [{"id": k, "PredictionString": v} for k, v in predictions.items()]
    submission_df = pd.DataFrame(submission_data)

    # Ensure all IDs from the input examples are present in the output
    # This handles cases where an example might have been filtered out during feature creation (unlikely but safe)
    all_ids = set(examples["id"].astype(str))
    pred_ids = set(submission_df["id"])
    missing_ids = all_ids - pred_ids

    if missing_ids:
        missing_data = [{"id": mid, "PredictionString": ""} for mid in missing_ids]
        missing_df = pd.DataFrame(missing_data)
        submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

    return submission_df


def save_submission(submission_df, output_path=Config.SUBMISSION_PATH):
    """
    Saves the submission dataframe to a CSV file.

    Args:
        submission_df (pd.DataFrame): The dataframe containing predictions.
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
