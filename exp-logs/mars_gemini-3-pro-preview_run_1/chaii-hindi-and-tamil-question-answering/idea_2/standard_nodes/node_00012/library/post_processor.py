import os
import collections
import numpy as np
import pandas as pd
from library.config import Config
from library.model_handler import get_tokenizer


def postprocess_qa_predictions(examples, features, raw_predictions):
    """
    Post-processes the raw logits from the model to generate final text predictions.
    Aggregates predictions from multiple sliding windows for each document and
    selects the span with the highest global confidence score.

    Args:
        examples (pd.DataFrame): The original dataset (test/val) containing context and questions.
        features (pd.DataFrame): The processed features (sliding windows).
        raw_predictions (tuple): A tuple (start_logits, end_logits) as numpy arrays.

    Returns:
        dict: A dictionary mapping example IDs to predicted answer strings.
    """
    all_start_logits, all_end_logits = raw_predictions

    # Load tokenizer to identify special tokens for context boundaries
    tokenizer = get_tokenizer()
    sep_token_id = tokenizer.sep_token_id

    # Create a mapping from example_id to context text for O(1) lookup
    example_id_to_context = dict(zip(examples["id"], examples["context"]))

    # Map example_id to feature indices
    # This allows us to gather all windows belonging to a single document
    example_to_features = collections.defaultdict(list)
    for idx, row in features.iterrows():
        example_to_features[row["example_id"]].append(idx)

    predictions = {}

    # Iterate over each example in the original dataset
    for example_id in examples["id"]:
        # If no features generated for this example (unlikely), predict empty
        if example_id not in example_to_features:
            predictions[example_id] = ""
            continue

        feature_indices = example_to_features[example_id]
        context_text = example_id_to_context[example_id]
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # Retrieve inputs to determine context boundaries
            input_ids = features.iloc[feature_index]["input_ids"]
            offset_mapping = features.iloc[feature_index]["offset_mapping"]

            # Ensure input_ids is a list
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()

            # Find separators to locate context
            # XLM-R format: <s> Question </s> </s> Context </s>
            # We look for the sequence of separators (id=2)
            sep_indices = [i for i, t in enumerate(input_ids) if t == sep_token_id]

            # We need at least 2 separators to distinguish Question and Context
            # The context sits between the second </s> and the third </s> (or end of seq)
            if len(sep_indices) >= 2:
                # Context starts after the second </s> (index 1 in list)
                context_start_idx = sep_indices[1] + 1

                # Context ends at the next </s> if it exists, else end of content
                if len(sep_indices) >= 3:
                    context_end_idx = sep_indices[2]
                else:
                    context_end_idx = len(input_ids)
            else:
                # Malformed input or unexpected structure, skip this window
                continue

            # Get top-k indices for start and end
            # argsort sorts ascending, so we take the last n_best
            start_indexes = np.argsort(start_logits)[
                -1 : -Config.n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[
                -1 : -Config.n_best_size - 1 : -1
            ].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 1. Validate indices relative to each other
                    if start_index > end_index:
                        continue

                    # 2. Validate indices relative to context bounds
                    # This ensures we don't predict answers from the question or padding
                    if start_index < context_start_idx or end_index >= context_end_idx:
                        continue

                    # 3. Validate answer length
                    if end_index - start_index + 1 > Config.max_answer_length:
                        continue

                    # 4. Extract text
                    try:
                        # offset_mapping contains [start_char, end_char] for each token
                        # We take the start of the first token and end of the last token
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]

                        # Validate offsets (some special tokens might have None or 0,0)
                        if start_char is None or end_char is None:
                            continue

                        # Extract answer from the original context string
                        # This ensures perfect reconstruction of the answer text
                        answer = context_text[start_char:end_char]

                        # Score is sum of logits
                        score = start_logits[start_index] + end_logits[end_index]

                        valid_answers.append({"score": score, "text": answer})
                    except Exception:
                        continue

        # Select the best answer for this example across all windows
        if valid_answers:
            # Sort by score descending and pick the top one
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
            predictions[example_id] = best_answer["text"]
        else:
            # Fallback if no valid answer found
            predictions[example_id] = ""

    return predictions


def save_submission(predictions, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (dict): Dictionary {id: prediction_string}
        output_path (str): Path to save the CSV.
    """
    # Convert to DataFrame
    df = pd.DataFrame(list(predictions.items()), columns=["id", "PredictionString"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    # Pandas automatically handles quoting for strings containing special characters
    df.to_csv(output_path, index=False)
