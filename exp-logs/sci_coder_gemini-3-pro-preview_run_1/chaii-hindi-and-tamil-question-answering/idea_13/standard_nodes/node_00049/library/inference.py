import pandas as pd
import numpy as np
import torch
import json
import os
import collections
from library.config import Config
from library.data import get_test_dataloader, get_tokenizer
from library.model import CustomXLMRoberta
from library.engine import predict_fn
from library.utils import seed_everything


def post_process_predictions(features_df, start_logits, end_logits, rel_logits):
    """
    Converts raw logits into final text predictions using the Gated Scoring formula.

    Args:
        features_df (pd.DataFrame): The dataframe containing feature metadata (offsets, sequence_ids).
        start_logits (np.ndarray): Averaged start logits.
        end_logits (np.ndarray): Averaged end logits.
        rel_logits (np.ndarray): Averaged relevance logits.

    Returns:
        dict: A mapping from example_id to the predicted answer string.
    """
    # Load original test data to get raw contexts for extraction
    test_df = pd.read_csv(Config.test_meta_path)
    id_to_context = dict(zip(test_df["id"], test_df["context"]))

    # Map example_id to list of feature indices (windows)
    example_to_features = collections.defaultdict(list)
    for idx, row in features_df.iterrows():
        example_to_features[row["example_id"]].append(idx)

    predictions = {}
    max_answer_length = 30
    n_best_size = 20

    # Iterate over each unique example
    for example_id, feature_indices in example_to_features.items():
        context = id_to_context.get(example_id, "")
        best_score = -float("inf")
        best_answer = ""

        if not feature_indices:
            predictions[example_id] = ""
            continue

        # Search across all windows for this example
        for feature_idx in feature_indices:
            start_log = start_logits[feature_idx]  # (Seq_Len,)
            end_log = end_logits[feature_idx]  # (Seq_Len,)
            rel_log = rel_logits[feature_idx]  # (1,)

            # Parse metadata stored as JSON strings
            offsets = json.loads(features_df.iloc[feature_idx]["offset_mapping"])
            sequence_ids = json.loads(features_df.iloc[feature_idx]["sequence_ids"])

            # Identify valid context tokens (sequence_id == 1)
            valid_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
            if not valid_indices:
                continue

            # Optimization: Only consider top-k start and end indices
            # argsort sorts ascending, so we take the last n_best_size
            start_candidates = np.argsort(start_log)[-n_best_size:]
            end_candidates = np.argsort(end_log)[-n_best_size:]

            for start_index in start_candidates:
                if start_index not in valid_indices:
                    continue

                for end_index in end_candidates:
                    if end_index not in valid_indices:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    # Gated Scoring Formula: (Start + End) + Relevance
                    # rel_log is an array of shape (1,), so we access [0]
                    score = start_log[start_index] + end_log[end_index] + rel_log[0]

                    if score > best_score:
                        best_score = score
                        try:
                            # Map token indices to character indices
                            char_start = offsets[start_index][0]
                            char_end = offsets[end_index][1]
                            best_answer = context[char_start:char_end]
                        except Exception:
                            continue

        predictions[example_id] = best_answer

    return predictions


def run_inference():
    """
    Main inference routine.
    1. Loads data.
    2. Loads multiple seed models and averages their logits.
    3. Post-processes logits to extract answers.
    4. Saves submission file.
    """
    seed_everything(Config.seed)

    # 1. Prepare Data
    tokenizer = get_tokenizer()
    # Ensure test features are cached/loaded
    test_dataloader, test_features_df = get_test_dataloader(
        tokenizer, load_cached_data=True
    )

    # 2. Initialize Accumulators
    num_samples = len(test_features_df)
    seq_len = Config.max_length

    avg_start_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_rel_logits = np.zeros((num_samples, 1), dtype=np.float32)

    seeds = Config.seeds
    device = Config.device
    models_found = 0

    print(f"Starting inference with ensemble seeds: {seeds}")

    # 3. Ensemble Loop
    for seed in seeds:
        model_path = os.path.join(Config.output_dir, f"model_seed_{seed}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading and predicting with seed {seed}...")

        # Load Model
        model = CustomXLMRoberta()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Predict
        s_log, e_log, r_log = predict_fn(test_dataloader, model, device)

        # Accumulate
        avg_start_logits += s_log
        avg_end_logits += e_log
        avg_rel_logits += r_log

        models_found += 1

        # Cleanup to save memory
        del model
        del state_dict
        torch.cuda.empty_cache()

    if models_found == 0:
        print("Error: No models found. Cannot generate predictions.")
        return

    # Average Logits
    avg_start_logits /= models_found
    avg_end_logits /= models_found
    avg_rel_logits /= models_found

    # 4. Post-Processing
    print("Post-processing predictions...")
    predictions_map = post_process_predictions(
        test_features_df, avg_start_logits, avg_end_logits, avg_rel_logits
    )

    # 5. Generate Submission CSV
    test_df = pd.read_csv(Config.test_meta_path)
    submission_data = []

    for _, row in test_df.iterrows():
        eid = row["id"]
        pred_str = predictions_map.get(eid, "")
        submission_data.append({"id": eid, "PredictionString": pred_str})

    submission_df = pd.DataFrame(submission_data)

    # Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(submission_df.head())
