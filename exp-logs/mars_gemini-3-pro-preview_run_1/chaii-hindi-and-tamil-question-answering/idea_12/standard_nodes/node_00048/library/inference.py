import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

from library.config import Config
from library.model import XLMRobertaForQA
from library.data import get_data
from library.utils import set_seed


def load_models(cfg, device):
    """
    Loads all trained seed models defined in the configuration.
    """
    models = []
    for seed in cfg.seeds:
        model_path = os.path.join(cfg.output_dir, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        model = XLMRobertaForQA(cfg.model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError(
            "No models loaded. Ensure training has completed and models are saved."
        )

    return models


def get_best_span(
    start_logits, end_logits, sequence_ids, relevance_logit, offset_mapping, context
):
    """
    Finds the optimal span within a single window using the Gated Scoring formula.

    Args:
        start_logits (np.array): Logits for start positions.
        end_logits (np.array): Logits for end positions.
        sequence_ids (list): Sequence IDs (1 indicates context).
        relevance_logit (float): Logit indicating answer presence.
        offset_mapping (list): List of (start_char, end_char) tuples.
        context (str): The original context text.

    Returns:
        tuple: (best_score, predicted_text)
    """
    # Constants
    MAX_ANSWER_LEN = 40  # Maximum token length for an answer
    MIN_SCORE = -1e9

    # Mask non-context tokens
    # sequence_ids are sanitized to -1 for None, 0 for question, 1 for context
    context_mask = np.array([1 if s == 1 else 0 for s in sequence_ids])

    # If no context tokens, return low score
    if np.sum(context_mask) == 0:
        return MIN_SCORE, ""

    # Apply mask to logits (set non-context to very negative number)
    start_logits = np.where(context_mask, start_logits, MIN_SCORE)
    end_logits = np.where(context_mask, end_logits, MIN_SCORE)

    # Find best span (s, e)
    # We iterate through all valid start positions and valid end positions
    # satisfying s <= e and e - s < MAX_ANSWER_LEN

    # Get indices sorted by probability to prune search space (optimization)
    # or just brute force since 384 is small. Brute force is safer for correctness.

    best_span_score = MIN_SCORE
    best_s, best_e = 0, 0

    # Get valid context indices
    valid_indices = np.where(context_mask)[0]
    if len(valid_indices) == 0:
        return MIN_SCORE, ""

    start_idx_min = valid_indices[0]
    end_idx_max = valid_indices[-1]

    # Optimization: Only look at top K starts to speed up if needed,
    # but for 384 tokens, O(N*M) is fine where M is max_len.

    # Vectorized approach for efficiency
    # Create a matrix of scores: score[s, e] = start[s] + end[e]
    # We only care about the band where 0 <= e - s < MAX_ANSWER_LEN

    # However, simple loop is readable and fast enough for inference
    for s in valid_indices:
        # We only check ends within [s, s + MAX_ANSWER_LEN)
        max_e = min(s + MAX_ANSWER_LEN, end_idx_max + 1)

        # Slice valid ends
        valid_ends = np.arange(s, max_e)
        if len(valid_ends) == 0:
            continue

        # Filter ends that are actually context (though our range logic mostly handles this)
        # Assuming contiguous context in sliding window, which is true for XLM-R

        current_end_logits = end_logits[valid_ends]
        scores = start_logits[s] + current_end_logits

        max_idx = np.argmax(scores)
        current_best_score = scores[max_idx]

        if current_best_score > best_span_score:
            best_span_score = current_best_score
            best_s = s
            best_e = valid_ends[max_idx]

    # Gated Score Calculation
    # Score = (Start_Logit + End_Logit) + Relevance_Logit
    final_score = best_span_score + relevance_logit

    # Reconstruct text
    try:
        start_char = offset_mapping[best_s][0]
        end_char = offset_mapping[best_e][1]
        predicted_text = context[start_char:end_char]
    except Exception:
        predicted_text = ""

    return final_score, predicted_text


def predict_and_aggregate(cfg, test_dataset, test_features):
    """
    Runs inference on the test set using the ensemble and aggregates logits.
    """
    device = cfg.device
    models = load_models(cfg, device)

    loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Store aggregated predictions
    # Key: feature_index (int), Value: dict of averaged logits
    all_predictions = {}

    print("Starting ensemble inference...")

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feature_indices = batch["feature_idx"].cpu().numpy()

            # Accumulators for this batch
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)

            avg_start_logits = torch.zeros(batch_size, seq_len, device=device)
            avg_end_logits = torch.zeros(batch_size, seq_len, device=device)
            avg_rel_logits = torch.zeros(batch_size, device=device)

            for model in models:
                outputs = model(input_ids, attention_mask=attention_mask)

                avg_start_logits += outputs["start_logits"]
                avg_end_logits += outputs["end_logits"]
                avg_rel_logits += outputs["relevance_logits"]

            # Average
            num_models = len(models)
            avg_start_logits /= num_models
            avg_end_logits /= num_models
            avg_rel_logits /= num_models

            # Move to CPU
            avg_start_logits = avg_start_logits.cpu().numpy()
            avg_end_logits = avg_end_logits.cpu().numpy()
            avg_rel_logits = avg_rel_logits.cpu().numpy()

            # Store
            for i, feature_idx in enumerate(feature_indices):
                all_predictions[feature_idx] = {
                    "start_logits": avg_start_logits[i],
                    "end_logits": avg_end_logits[i],
                    "relevance_logit": avg_rel_logits[i],
                }

    return all_predictions


def post_processing(cfg, all_predictions, test_features):
    """
    Processes aggregated predictions to generate final answer strings.
    Groups windows by example_id and selects the span with the global max score.
    """
    print("Post-processing predictions...")

    # Group features by example_id
    # example_id -> list of (feature_idx, feature_dict)
    example_groups = defaultdict(list)
    for i, feature in enumerate(test_features):
        example_groups[feature["example_id"]].append((i, feature))

    final_results = []

    for example_id, group in example_groups.items():
        best_global_score = -float("inf")
        best_global_text = ""

        # Iterate over all sliding windows for this document
        for feature_idx, feature in group:
            if feature_idx not in all_predictions:
                continue

            preds = all_predictions[feature_idx]

            # Get span score and text for this window
            score, text = get_best_span(
                start_logits=preds["start_logits"],
                end_logits=preds["end_logits"],
                sequence_ids=feature["sequence_ids"],
                relevance_logit=preds["relevance_logit"],
                offset_mapping=feature["offset_mapping"],
                context=feature["context"],
            )

            # Update global best for this document
            if score > best_global_score:
                best_global_score = score
                best_global_text = text

        # Fallback for empty predictions or failures
        if best_global_text is None:
            best_global_text = ""

        # Clean up text (basic whitespace)
        best_global_text = best_global_text.strip()

        final_results.append({"id": example_id, "PredictionString": best_global_text})

    return pd.DataFrame(final_results)


def generate_submission(cfg: Config):
    """
    Main function to generate the submission file.
    """
    # 1. Load Data
    # We don't need training data here, so we just get test data
    # Note: get_data returns (train_ds, test_ds, test_features)
    # We can ignore train_ds.
    _, test_dataset, test_features = get_data(cfg, load_cached_data=True)

    # 2. Inference
    all_predictions = predict_and_aggregate(cfg, test_dataset, test_features)

    # 3. Post-processing
    submission_df = post_processing(cfg, all_predictions, test_features)

    # 4. Save
    # Ensure columns are in correct order
    submission_df = submission_df[["id", "PredictionString"]]

    save_path = os.path.join(cfg.submission_dir, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(submission_df.head())
