import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

from library.config import Config
from library.model import HybridDeberta
from library.data import get_loaders


def predict_fold(fold_idx, test_loader, device):
    """
    Performs inference on the test set for a single fold model.

    Args:
        fold_idx (int): The index of the fold to predict.
        test_loader (DataLoader): The DataLoader for the test set.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: (list of ids, list of predicted scores)
    """
    model_path = os.path.join(Config.output_dir, f"model_fold_{fold_idx}.bin")

    # Check if model exists
    if not os.path.exists(model_path):
        print(
            f"Model for fold {fold_idx} not found at {model_path}. Skipping this fold."
        )
        return [], []

    print(f"Loading model for fold {fold_idx} from {model_path}...")

    # Initialize model structure
    model = HybridDeberta()

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    all_ids = []
    all_scores = []

    # Score values for expected value calculation: 0.0, 0.25, 0.5, 0.75, 1.0
    score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            ids = batch["id"]

            with autocast(enabled=True):
                logits = model(input_ids, attention_mask, features)
                probs = torch.softmax(logits, dim=1)

                # Calculate Expected Value: sum(prob * value)
                # This converts the classification distribution into a continuous score
                expected_scores = torch.sum(probs * score_values, dim=1)

            all_ids.extend(ids)
            all_scores.extend(expected_scores.cpu().numpy())

    return all_ids, all_scores


def generate_submission():
    """
    Main inference function.
    Loads data, iterates through trained folds, aggregates predictions via averaging,
    and saves the submission file.
    """
    print("Starting inference pipeline...")

    # 1. Setup
    device = Config.device
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 2. Get DataLoaders
    # We set load_cached_data=True to utilize processed data if available.
    # get_loaders returns (train, val, test). We only need test.
    print("Loading test data...")
    _, _, test_loader = get_loaders(tokenizer, load_cached_data=True)

    # 3. Inference Loop (Ensemble)
    id_score_sum = {}
    id_fold_count = {}

    # Determine folds to run based on Config
    folds = range(Config.n_folds)

    successful_folds = 0

    for fold_idx in folds:
        ids, scores = predict_fold(fold_idx, test_loader, device)

        # If model was missing or failed to load, skip
        if not ids:
            continue

        successful_folds += 1

        # Accumulate scores for averaging
        for uid, score in zip(ids, scores):
            if uid not in id_score_sum:
                id_score_sum[uid] = 0.0
                id_fold_count[uid] = 0
            id_score_sum[uid] += score
            id_fold_count[uid] += 1

    if successful_folds == 0:
        print("Error: No models were successfully loaded. Cannot generate submission.")
        return

    # 4. Aggregate and Save
    print(f"Aggregating predictions from {successful_folds} folds...")

    submission_data = []
    for uid, total_score in id_score_sum.items():
        # Simple average ensemble
        avg_score = total_score / id_fold_count[uid]
        submission_data.append({"id": uid, "score": avg_score})

    df_sub = pd.DataFrame(submission_data)

    # Ensure correct column order as per submission format
    df_sub = df_sub[["id", "score"]]

    # Save to the configured submission path
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    df_sub.to_csv(Config.submission_path, index=False)

    print(f"Submission saved to {Config.submission_path}")
    print(f"Total predictions generated: {len(df_sub)}")
    print("First 5 rows of submission:")
    print(df_sub.head())
