import sys
import os
import pandas as pd
import numpy as np
import torch
import nltk
from scipy.stats import pearsonr
from itertools import groupby

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import (
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    VAL_METADATA_PATH,
    SEED,
    BATCH_SIZE,
    DEVICE,
)
from library.utils import set_seed
from library.train import run_training
from library.inference import Predictor, run_inference
from library.data_loader import GestureDataset, collate_fn
from torch.utils.data import DataLoader


def calculate_levenshtein_error(predicted_seqs, truth_seqs):
    """
    Computes the Levenshtein distance error rate.
    Metric = Sum(LevenshteinDist) / Sum(NumGesturesInTruth)
    """
    total_distance = 0
    total_gestures = 0

    errors = []

    for pred, truth in zip(predicted_seqs, truth_seqs):
        # nltk.edit_distance computes Levenshtein distance
        dist = nltk.edit_distance(pred, truth)
        total_distance += dist
        total_gestures += len(truth)
        errors.append(dist)

    if total_gestures == 0:
        return float("inf"), errors

    metric = total_distance / total_gestures
    return metric, errors


def get_ground_truth_from_metadata(metadata_path):
    """
    Loads the ground truth sequences directly from the metadata CSV.
    This ensures we handle repeated gestures correctly (e.g., '2 2') which
    might be merged if we reconstructed solely from frame-wise labels.
    """
    df = pd.read_csv(metadata_path)

    # Convert space-separated string labels to lists of integers
    truth_seqs = []
    for x in df["labels"]:
        if pd.notna(x) and str(x).strip() != "":
            seq = [int(i) for i in str(x).split()]
        else:
            seq = []
        truth_seqs.append(seq)

    return df, truth_seqs


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Training
    # We use all data (max_samples=None) but limit epochs to 20 for a fast baseline
    # that is still capable of reaching the target score.
    run_training(
        max_samples=None,
        num_epochs=20,
        batch_size=BATCH_SIZE,
        load_cached_data=True,
        augment=True,
    )

    # 3. Validation & Metric Calculation

    # Load Validation Dataset
    val_dataset = GestureDataset(split="val", load_cached_data=True, augment=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Must be False to align with metadata
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Initialize Predictor with the best model
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    predictor = Predictor(best_model_path, torch.device(DEVICE))

    # Run Inference on Validation Set
    pred_seqs = predictor.predict(val_loader)

    # Get Ground Truth from Metadata
    val_df, truth_seqs = get_ground_truth_from_metadata(VAL_METADATA_PATH)

    # Ensure alignment (in case dataset skipped corrupted files, though unlikely)
    if len(pred_seqs) != len(truth_seqs):
        # If lengths differ, we assume the dataset loaded a subset.
        # Since GestureDataset processes sequentially, we slice the metadata.
        truth_seqs = truth_seqs[: len(pred_seqs)]
        val_df = val_df.iloc[: len(pred_seqs)]

    # Compute Metric
    metric, errors = calculate_levenshtein_error(pred_seqs, truth_seqs)

    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis

    # Collect features for correlation
    # Features: Sequence Length (frames), Number of Gestures (truth)
    lengths = val_df["num_frames"].tolist()
    num_gestures = [len(seq) for seq in truth_seqs]

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "length": lengths, "num_gestures": num_gestures}
    )

    # Calculate Correlations
    corr_len, _ = pearsonr(df_analysis["error"], df_analysis["length"])
    corr_gest, _ = pearsonr(df_analysis["error"], df_analysis["num_gestures"])

    print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Num Gestures): {corr_gest:.4f}")

    # 5. Submission
    threshold = 0.06789606035205364

    if metric < threshold:
        run_inference(
            checkpoint_name="best_model.pth",
            batch_size=BATCH_SIZE,
            load_cached_data=True,
            output_filename="submission.csv",
        )


if __name__ == "__main__":
    main()
