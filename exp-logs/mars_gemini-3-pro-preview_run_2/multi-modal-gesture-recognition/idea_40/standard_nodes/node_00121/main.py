import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Import from provided library
from library.config import Config
from library.train import run_training
from library.inference import predict_all, post_process_sequence, generate_submission
from library.model import GMG_CRGN
from library.utils import load_checkpoint, set_seed
from library.data_loader import get_dataloaders

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def compute_validation_metric(model, device):
    """
    Computes the Levenshtein-based error rate on the validation set.
    """
    print("Loading validation data for metric calculation...")
    # Load validation metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        print(f"Validation metadata not found at {val_meta_path}")
        return float("inf"), []

    df_val = pd.read_csv(val_meta_path)
    # Parse labels
    df_val["labels"] = df_val["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )

    # Get DataLoader
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=False,  # Use full validation set
        load_cached_data=True,
    )

    print("Running inference on validation set...")
    # Get raw frame predictions
    raw_results = predict_all(model, val_loader, device)

    # Map results to dictionary for easy lookup
    results_map = {sid: preds for sid, preds in raw_results}

    total_distance = 0
    total_gestures = 0

    sample_errors = []

    for _, row in df_val.iterrows():
        sid = row["sample_id"]
        truth = row["labels"]

        if sid in results_map:
            raw_preds = results_map[sid]
            # Post-process to get gesture sequence
            pred_seq = post_process_sequence(raw_preds, kernel_size=9)

            # Compute distance
            dist = levenshtein_distance(truth, pred_seq)

            total_distance += dist
            total_gestures += len(truth)

            sample_errors.append(
                {
                    "sample_id": sid,
                    "error": dist,
                    "num_frames": row["num_frames"],
                    "num_gestures": len(truth),
                }
            )
        else:
            # Should not happen if dataloader and metadata are aligned
            print(f"Warning: Sample {sid} not found in predictions.")

    if total_gestures == 0:
        return float("inf"), sample_errors

    metric = total_distance / total_gestures
    return metric, sample_errors


def perform_failure_analysis(sample_errors):
    """
    Analyzes the correlation between errors and input features.
    """
    print("\n=== Failure Analysis ===")
    if not sample_errors:
        print("No error data available.")
        return

    df_errors = pd.DataFrame(sample_errors)

    # Correlation with Sequence Length (Num Frames)
    corr_frames, _ = pearsonr(df_errors["error"], df_errors["num_frames"])
    print(f"Correlation (Error vs NumFrames): {corr_frames:.4f}")

    # Correlation with Complexity (Num Gestures)
    corr_gestures, _ = pearsonr(df_errors["error"], df_errors["num_gestures"])
    print(f"Correlation (Error vs NumGestures): {corr_gestures:.4f}")

    print(
        "Systematic Error Pattern: Positive correlation indicates longer/complex sequences are harder."
    )


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Configure Fast Baseline
    # Limit epochs to ensure completion within 20 mins
    # 1700 samples * 10 epochs / 8 batch size ~ 2000 steps. Very fast on A100.
    Config.EPOCHS = 10

    print("Starting Fast Baseline Training...")

    # 3. Train
    run_training(debug=False, load_cached_data=True)

    # 4. Load Best Model for Evaluation
    print("\nLoading best model for evaluation...")
    model = GMG_CRGN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Training might have failed.")
        return

    load_checkpoint(model, checkpoint_path)
    model.eval()

    # 5. Compute Metric
    metric, sample_errors = compute_validation_metric(model, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    perform_failure_analysis(sample_errors)

    # 7. Conditional Submission
    THRESHOLD = 0.05699916177703269

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            checkpoint_name="best_model.pth",
            output_file="submission.csv",
            device=device,
            debug=False,
        )
    else:
        print(
            f"\nMetric ({metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
