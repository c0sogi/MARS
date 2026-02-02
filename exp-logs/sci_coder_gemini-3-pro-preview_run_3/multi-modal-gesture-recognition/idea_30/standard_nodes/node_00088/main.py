import os
import sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

# Import provided libraries
from library import config, trainer, utils, data_loader


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(config.SEED)

    # Initialize Trainer
    # This handles data loading and model initialization internally
    print("Initializing Trainer...")
    model_trainer = trainer.Trainer()

    # 2. Training
    # Run the training loop
    print("Starting Training...")
    model_trainer.fit()

    # 3. Validation & Evaluation
    print("Evaluating on Validation Set...")

    # Load the best model weights for evaluation
    if os.path.exists(config.BEST_MODEL_PATH):
        model_trainer.model.load_state_dict(
            torch.load(config.BEST_MODEL_PATH, map_location=model_trainer.device)
        )

    # Get predictions on validation set
    # We use is_test=True to get the raw predictions list
    val_predictions = model_trainer.evaluate(model_trainer.val_loader, is_test=True)

    # Get Ground Truth from the dataset
    val_dataset = model_trainer.val_loader.dataset
    val_labels_raw = val_dataset.labels  # List of frame-wise label arrays

    # Calculate Metric and perform Failure Analysis
    total_distance = 0
    total_ref_gestures = 0

    # Lists for failure analysis
    sample_errors = []
    sample_lengths = []
    sample_num_gestures = []

    for i, pred_seq in enumerate(val_predictions):
        # Decode Ground Truth Sequence
        # Use min_length=1 to capture all annotated gestures in GT
        gt_frame_labels = val_labels_raw[i]
        gt_seq = utils.process_predictions(gt_frame_labels, min_length=1)

        # Calculate Distance
        dist = utils.levenshtein_distance(pred_seq, gt_seq)

        # Update Globals
        n_ref = len(gt_seq)
        total_distance += dist
        total_ref_gestures += n_ref

        # Collect stats for failure analysis
        # Error magnitude (Distance)
        sample_errors.append(dist)
        # Feature 1: Sequence Length (Frames)
        sample_lengths.append(len(gt_frame_labels))
        # Feature 2: Number of Gestures (Complexity)
        sample_num_gestures.append(n_ref)

    # Compute Final Metric
    if total_ref_gestures > 0:
        final_metric = total_distance / total_ref_gestures
    else:
        final_metric = float("inf")

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(sample_errors) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, _ = pearsonr(sample_errors, sample_lengths)
        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")

        # Correlation: Error vs Num Gestures
        corr_num, _ = pearsonr(sample_errors, sample_num_gestures)
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Identify worst performers
        # Error Rate per sample = dist / max(1, n_ref)
        error_rates = [
            d / max(1, n) for d, n in zip(sample_errors, sample_num_gestures)
        ]
        worst_idx = np.argmax(error_rates)
        print(
            f"Worst Sample Index: {worst_idx}, Error Rate: {error_rates[worst_idx]:.2f}, Dist: {sample_errors[worst_idx]}, GT Count: {sample_num_gestures[worst_idx]}"
        )
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 5. Submission Generation
    # Threshold check
    THRESHOLD = 0.2251
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        model_trainer.predict()
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
