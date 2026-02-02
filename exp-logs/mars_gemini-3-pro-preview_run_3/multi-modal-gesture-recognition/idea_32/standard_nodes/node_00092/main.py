import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config, trainer, utils


def set_reproducibility(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_failure_analysis(val_preds, val_counts, val_dataset):
    """
    Analyzes prediction errors on the validation set.
    Computes correlations between error magnitude and input features.
    """
    print("Performing Failure Analysis on Validation Set...")

    analysis_records = []

    # Iterate over all validation samples
    for i in range(len(val_dataset.ids)):
        sample_id = val_dataset.ids[i]

        # Check if we have predictions for this sample
        if i not in val_preds:
            continue

        # Reconstruct prediction from accumulated probabilities
        # Clamp count to avoid division by zero
        avg_probs = val_preds[i] / val_counts[i].clamp(min=1)

        # Get frame-wise class labels
        pred_labels_frame = torch.argmax(avg_probs, dim=1).cpu().numpy()

        # Process into sequence (RLE + filtering) using default config
        pred_seq = utils.process_gesture_sequence(pred_labels_frame)

        # Get Ground Truth
        gt_labels_frame = val_dataset.labels[i]
        # Process GT sequence similarly to ensure consistency with metric calculation
        gt_seq = utils.process_gesture_sequence(gt_labels_frame)

        # Calculate Levenshtein Distance (Error)
        dist = utils.levenshtein_distance(pred_seq, gt_seq)

        # Extract Meta-features
        seq_len_frames = len(gt_labels_frame)
        num_gt_gestures = len(gt_seq)

        analysis_records.append(
            {
                "sample_id": sample_id,
                "error": dist,
                "seq_len": seq_len_frames,
                "num_gestures": num_gt_gestures,
            }
        )

    df = pd.DataFrame(analysis_records)

    if not df.empty:
        # Correlation: Error vs Sequence Length
        corr_len, _ = scipy.stats.pearsonr(df["error"], df["seq_len"])
        print(f"Correlation (Error vs Sequence Length): {corr_len}")

        # Correlation: Error vs Number of Gestures
        corr_num, _ = scipy.stats.pearsonr(df["error"], df["num_gestures"])
        print(f"Correlation (Error vs Num Gestures): {corr_num}")

        print("Failure analysis complete.")
    else:
        print("No validation data found for analysis.")


def main():
    # 1. Set Seeds
    set_reproducibility(config.SEED)

    # 2. Initialize Trainer
    # This automatically loads cached data and initializes the GHCMN model
    print("Initializing Trainer...")
    ghcmn_trainer = trainer.Trainer()

    # 3. Train Model
    # Executes the training loop defined in library/trainer.py
    # Uses config.NUM_EPOCHS (60) which is efficient for this dataset size
    print("Starting Training...")
    ghcmn_trainer.train()

    # 4. Load Best Model for Validation
    # Ensure we evaluate the best checkpoint saved during training
    print("Loading best model for evaluation...")
    if os.path.exists(config.BEST_MODEL_PATH):
        ghcmn_trainer.model.load_state_dict(
            torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current weights.")

    # 5. Compute Final Validation Metric
    print("Computing validation metrics...")
    # validate returns: avg_loss, score, seq_preds, seq_counts
    # compute_metric=True ensures Levenshtein distance is calculated
    val_loss, val_score, val_preds, val_counts = ghcmn_trainer.validate(
        ghcmn_trainer.val_loader, compute_metric=True
    )

    # REQUIRED OUTPUT: Print the metric
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    # Analyze systematic errors
    run_failure_analysis(val_preds, val_counts, ghcmn_trainer.val_loader.dataset)

    # 7. Conditional Submission
    # Generate submission only if metric is below threshold
    submission_threshold = 0.2251
    if val_score < submission_threshold:
        print(
            f"Validation score {val_score} is below threshold {submission_threshold}. Generating submission..."
        )
        ghcmn_trainer.generate_submission()
    else:
        print(
            f"Validation score {val_score} is not below threshold {submission_threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
