import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein_distance,
    median_filter_predictions,
    decode_predictions,
)
from library.data_loader import get_data_loaders
from library.model import RSG_CRCN
from library.trainer import Trainer


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlation between error magnitude and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    errors = []
    seq_lengths = []
    num_gestures_gt = []

    with torch.no_grad():
        for features, labels, boundaries, mask in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(features, mask)

            # Use Stage 3 output
            p_cls_s3, _ = outputs["stage3"]  # (B, C, T)
            preds = torch.argmax(p_cls_s3, dim=1).cpu().numpy()
            gt_labels = labels.cpu().numpy()

            batch_size = preds.shape[0]
            lengths = mask.sum(dim=1).long().cpu().numpy()

            for i in range(batch_size):
                length = lengths[i]

                # Extract sequences
                pred_seq_raw = preds[i, :length]
                gt_seq_raw = gt_labels[i, :length]

                # Post-process
                pred_seq_smooth = median_filter_predictions(
                    pred_seq_raw, kernel_size=15
                )
                pred_gestures = decode_predictions(pred_seq_smooth)
                gt_gestures = decode_predictions(gt_seq_raw)

                # Compute Error
                dist = compute_levenshtein_distance(pred_gestures, gt_gestures)

                errors.append(dist)
                seq_lengths.append(length)
                num_gestures_gt.append(len(gt_gestures))

    # Compute Correlations
    if len(errors) > 1:
        corr_len, _ = pearsonr(errors, seq_lengths)
        corr_count, _ = pearsonr(errors, num_gestures_gt)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_count:.4f}")
    else:
        print("Insufficient data for correlation analysis.")


def generate_submission(model, test_loader, test_ids, device):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")
    model.eval()

    results = []

    with torch.no_grad():
        # Iterate through test loader
        # Note: test_loader yields batches. We need to map them to test_ids.
        # test_ids is a list of all IDs. We need to handle batching index carefully.

        current_idx = 0

        for features, _, _, mask in test_loader:
            features = features.to(device)
            mask = mask.to(device)

            outputs = model(features, mask)
            p_cls_s3, _ = outputs["stage3"]

            preds = torch.argmax(p_cls_s3, dim=1).cpu().numpy()
            lengths = mask.sum(dim=1).long().cpu().numpy()

            batch_size = preds.shape[0]

            for i in range(batch_size):
                length = lengths[i]
                sample_id = test_ids[current_idx]
                current_idx += 1

                pred_seq_raw = preds[i, :length]
                pred_seq_smooth = median_filter_predictions(
                    pred_seq_raw, kernel_size=15
                )
                pred_gestures = decode_predictions(pred_seq_smooth)

                # Format: Id, Sequence (space-separated)
                # Sanitize ID: "Sample00300" -> 300
                try:
                    id_num = int(sample_id.replace("Sample", ""))
                except ValueError:
                    id_num = sample_id

                # Serialize sequence
                label_str = " ".join(map(str, pred_gestures))
                results.append({"Id": id_num, "Sequence": label_str})

    # Save to file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df = pd.DataFrame(results)
    df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")


def main():
    # 1. Configuration Override for Fast Baseline
    # Limit epochs to ensure completion within time limit (2h)
    Config.NUM_EPOCHS = 25

    # Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = RSG_CRCN().to(device)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # 5. Metric Reporting
    final_metric = trainer.best_metric
    print(f"Final Validation Metric: {final_metric}")

    # Load best model for analysis and inference
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: Best model not found. Using current model state.")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.08548168249660787

    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, test_ids, device)
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
