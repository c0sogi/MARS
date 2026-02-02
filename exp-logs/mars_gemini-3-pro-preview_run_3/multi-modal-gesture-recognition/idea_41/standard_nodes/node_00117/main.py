import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch.nn.functional as F

# Import from provided library
from library.config import Config
from library.data_loader import get_dataloaders
from library.model import PAKRNet
from library.train import Trainer
from library.utils import process_predictions, compute_levenshtein


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(results_df):
    """
    Analyzes the correlation between error (Levenshtein distance) and
    sequence properties (length, number of gestures).
    """
    print("\n=== Failure Analysis ===")
    if len(results_df) < 2:
        print("Not enough samples for correlation analysis.")
        return

    # Calculate correlations
    corr_len, _ = pearsonr(results_df["error"], results_df["num_frames"])
    corr_count, _ = pearsonr(results_df["error"], results_df["num_gestures"])

    print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Num Gestures): {corr_count:.4f}")

    # Identify worst cases
    worst_cases = results_df.sort_values("error", ascending=False).head(5)
    print("\nTop 5 Worst Predictions:")
    for _, row in worst_cases.iterrows():
        print(
            f"  ID: {row['sample_id']}, Error: {row['error']}, "
            f"Len: {row['num_frames']}, Gestures: {row['num_gestures']}"
        )
        print(f"    GT:   {row['gt_seq']}")
        print(f"    Pred: {row['pred_seq']}")


def generate_submission(model, test_loader, device, output_path):
    """
    Generates submission file for the test set.
    """
    print(f"\nGenerating submission to {output_path}...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (features, dummy_labels, sample_id)
            features, _, sample_ids = batch
            features = features.to(device)

            # Forward pass
            outputs = model(features)
            final_logits = outputs[-1]  # Use Stage 3 output

            # Get probabilities
            frame_probs = F.softmax(final_logits.squeeze(0), dim=1).cpu().numpy()

            # Decode
            pred_seq = process_predictions(frame_probs)

            # Format: SessionID,label1,label2,...
            sid = sample_ids[0]
            pred_str = ",".join(map(str, pred_seq))
            predictions.append(f"{sid},{pred_str}")

    # Write to file
    with open(output_path, "w") as f:
        for line in predictions:
            f.write(line + "\n")
    print("Submission generated successfully.")


def run_validation_analysis(model, val_loader, device):
    """
    Runs validation to compute the final metric and collect data for failure analysis.
    """
    model.eval()
    total_dist = 0
    total_gestures = 0

    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            features, labels, sample_ids = batch
            features = features.to(device)

            # Forward pass
            outputs = model(features)
            final_logits = outputs[-1]

            # Prediction
            frame_probs = F.softmax(final_logits.squeeze(0), dim=1).cpu().numpy()
            pred_seq = process_predictions(frame_probs)

            # Ground Truth
            gt_frame_ids = labels.squeeze(0).cpu().numpy()
            gt_seq = process_predictions(gt_frame_ids)

            # Metric
            dist = compute_levenshtein(pred_seq, gt_seq)
            n_gestures = len(gt_seq)

            total_dist += dist
            total_gestures += n_gestures

            # Collect data
            analysis_data.append(
                {
                    "sample_id": sample_ids[0],
                    "error": dist,
                    "num_frames": features.shape[1],
                    "num_gestures": n_gestures,
                    "pred_seq": pred_seq,
                    "gt_seq": gt_seq,
                }
            )

    final_metric = total_dist / total_gestures if total_gestures > 0 else 0.0
    return final_metric, pd.DataFrame(analysis_data)


def main():
    # 1. Setup and Configuration
    # Adjust config for fast baseline execution
    Config.NUM_EPOCHS = 50  # Ensure sufficient convergence
    Config.ensure_dirs()
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(
        f"Train batches: {len(train_loader)}, Val samples: {len(val_loader)}, Test samples: {len(test_loader)}"
    )

    # 3. Model Initialization
    model = PAKRNet().to(device)

    # 4. Training
    print("Initializing Trainer...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train()

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    else:
        print("Warning: No best model found. Using current model state.")

    val_metric, analysis_df = run_validation_analysis(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(analysis_df)

    # 7. Submission
    # Threshold check as per requirements
    THRESHOLD = 0.2251
    if val_metric < THRESHOLD:
        print(
            f"\nMetric ({val_metric:.4f}) is below threshold ({THRESHOLD}). Generating submission."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({val_metric:.4f}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
