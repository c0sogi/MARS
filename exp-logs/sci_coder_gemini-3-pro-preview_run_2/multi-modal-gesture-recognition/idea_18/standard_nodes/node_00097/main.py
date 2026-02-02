import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import groupby

from library.config import Config
from library.train import Trainer
from library.data_loader import get_dataloaders
from library.model import DSL_CRCN
from library.utils import decode_predictions, levenshtein_distance


def main():
    # 1. Configuration Overrides for Fast Baseline
    Config.MAX_EPOCHS = 25
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("Initializing training...")

    # 2. Train the Model
    trainer = Trainer()
    trainer.fit()

    # 3. Load Best Model for Analysis
    print("Loading best model for analysis...")
    device = torch.device(Config.DEVICE)
    model = DSL_CRCN().to(device)
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 4. Validation & Failure Analysis
    print("Running validation inference...")
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    val_preds = []
    val_targets = []
    val_lengths = []
    val_target_lengths = []
    val_distances = []

    with torch.no_grad():
        for features, targets, lengths in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            # Generate mask
            max_len = features.size(1)
            batch_size = features.size(0)
            idx_range = (
                torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )
            mask = idx_range < lengths.unsqueeze(1)

            # Forward
            outputs = model(features, mask=mask)
            # Stage 3 output is index 2
            stage3_logits = outputs[2]
            stage3_probs = torch.softmax(stage3_logits, dim=2).cpu().numpy()

            targets_np = targets.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            for i in range(batch_size):
                length = lengths_np[i]

                # Prediction
                valid_probs = stage3_probs[i, :length, :]
                pred_seq = decode_predictions(valid_probs)
                val_preds.append(pred_seq)

                # Target
                valid_target = targets_np[i, :length]
                target_seq = [
                    k
                    for k, g in groupby(valid_target)
                    if k != Config.BACKGROUND_CLASS_IDX
                ]
                val_targets.append(target_seq)

                # Metrics for Analysis
                dist = levenshtein_distance(pred_seq, target_seq)
                val_distances.append(dist)
                val_lengths.append(length)
                val_target_lengths.append(len(target_seq))

    # Compute Final Metric
    total_distance = sum(val_distances)
    total_ref_length = sum(val_target_lengths)
    final_metric = (
        total_distance / total_ref_length if total_ref_length > 0 else float("inf")
    )

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(val_distances) > 1:
        # Correlation with Input Sequence Length
        corr_len, _ = pearsonr(val_distances, val_lengths)
        print(f"Correlation (Error vs Input Length): {corr_len:.4f}")

        # Correlation with Number of Gestures
        corr_num, _ = pearsonr(val_distances, val_target_lengths)
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 5. Submission
    threshold = 0.08548168249660787
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric:.6f}) is below threshold ({threshold}). Generating submission..."
        )

        # Load Test Metadata to get IDs
        test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_df = pd.read_csv(test_meta_path)
        sample_ids = test_df["sample_id"].tolist()

        submission_lines = []

        # Inference on Test Set
        # Note: DataLoader preserves order if shuffle=False
        batch_idx = 0
        current_sample_idx = 0

        with torch.no_grad():
            for features, targets, lengths in test_loader:
                features = features.to(device)
                lengths = lengths.to(device)

                # Mask
                max_len = features.size(1)
                batch_size = features.size(0)
                idx_range = (
                    torch.arange(max_len, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                mask = idx_range < lengths.unsqueeze(1)

                outputs = model(features, mask=mask)
                stage3_logits = outputs[2]
                stage3_probs = torch.softmax(stage3_logits, dim=2).cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(batch_size):
                    length = lengths_np[i]
                    valid_probs = stage3_probs[i, :length, :]
                    pred_seq = decode_predictions(valid_probs)

                    # Format: SessionID,Label1,Label2,...
                    sid = sample_ids[current_sample_idx]
                    labels_str = ",".join(map(str, pred_seq))
                    line = f"{sid},{labels_str}" if labels_str else f"{sid},"
                    submission_lines.append(line)

                    current_sample_idx += 1

        # Save Submission
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
