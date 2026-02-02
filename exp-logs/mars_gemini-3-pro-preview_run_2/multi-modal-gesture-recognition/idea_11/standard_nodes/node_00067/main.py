import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch.nn.functional as F

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    levenshtein_distance,
    decode_predictions,
    median_filter_1d,
)
from library.trainer import Trainer
from library.data_loader import get_dataloaders
from library.model import NMD_CRCN


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Training
    # The Trainer class handles the training loop, early stopping, and saving the best model.
    trainer = Trainer()
    print("Starting training...")
    trainer.fit()

    # 3. Evaluation on Validation Set (using Best Model)
    print("\nEvaluating best model on validation set...")

    # Load best model architecture
    model = NMD_CRCN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Training might have failed.")
        return

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get data loaders (using cached data for speed)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_targets = []

    # Storage for Failure Analysis
    sample_metrics = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]
            sample_ids = batch["sample_ids"]

            # Inference
            outputs = model(features, mask)

            # Use Stage 3 output for final predictions
            logits = outputs["stage3"]
            probs = F.softmax(logits, dim=2)
            preds_batch = torch.argmax(probs, dim=2).cpu().numpy()
            targets_batch = labels.cpu().numpy()

            for i in range(len(lengths)):
                length = lengths[i]
                sid = sample_ids[i]

                # Extract valid sequence (ignoring padding)
                raw_pred = preds_batch[i, :length]
                raw_target = targets_batch[i, :length]

                # Apply Median Filter to smooth predictions
                filtered_pred = median_filter_1d(raw_pred, kernel_size=7)

                # Decode to gesture list
                decoded_pred = decode_predictions(filtered_pred, background_class=0)
                decoded_target = decode_predictions(raw_target, background_class=0)

                all_preds.append(decoded_pred)
                all_targets.append(decoded_target)

                # Calculate sample-level metric for analysis
                dist = levenshtein_distance(decoded_pred, decoded_target)

                # Collect stats
                sample_metrics.append(
                    {
                        "sample_id": sid,
                        "levenshtein_dist": dist,
                        "seq_length": length.item(),
                        "num_gestures_target": len(decoded_target),
                        "num_gestures_pred": len(decoded_pred),
                    }
                )

    # Compute Global Metric
    final_metric = compute_normalized_levenshtein(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(sample_metrics)

    if not df_analysis.empty:
        # Correlation: Error vs Sequence Length
        # Checks if longer sequences are harder to predict
        if len(df_analysis) > 1:
            corr_len, _ = pearsonr(
                df_analysis["levenshtein_dist"], df_analysis["seq_length"]
            )
            print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")

            # Correlation: Error vs Num Gestures
            # Checks if sequences with more gestures are harder
            corr_num, _ = pearsonr(
                df_analysis["levenshtein_dist"], df_analysis["num_gestures_target"]
            )
            print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Display samples with highest errors
        print("\nTop 5 High Error Samples:")
        print(
            df_analysis.sort_values(by="levenshtein_dist", ascending=False).head(5)[
                ["sample_id", "levenshtein_dist", "num_gestures_target"]
            ]
        )
    else:
        print("No analysis data available.")

    # 5. Submission Logic
    # Only generate submission if metric is better than the threshold
    THRESHOLD = 0.10854816824966079

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
