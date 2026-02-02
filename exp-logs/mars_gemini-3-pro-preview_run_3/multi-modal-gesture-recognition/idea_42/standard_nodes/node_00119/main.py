import sys
import os
import torch
import numpy as np
import pandas as pd
import scipy.stats
from itertools import groupby

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.utils import decode_predictions, compute_levenshtein


def main():
    # ==========================================
    # 1. Configuration Override (Fast Baseline)
    # ==========================================
    # Reduce epochs to ensure execution finishes well within time limits
    Config.NUM_EPOCHS = 25
    print(f"Configuration: Running for {Config.NUM_EPOCHS} epochs.")

    # ==========================================
    # 2. Training
    # ==========================================
    trainer = Trainer()
    trainer.train(num_epochs=Config.NUM_EPOCHS)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Failure Analysis and Final Metric Calculation...")

    # Load the best model checkpoint
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
        print("Loaded best model checkpoint for analysis.")
    else:
        print("Warning: Best model not found, using current weights.")

    trainer.model.eval()
    val_loader = trainer.val_loader

    results = []
    total_dist = 0
    total_gt_gestures = 0

    with torch.no_grad():
        for x, y, sample_ids in val_loader:
            x = x.to(trainer.device)
            # y is shape (1, Time) containing label indices

            # Forward pass
            _, _, p3 = trainer.model(x)

            # Decode Predictions (Batch size is 1)
            # p3: (1, Time, Classes) -> squeeze to (Time, Classes)
            pred_seq = decode_predictions(
                p3.squeeze(0), min_duration=Config.MIN_GESTURE_DURATION
            )

            # Decode Ground Truth
            y_np = y.cpu().numpy().squeeze(0)
            gt_seq = []
            for k, g in groupby(y_np):
                if k != Config.BACKGROUND_CLASS_ID:
                    gt_seq.append(int(k))

            # Compute Metric
            dist = compute_levenshtein(pred_seq, gt_seq)

            total_dist += dist
            total_gt_gestures += len(gt_seq)

            # Collect stats for failure analysis
            seq_len = x.shape[1]
            num_gestures = len(gt_seq)

            results.append(
                {
                    "sample_id": sample_ids[0],
                    "lev_dist": dist,
                    "seq_len": seq_len,
                    "num_gestures": num_gestures,
                }
            )

    # Calculate Final Metric
    final_metric = total_dist / total_gt_gestures if total_gt_gestures > 0 else 0.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    df = pd.DataFrame(results)
    if len(df) > 1:
        # Correlation: Error Magnitude vs Sequence Length
        corr_len, _ = scipy.stats.pearsonr(df["lev_dist"], df["seq_len"])
        print(f"Correlation (Error Magnitude vs Sequence Length): {corr_len:.4f}")

        # Correlation: Error Magnitude vs Number of Gestures
        corr_num, _ = scipy.stats.pearsonr(df["lev_dist"], df["num_gestures"])
        print(f"Correlation (Error Magnitude vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # ==========================================
    # 4. Conditional Submission
    # ==========================================
    threshold = 0.2251
    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric:.5f}) is below threshold ({threshold}). Generating submission..."
        )
        trainer.predict_test()
    else:
        print(
            f"\nValidation metric ({final_metric:.5f}) is NOT below threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
