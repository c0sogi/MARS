import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein_score,
    post_process_predictions,
    levenshtein_distance,
)
from library.dataset import GestureDataset
from library.trainer import Trainer


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Optimized Run
    Config.NUM_EPOCHS = 40
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 4

    # Setup directories and seeds
    Config.setup()
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Training
    # =========================================================================
    trainer = Trainer()
    trainer.fit()

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    # Load the best model for evaluation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    trainer.model.load_state_dict(
        torch.load(best_model_path, map_location=trainer.device)
    )
    trainer.model.eval()

    all_preds = []
    all_targets = []

    # For failure analysis
    sample_errors = []
    sample_lengths = []  # Number of frames
    sample_target_counts = []  # Number of gestures

    with torch.no_grad():
        for features, targets, mask in trainer.val_loader:
            features = features.to(trainer.device)
            targets = targets.to(trainer.device)
            mask = mask.to(trainer.device)

            # Forward pass
            outputs = trainer.model(features, mask)
            final_stage_output = outputs[-1]

            # Get predictions
            batch_preds = post_process_predictions(
                final_stage_output, median_window=Config.MEDIAN_WINDOW_SIZE
            )

            # Process targets and collect metadata for analysis
            targets_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for i in range(targets_np.shape[0]):
                # 1. Extract valid target sequence
                valid_len = int(np.sum(mask_np[i]))
                t_seq = targets_np[i, :valid_len]

                decoded_target = []
                prev = -1
                for val in t_seq:
                    if val != prev:
                        if val != 0:
                            decoded_target.append(int(val))
                        prev = val

                all_targets.append(decoded_target)

                # 2. Compute Error for this sample
                # Note: batch_preds[i] corresponds to this target
                pred_seq = batch_preds[i]
                dist = levenshtein_distance(pred_seq, decoded_target)

                sample_errors.append(dist)
                sample_lengths.append(valid_len)
                sample_target_counts.append(len(decoded_target))

            all_preds.extend(batch_preds)

    # Compute Final Metric
    final_metric = compute_levenshtein_score(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if len(sample_errors) > 1:
        # Correlation with Sequence Length (Frames)
        corr_frames, _ = pearsonr(sample_errors, sample_lengths)
        print(f"Correlation (Error vs NumFrames): {corr_frames}")

        # Correlation with Sequence Complexity (Number of Gestures)
        corr_gestures, _ = pearsonr(sample_errors, sample_target_counts)
        print(f"Correlation (Error vs NumGestures): {corr_gestures}")

    # =========================================================================
    # 4. Submission
    # =========================================================================
    # Generate submission only if metric is good enough
    if final_metric < 0.424:
        print("Validation metric meets threshold. Generating submission...")
        trainer.predict()
    else:
        print(
            f"Validation metric {final_metric} is not lower than 0.424. Skipping submission."
        )


if __name__ == "__main__":
    main()
