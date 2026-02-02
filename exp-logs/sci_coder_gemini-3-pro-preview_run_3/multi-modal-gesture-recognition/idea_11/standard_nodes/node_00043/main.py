import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 1. Configuration Setup (Must be before other library imports)
from library.config import Config

# Override Config for Fast Baseline
Config.EPOCHS = 25  # Limit epochs for speed (Baseline requirement)
Config.DEBUG = False  # Use full dataset to aim for the score threshold

# Import Library Modules
from library.utils import set_seed, compute_levenshtein, collapse_predictions
from library.trainer import Trainer
from library.data_loader import get_dataloaders
from library.inference import run_inference


def main():
    # Set reproducible seeds
    set_seed(Config.SEED)

    print("Initializing Baseline Run...")

    # 2. Training
    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    # This will train for Config.EPOCHS and save 'best_model.pth' based on window-level validation loss
    trainer.fit(epochs=Config.EPOCHS, load_cached_data=True)

    # 3. Rigorous Validation (Sequence Level)
    print("Performing Final Validation on Full Sequences...")

    # Load the best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current weights.")

    trainer.model.eval()

    # Get Validation Loader (cached)
    # We need the dataset to access raw sequence structures for reconstruction
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True, debug=Config.DEBUG)
    val_ds = val_loader.dataset
    raw_skeletons = val_ds.skeletons

    # Initialize buffers for sequence reconstruction
    # seq_probs: List of (T, NumClasses) arrays
    seq_probs = [
        np.zeros((len(s), Config.NUM_CLASSES), dtype=np.float32) for s in raw_skeletons
    ]
    # seq_counts: List of (T,) arrays for averaging overlaps
    seq_counts = [np.zeros((len(s),), dtype=np.float32) for s in raw_skeletons]

    current_window_idx = 0

    # Inference Loop on Validation Set
    with torch.no_grad():
        for features, _, _ in val_loader:
            features = features.to(Config.DEVICE)
            batch_size = features.size(0)

            # Forward Pass
            outputs = trainer.model(features)
            # Use Stage 3 output
            logits = outputs["stage3_cls"]
            probs = (
                torch.softmax(logits, dim=1).cpu().numpy()
            )  # Shape: (Batch, Classes, Window)

            # Map windows back to sequences
            for b in range(batch_size):
                global_idx = current_window_idx + b
                if global_idx >= len(val_ds.windows):
                    break

                seq_idx, start_frame = val_ds.windows[global_idx]

                # Transpose to (Window, Classes)
                window_probs = probs[b].transpose(1, 0)

                # Determine valid range
                seq_len = len(raw_skeletons[seq_idx])
                end_frame = min(start_frame + Config.WINDOW_SIZE, seq_len)
                valid_len = end_frame - start_frame

                if valid_len > 0:
                    seq_probs[seq_idx][start_frame:end_frame] += window_probs[
                        :valid_len
                    ]
                    seq_counts[seq_idx][start_frame:end_frame] += 1.0

            current_window_idx += batch_size

    # Compute Metrics & Failure Analysis Data
    total_dist = 0
    total_len = 0

    errors = []
    lengths = []
    num_gestures = []

    for i in range(len(raw_skeletons)):
        # Normalize probabilities
        counts = seq_counts[i][:, None]
        counts[counts == 0] = 1.0  # Avoid div by zero
        avg_probs = seq_probs[i] / counts

        # Decode
        frame_preds = np.argmax(avg_probs, axis=1)
        pred_seq = collapse_predictions(frame_preds)

        # Ground Truth
        target_seq = collapse_predictions(val_ds.class_labels[i])

        # Metric
        dist = compute_levenshtein(pred_seq, target_seq)
        total_dist += dist
        total_len += len(target_seq)

        # Store for analysis
        errors.append(dist)
        lengths.append(len(raw_skeletons[i]))
        num_gestures.append(len(target_seq))

    final_metric = total_dist / total_len if total_len > 0 else 1.0

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric:.10f}")

    # 4. Failure Analysis
    print("-" * 30)
    print("Failure Analysis")

    if len(errors) > 1:
        # Correlation with Sequence Length
        corr_len, _ = pearsonr(errors, lengths)
        print(f"Correlation between Error and Sequence Length: {corr_len:.4f}")

        # Correlation with Number of Gestures
        corr_num, _ = pearsonr(errors, num_gestures)
        print(f"Correlation between Error and Num Gestures: {corr_num:.4f}")
    else:
        print("Insufficient data for correlation analysis.")
    print("-" * 30)

    # 5. Submission
    # Generate submission only if metric is good enough
    THRESHOLD = 0.2251
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric:.4f} < {THRESHOLD}. Generating Submission...")
        run_inference(load_cached_data=True)
    else:
        print(f"Metric {final_metric:.4f} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
