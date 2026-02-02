import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    decode_predictions_to_gestures,
    median_filter_predictions,
)
from library.data_loader import get_dataloaders
from library.model import DW_AIIN
from library.train import run_training
from library.predict import run_prediction


def analyze_failures_and_validate(model, val_loader, device):
    """
    Computes the final validation metric and performs failure analysis.
    """
    model.eval()

    # Storage for analysis
    sample_errors = []
    sample_lengths = []
    sample_audio_energy = []
    sample_skel_energy = []

    total_levenshtein_dist = 0
    total_gt_gestures = 0

    print("Running Validation and Failure Analysis...")

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            # Labels are needed for GT
            labels = batch["labels"].cpu().numpy()

            # Inference
            logits = model(skeleton, audio, lengths)

            # Decode predictions
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()

            # Iterate batch
            for i in range(len(lengths)):
                seq_len = lengths[i].item()

                # Extract valid sequence
                valid_pred = preds[i, :seq_len]
                valid_label = labels[i, :seq_len]

                # Smooth and Decode
                smoothed_pred = median_filter_predictions(
                    valid_pred, window_size=Config.MEDIAN_FILTER_WINDOW
                )

                pred_gestures = decode_predictions_to_gestures(
                    smoothed_pred,
                    background_label=Config.BACKGROUND_LABEL,
                    min_length=Config.MIN_GESTURE_LENGTH,
                )

                # GT Decoding (filter background, but keep short gestures if they exist in GT)
                gt_gestures = decode_predictions_to_gestures(
                    valid_label, background_label=Config.BACKGROUND_LABEL, min_length=1
                )

                # Compute Metric
                dist = levenshtein_distance(pred_gestures, gt_gestures)

                total_levenshtein_dist += dist
                total_gt_gestures += len(gt_gestures)

                # Failure Analysis Data
                # Normalize error by GT length (or 1 to avoid div/0)
                denom = len(gt_gestures) if len(gt_gestures) > 0 else 1
                norm_error = dist / denom

                sample_errors.append(norm_error)
                sample_lengths.append(seq_len)

                # Compute simple energy features
                # Audio: Mean absolute amplitude
                audio_e = torch.mean(torch.abs(audio[i, :seq_len])).item()
                sample_audio_energy.append(audio_e)

                # Skeleton: Mean absolute value of normalized coords
                skel_e = torch.mean(torch.abs(skeleton[i, :seq_len])).item()
                sample_skel_energy.append(skel_e)

    # Final Metric
    final_metric = (
        total_levenshtein_dist / total_gt_gestures if total_gt_gestures > 0 else 0.0
    )
    print(f"Final Validation Metric: {final_metric}")

    # Correlation Analysis
    if len(sample_errors) > 1:
        df = pd.DataFrame(
            {
                "error": sample_errors,
                "length": sample_lengths,
                "audio_energy": sample_audio_energy,
                "skel_energy": sample_skel_energy,
            }
        )

        print("\n--- Failure Analysis: Correlation with Error Rate ---")
        features = ["length", "audio_energy", "skel_energy"]
        for feat in features:
            # Check for constant values to avoid warnings
            if df[feat].std() > 1e-9 and df["error"].std() > 1e-9:
                corr, _ = pearsonr(df["error"], df[feat])
                print(f"Correlation (Error vs {feat}): {corr:.4f}")
            else:
                print(f"Correlation (Error vs {feat}): Undefined (constant variance)")

    return final_metric


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Orchestration started on device: {device}")

    # 2. Train (Fast Baseline)
    # We limit epochs to 30 to ensure it runs quickly within the time limit.
    # load_cached_data=True ensures we use preprocessed .npz files if available.
    print("\n=== Phase 1: Training ===")
    run_training(num_epochs=30, load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("\n=== Phase 2: Validation & Analysis ===")

    # Load Data
    _, val_loader, _ = get_dataloaders()

    # Load Best Model
    model = DW_AIIN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Error: Best model checkpoint not found! Using untrained model.")

    # Run Analysis
    metric = analyze_failures_and_validate(model, val_loader, device)

    # 4. Submission
    print("\n=== Phase 3: Submission Generation ===")
    THRESHOLD = 0.05697278911564626

    if metric < THRESHOLD:
        print(
            f"Metric {metric:.6f} is below threshold {THRESHOLD:.6f}. Generating submission..."
        )
        run_prediction(load_cached_data=True)
    else:
        print(
            f"Metric {metric:.6f} did not meet threshold {THRESHOLD:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
