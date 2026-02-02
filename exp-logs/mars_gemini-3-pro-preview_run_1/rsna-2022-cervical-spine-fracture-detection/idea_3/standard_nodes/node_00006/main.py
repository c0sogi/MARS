import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import glob
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.train_stage1 import train_localizer
from library.train_stage2 import train_encoder
from library.train_stage3 import train_aggregator
from library.inference import load_models, predict_study, generate_predictions
from library.utils import calculate_weighted_loss


def run_validation_and_analysis():
    """
    Runs inference on the validation set, calculates the metric,
    and performs failure analysis.
    """
    print("\n" + "=" * 40)
    print("Running Validation & Failure Analysis")
    print("=" * 40)

    device = Config.DEVICE
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Load trained models
    # We use the load_models function from inference.py
    try:
        models = load_models(device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return float("inf")

    target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    y_true_all = []
    y_pred_all = []
    sample_losses = []
    meta_features = {"num_slices": []}

    print(f"Validating on {len(val_df)} studies...")

    # Inference Loop
    for idx, row in val_df.iterrows():
        uid = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Get Ground Truth
        y_true = row[target_cols].values.astype(float)  # (8,)

        # Get Prediction
        # predict_study returns a list/array of 8 probabilities
        try:
            # Check slice count for failure analysis
            dcm_files = glob.glob(os.path.join(full_path, "*.dcm"))
            num_slices = len(dcm_files)

            y_pred = predict_study(uid, full_path, models, device)
            y_pred = np.array(y_pred)
        except Exception as e:
            print(f"Validation error on {uid}: {e}")
            y_pred = np.array([0.5] * 8)
            num_slices = 0

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        meta_features["num_slices"].append(num_slices)

        # Calculate individual sample loss for failure analysis
        # We reshape to (1, 8) for the utility function if needed,
        # but here we can just compute it directly for the sample.
        # Using the formula: -w * [y*log(p) + (1-y)*log(1-p)]
        # Weights: patient=7, others=1
        weights = np.array(
            [Config.WEIGHT_PATIENT_OVERALL] + [Config.WEIGHT_VERTEBRAE] * 7
        )

        # Clip
        eps = 1e-15
        p_clip = np.clip(y_pred, eps, 1 - eps)

        loss_terms = -(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
        weighted_loss = np.sum(loss_terms * weights) / np.sum(
            weights
        )  # Average over weighted sum?
        # The metric description says: "Finally, loss is averaged across all rows."
        # The formula L_ij is for a specific label.
        # Usually competition metrics average the weighted loss over all N*8 entries,
        # or average the per-row weighted loss.
        # library.utils.calculate_weighted_loss averages mean(loss).
        # Let's align with library.utils.

        # Calculate using library util for consistency on single sample
        sample_loss = calculate_weighted_loss(y_true[None, :], y_pred[None, :])
        sample_losses.append(sample_loss)

    # 1. Calculate Final Metric
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    final_metric = calculate_weighted_loss(y_true_all, y_pred_all)

    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Correlation between Error (Loss) and Num Slices
    losses = np.array(sample_losses)
    n_slices = np.array(meta_features["num_slices"])

    if len(losses) > 1:
        corr, p_val = pearsonr(losses, n_slices)
        print(f"Correlation (Loss vs Num Slices): {corr:.4f} (p={p_val:.4f})")

        if corr > 0.3:
            print("Observation: Higher error rates detected in scans with more slices.")
        elif corr < -0.3:
            print(
                "Observation: Higher error rates detected in scans with fewer slices."
            )
        else:
            print(
                "Observation: No strong correlation between scan depth and error rate."
            )

    return final_metric


def main():
    # 1. Setup
    Config.setup()

    # Override Config for Fast Baseline Execution
    # We reduce epochs to ensure completion within time limits
    Config.LOCALIZER_EPOCHS = 5
    Config.ENCODER_EPOCHS = 2
    Config.SEQ_EPOCHS = 5

    # Ensure we use the full provided training set (161 samples)
    # but keep epochs low for speed.
    Config.DEBUG = False

    print("Starting End-to-End Pipeline")
    print(f"Device: {Config.DEVICE}")

    # 2. Training Stages
    print("\n" + "=" * 40)
    print("Stage 1: Training Spine Localizer")
    print("=" * 40)
    train_localizer(debug=Config.DEBUG)

    print("\n" + "=" * 40)
    print("Stage 2: Training Slice Encoder")
    print("=" * 40)
    train_encoder(debug=Config.DEBUG)

    print("\n" + "=" * 40)
    print("Stage 3: Training Sequence Aggregator")
    print("=" * 40)
    train_aggregator(debug=Config.DEBUG)

    # 3. Validation & Analysis
    val_metric = run_validation_and_analysis()

    # 4. Submission
    # Threshold from requirements
    THRESHOLD = 0.9440845186799401

    if val_metric < THRESHOLD:
        print("\n" + "=" * 40)
        print("Generating Submission")
        print("=" * 40)
        generate_predictions(debug=False)
    else:
        print(
            f"\nValidation metric ({val_metric}) is not lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
