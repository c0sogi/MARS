import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.train import train_model
from library.inference import generate_submission
from library.models import HybridEEGModel
from library.data_loader import get_dataloaders


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    logger = get_logger("runfile")

    # Override Config for a fast baseline execution
    # Reducing epochs to 4 ensures we complete within the 2-hour limit
    # while allowing enough convergence for the hybrid model.
    Config.EPOCHS = 4

    logger.info("Starting pipeline...")

    # 2. Train the Model
    # We use debug=False to train on the full dataset to maximize performance
    # and meet the metric threshold.
    best_val_loss = train_model(debug=False, load_cached=True, epochs=Config.EPOCHS)

    # 3. Validation Assessment & Failure Analysis
    logger.info("Starting Validation and Failure Analysis...")

    # Load full validation data
    # We discard the train_loader here, we only need val_loader
    _, val_loader = get_dataloaders(debug=False, load_cached=True)

    # Load the best saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridEEGModel(num_classes=Config.N_CLASSES, pretrained_spec=False)

    if not os.path.exists(Config.MODEL_PATH):
        logger.error("Model path does not exist. Training failed.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for raw_x, spec_x, targets in val_loader:
            raw_x = raw_x.to(device)
            spec_x = spec_x.to(device)

            # Forward pass
            outputs = model(raw_x, spec_x)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Calculate Final Metric (KL Divergence)
    # KL(P || Q) = sum(P * log(P / Q))
    # Clip predictions for numerical stability
    epsilon = 1e-15
    preds_clipped = np.clip(preds, epsilon, 1.0 - epsilon)

    # Calculate row-wise KL divergence
    # Note: targets are probabilities P, preds are Q
    # KL = sum(target * (log(target) - log(pred)))
    # We handle the case where target is 0 (0 * log(0) = 0)

    # Safe log(target)
    log_target = np.log(targets + epsilon)
    log_pred = np.log(preds_clipped)

    kl_per_class = targets * (log_target - log_pred)
    kl_per_row = np.sum(kl_per_class, axis=1)

    final_metric = np.mean(kl_per_row)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Load validation metadata to correlate errors with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (DataLoader should preserve order, but check length)
    if len(val_df) != len(kl_per_row):
        logger.warning(
            f"Shape mismatch: Metadata {len(val_df)} vs Preds {len(kl_per_row)}. Truncating to minimum."
        )
        min_len = min(len(val_df), len(kl_per_row))
        val_df = val_df.iloc[:min_len]
        kl_per_row = kl_per_row[:min_len]

    val_df["error_kl"] = kl_per_row

    # Calculate correlations with numerical features
    # We exclude the probability targets themselves from correlation analysis
    exclude_cols = Config.TARGET_COLS + [
        "eeg_id",
        "spectrogram_id",
        "patient_id",
        "label_id",
    ]
    feature_cols = [
        c
        for c in val_df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols and c != "error_kl"
    ]

    correlations = (
        val_df[feature_cols].corrwith(val_df["error_kl"]).sort_values(ascending=False)
    )

    print(
        "\nFailure Analysis - Correlation between Input Features and Error Magnitude:"
    )
    print(correlations)

    # 5. Submission Generation
    # Threshold check
    THRESHOLD = 1.0081

    if final_metric < THRESHOLD:
        logger.info(
            f"Metric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            debug=False,
            load_cached=True,
            model_path=Config.MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        logger.info(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
