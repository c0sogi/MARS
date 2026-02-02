import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score

# Import from provided library
from library.config import Config
from library.engine import Trainer
from library.utils import (
    seed_everything,
    optimize_threshold,
    load_checkpoint,
    save_checkpoint,
)
from library.dataset import get_dataloader
from library.preprocessing import Preprocessor

# Modify Config for Fast Baseline Execution
# Limiting to 1 epoch to ensure completion within 2 hours
Config.EPOCHS = 1
# Increasing batch size to utilize A100 GPU memory and speed up training
Config.BATCH_SIZE = 1024
Config.NUM_WORKERS = 4


def main():
    # Set reproducibility
    seed_everything(Config.SEED)

    print("Initializing Trainer...")
    trainer = Trainer()

    # Load Data
    # Explicitly pass batch_size because the default in get_dataloader was bound at import time
    print("Creating DataLoaders...")
    train_loader = get_dataloader("train", shuffle=True, batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader("val", shuffle=False, batch_size=Config.BATCH_SIZE)

    # Training Loop
    print(f"Starting training for {Config.EPOCHS} epoch(s)...")
    best_val_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, val_probs, val_targets = trainer.evaluate(val_loader)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                trainer.model,
                trainer.optimizer,
                epoch,
                val_loss,
                Config.MODEL_SAVE_PATH,
            )
            print(f"Checkpoint saved at epoch {epoch}")

    # Validation & Threshold Optimization
    print("Loading best model for final validation...")
    load_checkpoint(Config.MODEL_SAVE_PATH, trainer.model, device=trainer.device)

    # Get predictions from best model
    val_loss, val_probs, val_targets = trainer.evaluate(val_loader)

    # Optimize Threshold
    print("Optimizing threshold...")
    best_threshold, best_f1 = optimize_threshold(val_targets, val_probs)
    print(f"Final Validation Metric: {best_f1}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")

    # 1. Calculate Error Magnitude (1 - F1 per sample)
    val_preds_bin = (val_probs >= best_threshold).astype(int)

    # Calculate sample-wise F1 manually
    # (sklearn f1_score with average='samples' returns a scalar mean, we need the array)
    def calculate_sample_f1_array(y_true, y_pred):
        tp = (y_true * y_pred).sum(axis=1)
        fp = ((1 - y_true) * y_pred).sum(axis=1)
        fn = (y_true * (1 - y_pred)).sum(axis=1)
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1

    sample_f1 = calculate_sample_f1_array(val_targets, val_preds_bin)
    error_magnitude = 1.0 - sample_f1

    # 2. Load Metadata and Align
    try:
        val_df = pd.read_csv(Config.VAL_PATH)

        # Ensure lengths match (in case of dropped batches or other anomalies, though unlikely)
        min_len = min(len(val_df), len(error_magnitude))
        if len(val_df) != len(error_magnitude):
            print(
                f"Warning: Metadata length ({len(val_df)}) differs from prediction length ({len(error_magnitude)}). Truncating to minimum."
            )
            val_df = val_df.iloc[:min_len]
            error_magnitude = error_magnitude[:min_len]

        # 3. Compute Features
        # Fill NaNs just in case
        titles = val_df["Title"].fillna("").astype(str)
        bodies = val_df["Body"].fillna("").astype(str)
        tags = val_df["Tags"].fillna("").astype(str)

        val_df["title_len"] = titles.apply(len)
        val_df["body_len"] = bodies.apply(len)
        val_df["num_tags"] = tags.apply(lambda x: len(x.split()))

        # 4. Compute Correlations
        corr_title = np.corrcoef(val_df["title_len"], error_magnitude)[0, 1]
        corr_body = np.corrcoef(val_df["body_len"], error_magnitude)[0, 1]
        corr_tags = np.corrcoef(val_df["num_tags"], error_magnitude)[0, 1]

        print(f"Correlation (Error vs Title Length): {corr_title:.6f}")
        print(f"Correlation (Error vs Body Length): {corr_body:.6f}")
        print(f"Correlation (Error vs Num Tags): {corr_tags:.6f}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # Conditional Submission
    submission_threshold = 0.0542101508997596

    if best_f1 > submission_threshold:
        print(f"\nMetric {best_f1} > {submission_threshold}. Generating submission...")

        test_loader = get_dataloader(
            "test", shuffle=False, batch_size=Config.BATCH_SIZE
        )
        test_probs = trainer.predict(test_loader)

        # Apply threshold
        test_preds_bin = (test_probs >= best_threshold).astype(int)

        # Convert to tags
        print("Converting predictions to tags...")
        preprocessor = Preprocessor()
        preprocessor.tag_encoder.load(Config.TAG_ENCODER_PATH)

        pred_tags = preprocessor.inverse_transform_tags(test_preds_bin)
        test_ids = preprocessor.get_test_ids()

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub = pd.DataFrame({"Id": test_ids, "Tags": pred_tags})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission Saved.")

    else:
        print(f"\nMetric {best_f1} <= {submission_threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
