import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats
import nltk
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_loaders
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Use cached data to speed up loading process
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Training
    print("Initializing Trainer...")
    trainer = Trainer(device, train_loader, val_loader, test_loader)

    # Train for a limited number of epochs to ensure completion within time limits.
    # The SG-CRCN architecture is efficient, so 35 epochs should be sufficient
    # for a strong baseline while keeping runtime low.
    print("Starting training...")
    trainer.fit(num_epochs=45)

    # 4. Final Validation & Metric Calculation
    print("Performing final validation...")

    # Load the best model checkpoint
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    trainer.model.eval()

    val_preds = []
    val_targets = []
    val_lengths = []
    val_sample_ids = []
    val_errors = []  # Per sample error rate

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]
            target_cls = batch["target_cls"]  # Keep on CPU for decoding
            sample_ids = batch["sample_ids"]

            # Forward pass
            outputs = trainer.model(features, mask)

            # Use Stage 3 outputs for final prediction
            stage3_logits = outputs["stage3_cls"]

            # Decode predictions
            batch_preds = trainer.decode_predictions(stage3_logits, lengths)

            # Decode targets (convert tensor to list of lists)
            target_cls_np = target_cls.numpy()
            batch_targets = []
            for i in range(len(target_cls_np)):
                l = lengths[i]
                seq = target_cls_np[i, :l]
                decoded_target = []
                prev = -1
                for lbl in seq:
                    if lbl != prev:
                        if lbl != 0:  # 0 is background
                            decoded_target.append(int(lbl))
                        prev = lbl
                batch_targets.append(decoded_target)

            # Compute per-sample metrics for failure analysis
            for p, t in zip(batch_preds, batch_targets):
                dist = nltk.edit_distance(p, t)
                t_len = len(t)
                # Error rate per sample: dist / max(1, t_len)
                # This is a proxy for "how bad is this sample"
                err = dist / t_len if t_len > 0 else float(dist)
                val_errors.append(err)

            val_preds.extend(batch_preds)
            val_targets.extend(batch_targets)
            val_lengths.extend(lengths.tolist())
            val_sample_ids.extend(sample_ids)

    # Compute Global Metric (Levenshtein Error Rate)
    # Metric = Sum(Distances) / Sum(Target Lengths)
    total_distance = 0
    total_target_length = 0

    for p, t in zip(val_preds, val_targets):
        total_distance += nltk.edit_distance(p, t)
        total_target_length += len(t)

    final_metric = (
        total_distance / total_target_length if total_target_length > 0 else 0.0
    )

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "error_rate": val_errors,
            "frame_length": val_lengths,
            "num_gestures": [len(t) for t in val_targets],
        }
    )

    # Calculate correlations
    # We check if error rate correlates with sequence length or complexity (num gestures)
    if len(df_analysis) > 1:
        corr_frames, _ = scipy.stats.pearsonr(
            df_analysis["error_rate"], df_analysis["frame_length"]
        )
        corr_gestures, _ = scipy.stats.pearsonr(
            df_analysis["error_rate"], df_analysis["num_gestures"]
        )

        print(f"Correlation (Error Rate vs Frame Length): {corr_frames:.4f}")
        print(f"Correlation (Error Rate vs Num Gestures): {corr_gestures:.4f}")

        # Identify worst samples
        df_analysis["sample_id"] = val_sample_ids
        worst_samples = df_analysis.sort_values("error_rate", ascending=False).head(5)
        print("\nTop 5 Worst Samples:")
        print(
            worst_samples[["sample_id", "error_rate", "num_gestures", "frame_length"]]
        )
    else:
        print("Not enough samples for correlation analysis.")

    # 6. Submission
    # Threshold from task description logic
    THRESHOLD = 0.08548168249660787

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        # trainer.predict() handles loading best model and saving submission
        trainer.predict()
    else:
        print(
            f"\nMetric {final_metric} is not below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
