import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, levenshtein_distance, rle_decode
from library.data_loader import get_dataloaders
from library.model import GCA_IIN
from library.train_eval import train_model, predict_test, decode_target_seq


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Hyperparameters for Fast Baseline
    # We limit epochs to ensure the script completes quickly (< 2 hours)
    FAST_EPOCHS = 30
    BATCH_SIZE = Config.BATCH_SIZE

    print(f"Starting Fast Baseline Training (Epochs={FAST_EPOCHS})...")

    # 2. Train Model
    # train_model handles the training loop, checkpointing, and returns path to best model
    best_model_path = train_model(num_epochs=FAST_EPOCHS, batch_size=BATCH_SIZE)
    print(f"Best model saved at: {best_model_path}")

    # 3. Validation and Metric Calculation
    print("Loading validation data for evaluation...")
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # Load the best model for evaluation
    model = GCA_IIN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print("Performing inference on validation set...")

    all_preds = []
    all_targets = []
    sample_ids = []
    seq_lengths = []
    target_counts = []
    sample_errors = []

    # Disable gradient calculation for inference efficiency
    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            targets = batch["target"].to(device)
            lengths = batch["lengths"].to(device)
            ids = batch["sample_ids"]

            # Forward pass
            logits = model(skeleton, audio, lengths)

            # Get frame-wise predictions
            preds_frame = torch.argmax(logits, dim=2).cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            for i in range(len(ids)):
                # Handle variable lengths (ignore padding)
                length = lengths[i].item()
                p_seq = preds_frame[i, :length]
                t_seq = targets_cpu[i, :length]

                # Decode sequences
                pred_gestures = rle_decode(p_seq)
                target_gestures = decode_target_seq(t_seq)

                # Compute Levenshtein distance for this sample
                dist = levenshtein_distance(pred_gestures, target_gestures)

                # Store data for metric and analysis
                all_preds.append(pred_gestures)
                all_targets.append(target_gestures)
                sample_ids.append(ids[i])
                seq_lengths.append(length)
                target_counts.append(len(target_gestures))
                sample_errors.append(dist)

    # Compute Global Metric (Levenshtein Error Rate)
    total_distance = sum(sample_errors)
    total_reference_length = sum(target_counts)

    # Avoid division by zero
    final_metric = (
        total_distance / total_reference_length if total_reference_length > 0 else 0.0
    )

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "seq_length": seq_lengths,
            "num_gestures": target_counts,
            "error": sample_errors,
        }
    )

    # Calculate correlations
    # 1. Correlation between Sequence Length (frames) and Error Magnitude
    if len(df_analysis) > 1 and df_analysis["seq_length"].std() > 0:
        corr_len, _ = pearsonr(df_analysis["seq_length"], df_analysis["error"])
    else:
        corr_len = 0.0

    # 2. Correlation between Number of Gestures (complexity) and Error Magnitude
    if len(df_analysis) > 1 and df_analysis["num_gestures"].std() > 0:
        corr_gest, _ = pearsonr(df_analysis["num_gestures"], df_analysis["error"])
    else:
        corr_gest = 0.0

    print(f"Correlation (Sequence Length vs Error): {corr_len:.4f}")
    print(f"Correlation (Num Gestures vs Error): {corr_gest:.4f}")

    # 5. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) is strictly lower than threshold ({THRESHOLD})."
        )
        print("Generating submission file for test set...")
        predict_test(model_path=best_model_path, batch_size=BATCH_SIZE)
    else:
        print(
            f"\nValidation Metric ({final_metric}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation as per requirements.")


if __name__ == "__main__":
    main()
