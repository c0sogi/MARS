import os
import torch
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from nltk import edit_distance

# Import configuration and library modules
import library.config as config
import library.trainer
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import HybridGestureNet
from library.trainer import Trainer
from library.inference import generate_submission, decode_sequence

# -----------------------------------------------------------------------------
# 1. Configuration Override for Fast Baseline
# -----------------------------------------------------------------------------
# Reduce epochs to ensure the script finishes within the 2-hour limit
# while still providing a meaningful baseline.
library.trainer.NUM_EPOCHS = 15
config.NUM_EPOCHS = 15


def main():
    # Set reproducibility
    set_seed(config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = HybridGestureNet().to(device)

    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("Loading best model for evaluation...")
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    all_preds = []
    all_targets = []
    sample_errors = []
    sample_lengths = []

    print("Evaluating on Validation Set...")
    with torch.no_grad():
        for batch_idx, (x, y, lengths, ids) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward Pass
            logits = model(x, lengths)

            # Get probabilities and raw frame predictions
            probs = torch.softmax(logits, dim=2)
            preds_raw = torch.argmax(probs, dim=2).cpu().numpy()
            y_cpu = y.cpu().numpy()

            # Process each sample in the batch
            for i in range(len(ids)):
                length = lengths[i].item()

                # Extract valid sequence (remove padding)
                # Apply Median Filter to match Inference pipeline
                raw_seq = preds_raw[i, :length]
                filtered_seq = median_filter(
                    raw_seq, size=config.MEDIAN_FILTER_KERNEL, mode="nearest"
                )

                # Decode predictions
                decoded_pred = decode_sequence(filtered_seq)

                # Decode targets (Ground Truth)
                target_seq = y_cpu[i, :length]
                decoded_target = decode_sequence(target_seq)

                # Compute Levenshtein Distance for this sample
                dist = edit_distance(decoded_pred, decoded_target)

                all_preds.append(decoded_pred)
                all_targets.append(decoded_target)
                sample_errors.append(dist)
                sample_lengths.append(length)

    # Calculate Final Metric
    # Metric = Total Distance / Total Number of Ground Truth Gestures
    total_distance = sum(sample_errors)
    total_gestures = sum(len(t) for t in all_targets)

    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    if len(sample_errors) > 1:
        # Correlation between Error Magnitude and Sequence Length
        corr, p_value = pearsonr(sample_errors, sample_lengths)
        print(
            f"Correlation between Error (Levenshtein Dist) and Sequence Length: {corr:.4f}"
        )
        print(f"P-value: {p_value:.4f}")

        if abs(corr) > 0.3:
            print("Observation: Moderate to strong correlation detected.")
            if corr > 0:
                print("-> Longer sequences tend to have higher error rates.")
            else:
                print("-> Shorter sequences tend to have higher error rates.")
        else:
            print(
                "Observation: Weak or no correlation detected between error and length."
            )
    else:
        print("Insufficient data for correlation analysis.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.424
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(device=str(device))
    else:
        print(
            f"\nMetric ({final_metric:.4f}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
