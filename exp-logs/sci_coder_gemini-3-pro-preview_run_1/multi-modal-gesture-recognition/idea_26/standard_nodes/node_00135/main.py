import os
import sys
import torch
import numpy as np
import scipy.stats
from library.config import Config
from library.utils import set_seed, _levenshtein_distance
from library.data_loader import get_data_loaders
from library.model import GCINet
from library.train import run_training, decode_predictions, decode_ground_truth
from library.inference import generate_predictions


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    # The dataset is small (~238 training samples), so we use the full set
    # but limit epochs to ensure completion within the time limit.
    Config.EPOCHS = 30

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # Execute the training pipeline. This handles data loading, model init,
    # training loop, validation, and checkpoint saving.
    print("Starting Training Pipeline...")
    run_training(debug=False)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    print("Starting Validation Assessment...")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = GCINet().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Checkpoint not found at {Config.BEST_MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Get Validation Loader
    # Note: get_data_loaders handles stats loading internally
    _, val_loader, _ = get_data_loaders(debug=False)

    all_dists = []
    all_lengths = []
    all_num_gestures = []

    total_distance = 0
    total_gestures_count = 0

    with torch.no_grad():
        for batch in val_loader:
            skel, audio, labels, lengths = batch
            if skel is None:
                continue

            skel = skel.to(device)
            audio = audio.to(device)

            # Forward Pass
            logits = model(skel, audio, lengths)

            # Decode
            batch_preds = decode_predictions(logits, lengths)
            batch_truths = decode_ground_truth(labels, lengths)

            # Calculate Metrics per Sample
            for pred, truth, length in zip(batch_preds, batch_truths, lengths):
                # Levenshtein Distance
                dist = _levenshtein_distance(list(pred), list(truth))

                # Store for analysis
                all_dists.append(dist)
                all_lengths.append(length.item())
                all_num_gestures.append(len(truth))

                # Accumulate for global metric
                total_distance += dist
                total_gestures_count += len(truth)

    # Compute Final Metric
    # Metric = Sum(Levenshtein) / Sum(NumGestures)
    final_metric = (
        total_distance / total_gestures_count if total_gestures_count > 0 else 0.0
    )

    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing Failure Analysis...")
    if len(all_dists) > 1:
        # Correlation: Error vs Sequence Length
        corr_len, _ = scipy.stats.pearsonr(all_dists, all_lengths)
        # Correlation: Error vs Number of Gestures
        corr_num, _ = scipy.stats.pearsonr(all_dists, all_num_gestures)

        print(f"Correlation (Error vs Seq Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.05697278911564626

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(debug=False)
    else:
        print(
            f"Metric ({final_metric}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
