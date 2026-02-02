import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.utils import (
    set_seeds,
    decode_predictions_to_sequence,
    run_length_encoding,
    compute_levenshtein_score,
    levenshtein_distance,
)
from library.data_loader import get_dataloaders
from library.model import DGC_KN
from library.train import train_model
from library.predict import generate_submission


def main():
    # 1. Setup and Reproducibility
    set_seeds()

    # Configure for a fast baseline run
    # Increased to 35 to allow CosineAnnealing scheduler to complete its cycle
    NUM_EPOCHS = 35

    print("=== Starting Optimized DGC-KN Run ===")

    # 2. Training
    # train_model handles the entire training loop, saving the best model to Config.BEST_MODEL_PATH
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Model checkpoint found at {Config.BEST_MODEL_PATH}. Skipping training.")
    else:
        print(f"Training for {NUM_EPOCHS} epochs...")
        train_model(epochs=NUM_EPOCHS)

    # 3. Validation & Failure Analysis
    print("\n=== Performing Validation and Failure Analysis ===")

    device = Config.DEVICE
    model = DGC_KN().to(device)

    # Load best model weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Best model checkpoint not found after training.")

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Get validation loader (batch_size=1 for full sequences)
    _, val_loader, _ = get_dataloaders(debug=False)

    predictions = {}
    ground_truths = {}

    # Lists for failure analysis
    fa_errors = []
    fa_lengths = []
    fa_num_gestures = []

    total_distance = 0
    total_gestures_count = 0

    with torch.no_grad():
        for i, (features, dense_labels, sample_ids) in enumerate(val_loader):
            features = features.to(device)
            sample_id = sample_ids[0]

            # Forward pass
            outputs = model(features)

            # Get probabilities from Stage 3
            probs = outputs["probs_3"].squeeze(0).cpu().numpy()

            # Decode predictions
            pred_seq = decode_predictions_to_sequence(probs)
            predictions[sample_id] = pred_seq

            # Decode Ground Truth
            # dense_labels is (1, Time)
            gt_dense = dense_labels[0].numpy()
            gt_segments = run_length_encoding(gt_dense)
            gt_seq = [
                int(cls)
                for cls, _, _ in gt_segments
                if cls != Config.BACKGROUND_CLASS_ID
            ]
            ground_truths[sample_id] = gt_seq

            # --- Metrics Calculation ---
            dist = levenshtein_distance(pred_seq, gt_seq)
            n_gestures = len(gt_seq)

            total_distance += dist
            total_gestures_count += n_gestures

            # --- Failure Analysis Data Collection ---
            # Sequence duration in frames
            seq_len = features.shape[1]

            fa_errors.append(dist)
            fa_lengths.append(seq_len)
            fa_num_gestures.append(n_gestures)

    # Compute Final Metric
    # Metric = Sum(Levenshtein) / Total GT Gestures
    if total_gestures_count > 0:
        final_metric = total_distance / total_gestures_count
    else:
        final_metric = 0.0

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis Report ---
    print("\n--- Failure Analysis ---")
    if len(fa_errors) > 1:
        df_fa = pd.DataFrame(
            {"error": fa_errors, "length": fa_lengths, "num_gestures": fa_num_gestures}
        )

        # Correlation: Error Magnitude vs Input Features
        # We check if error correlates with sequence length or complexity (num gestures)
        corr_len, _ = pearsonr(df_fa["error"], df_fa["length"])
        corr_num, _ = pearsonr(df_fa["error"], df_fa["num_gestures"])

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        if abs(corr_len) > 0.3:
            print(
                ">> Observation: Significant correlation between sequence length and error."
            )
        if abs(corr_num) > 0.3:
            print(
                ">> Observation: Significant correlation between gesture count and error."
            )
    else:
        print("Not enough validation samples for correlation analysis.")

    # 4. Submission Generation
    # Threshold check
    THRESHOLD = 0.16539050535987748

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is strictly lower than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
