import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Subset
from itertools import groupby

# Ensure library modules can be imported
sys.path.append(os.path.join(os.getcwd(), "library"))

from library import config
from library import utils
from library import data_loader
from library import model
from library import trainer
from library import inference


def run_pipeline():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()

    print(f"Running Full Training with {config.EPOCHS} epochs...")

    # 2. Data Loading
    # Load cached data if available
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        load_cached_data=True
    )

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # 3. Model Initialization
    net = model.DSR_CRCN()
    net = net.to(device)

    # 4. Training
    model_trainer = trainer.Trainer(net, train_loader, val_loader)
    model_trainer.fit()

    # 5. Validation & Metric Calculation
    print("Performing full validation evaluation...")
    net.eval()

    all_predictions = []
    all_targets = []

    # For Failure Analysis
    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            targets_padded = batch[
                "targets"
            ]  # Keep on CPU for decoding if possible, or move back later
            lengths = batch["lengths"]
            ids = batch["ids"]

            # Forward pass
            _, _, logits2 = net(features, mask)

            # Process batch
            for i in range(len(ids)):
                seq_len = lengths[i].item()

                # Extract valid logits
                sample_logits = logits2[i, :seq_len, :]
                sample_probs = F.softmax(sample_logits, dim=1).cpu().numpy()

                # Smooth and Decode Prediction
                smoothed_probs = utils.smooth_predictions(
                    sample_probs, window_size=config.MEDIAN_WINDOW
                )
                pred_seq = utils.decode_sequence(smoothed_probs)

                # Decode Target
                # targets_padded is [B, T]
                target_seq = utils.decode_target_sequence(
                    targets_padded[i, :seq_len].numpy()
                )

                all_predictions.append(pred_seq)
                all_targets.append(target_seq)

                # Compute individual error for failure analysis
                # Levenshtein distance
                import nltk

                dist = nltk.edit_distance(pred_seq, target_seq)

                analysis_data.append(
                    {
                        "id": ids[i],
                        "seq_len": seq_len,
                        "num_gestures": len(target_seq),
                        "error": dist,
                    }
                )

    # Compute Final Metric
    # Metric = Sum(Levenshtein) / Total Gestures in Ground Truth
    total_levenshtein = sum(item["error"] for item in analysis_data)
    total_gestures = sum(item["num_gestures"] for item in analysis_data)

    final_metric = total_levenshtein / total_gestures if total_gestures > 0 else 0.0

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(analysis_data)

    if not df_analysis.empty:
        # Correlation between Error and Features
        correlation_len = df_analysis["error"].corr(df_analysis["seq_len"])
        correlation_gestures = df_analysis["error"].corr(df_analysis["num_gestures"])

        print(f"Correlation (Error vs Sequence Length): {correlation_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {correlation_gestures:.4f}")

        # Check for high error samples
        high_error = df_analysis.sort_values(by="error", ascending=False).head(3)
        print("\nTop 3 High Error Samples:")
        print(high_error)
    else:
        print("No validation data available for analysis.")

    # 7. Submission Generation
    # Threshold from prompt: 0.1282225237449118
    THRESHOLD = 0.1282225237449118

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.5f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        inference.generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric:.5f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
