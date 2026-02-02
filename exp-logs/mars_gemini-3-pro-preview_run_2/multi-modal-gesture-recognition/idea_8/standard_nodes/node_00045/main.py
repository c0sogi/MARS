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


def decode_frame_targets(targets_tensor):
    """
    Decodes frame-wise target tensor into a list of gesture IDs.
    Args:
        targets_tensor: [Time] tensor of integers.
    Returns:
        list of int: Sequence of gesture IDs.
    """
    # Convert to numpy
    labels = targets_tensor.cpu().numpy()
    # Collapse repeats and remove background (0)
    collapsed = [k for k, g in groupby(labels)]
    sequence = [k for k in collapsed if k != 0]
    return sequence


def run_pipeline():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()

    # Modify config for Fast Baseline
    config.EPOCHS = 5
    config.BATCH_SIZE = 8  # Increase batch size slightly for speed if memory allows

    print(f"Running Fast Baseline with {config.EPOCHS} epochs...")

    # 2. Data Loading
    # Load cached data if available
    train_loader_full, val_loader, test_loader = data_loader.get_data_loaders(
        load_cached_data=True
    )

    # Subset training data for speed (first 500 samples)
    train_dataset_full = train_loader_full.dataset
    subset_indices = list(range(min(len(train_dataset_full), 500)))
    train_subset = Subset(train_dataset_full, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=data_loader.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Training set size (subset): {len(train_subset)}")
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
                smoothed_probs = inference.smooth_predictions(
                    sample_probs, window_size=config.MEDIAN_WINDOW
                )
                pred_seq = inference.decode_sequence(smoothed_probs)

                # Decode Target
                # targets_padded is [B, T]
                sample_targets_tensor = targets_padded[i, :seq_len]
                target_seq = decode_frame_targets(sample_targets_tensor)

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
