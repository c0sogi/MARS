import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from provided library
from library.config import Config
from library.utils import set_seed, levenshtein_distance, rle_decode, median_filter
from library.data_loader import get_dataloaders
from library.engine import train_model, validate, get_loss_function
from library.model import MPWINet
from library.inference import generate_predictions


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between Levenshtein error and sequence length.
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()

    errors = []
    seq_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"]
            mask = batch["mask"].to(device)
            labels_seq_gt = batch["labels_seq"]

            # Forward pass
            logits = model(skeleton, audio, lengths, mask)

            # Get hard predictions
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            lengths_np = lengths.numpy()

            for i in range(preds.shape[0]):
                valid_len = lengths_np[i]
                raw_pred_seq = preds[i, :valid_len]

                # Post-processing
                smoothed_seq = median_filter(
                    raw_pred_seq, window_size=Config.MEDIAN_FILTER_WINDOW
                )
                decoded_gestures = rle_decode(
                    smoothed_seq,
                    background_label=Config.BACKGROUND_LABEL,
                    min_len=Config.MIN_SEGMENT_LENGTH,
                )

                # Ground Truth
                gt_seq = labels_seq_gt[i]
                if isinstance(gt_seq, torch.Tensor):
                    gt_seq = gt_seq.tolist()

                # Calculate Error
                dist = levenshtein_distance(decoded_gestures, gt_seq)

                errors.append(dist)
                seq_lengths.append(valid_len)

    errors = np.array(errors)
    seq_lengths = np.array(seq_lengths)

    # Calculate Correlation
    if len(errors) > 1:
        # Using numpy for correlation to avoid potential missing scipy dependency
        corr_matrix = np.corrcoef(errors, seq_lengths)
        corr = corr_matrix[0, 1]
        print(
            f"Correlation between Error (Levenshtein) and Sequence Length (Frames): {corr:.16f}"
        )
    else:
        print("Not enough samples for correlation analysis.")


def main():
    # 1. Setup & Configuration
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Fast Baseline Configuration
    # We reduce the number of epochs to ensure the run completes quickly while
    # leveraging the full dataset (which is small) for maximum accuracy.
    Config.NUM_EPOCHS = 30

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 2. Training
    print("Starting Training...")
    # train_model returns the path to the best checkpoint saved during training
    best_model_path = train_model(train_loader, val_loader)

    # 3. Validation Assessment
    print("Loading Best Model for Final Validation...")
    model = MPWINet().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get Loss Function for validation (required by engine.validate)
    criterion = get_loss_function(device)

    # Compute Final Metric on the entire hold-out validation set
    _, final_metric = validate(model, val_loader, criterion, device)

    # Print Metric in required format (full precision)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 5. Submission Generation
    # Submission is generated only if the metric is below the specified threshold.
    THRESHOLD = 0.05697278911564626

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(
            model_path=best_model_path, output_path=Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
