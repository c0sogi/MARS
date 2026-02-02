import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    set_seed,
    post_process_sequence,
    decode_predictions_rle,
    get_levenshtein_distance,
)
from library.data_loader import get_data_loaders
from library.train import Trainer
from library.model import CGRNet


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override Config for Fast Baseline
    # Reducing epochs to ensure execution within time limits while allowing convergence
    Config.EPOCHS = 30

    # ==========================================
    # 2. Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer(device=device)

    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer.fit(epochs=Config.EPOCHS)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("Loading best model for analysis...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    model = CGRNet().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get loaders
    _, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    print("Running inference on validation set...")
    val_preds = []
    val_targets = []
    val_errors = []
    val_lengths = []  # Input feature for correlation analysis

    with torch.no_grad():
        for batch_data in val_loader:
            skels, audios, labels, lengths = batch_data
            if skels is None:
                continue

            skels = skels.to(device)
            audios = audios.to(device)

            # Forward
            logits = model(skels, audios)
            probs = torch.softmax(logits, dim=2)

            # Process batch
            for i in range(len(skels)):
                length = lengths[i].item()

                # Get valid sequence
                sample_probs = probs[i, :length]
                sample_labels = labels[i, :length]

                # Decode
                pred_seq = post_process_sequence(sample_probs)
                target_seq = decode_predictions_rle(sample_labels.cpu().numpy())

                # Metric
                dist = get_levenshtein_distance(pred_seq, target_seq)

                val_preds.append(pred_seq)
                val_targets.append(target_seq)
                val_errors.append(dist)
                val_lengths.append(length)

    # Compute Final Validation Metric
    total_distance = sum(val_errors)
    total_gestures = sum(len(t) for t in val_targets)

    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Sequence Length
    if len(val_errors) > 1:
        correlation = np.corrcoef(val_lengths, val_errors)[0, 1]
        print(
            f"Correlation between Sequence Length (Frames) and Error (Levenshtein Dist): {correlation:.4f}"
        )
    else:
        print("Insufficient data for correlation analysis.")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = 0.0824829931972789

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test IDs to map predictions
        test_df = pd.read_csv(Config.TEST_CSV)
        if Config.DEBUG:
            test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)
        test_ids = test_df["sample_id"].tolist()

        submission_lines = []
        current_idx = 0

        with torch.no_grad():
            for batch_data in test_loader:
                skels, audios, labels, lengths = batch_data
                if skels is None:
                    continue

                skels = skels.to(device)
                audios = audios.to(device)

                logits = model(skels, audios)
                probs = torch.softmax(logits, dim=2)

                for i in range(len(skels)):
                    if current_idx >= len(test_ids):
                        break

                    length = lengths[i].item()
                    sample_probs = probs[i, :length]

                    # Decode
                    pred_seq = post_process_sequence(sample_probs)

                    # Format: SessionID,Label1,Label2...
                    sid = test_ids[current_idx]
                    pred_str = ",".join(map(str, pred_seq))
                    line = f"{sid},{pred_str}"
                    submission_lines.append(line)

                    current_idx += 1

        # Save submission
        with open(Config.SUBMISSION_PATH, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Metric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
