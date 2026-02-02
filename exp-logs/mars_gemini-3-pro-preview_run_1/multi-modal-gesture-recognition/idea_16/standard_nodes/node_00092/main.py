import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    SUBMISSION_DIR,
    LABEL_MAP,
    WORK_DIR,
)
from library.utils import set_seed, decode_predictions, compute_levenshtein
from library.data_loader import GestureDataset, collate_fn
from library.model import PCA_IIN
from library.train import train_model


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Correlates error magnitude with input features.
    """
    print("\n=== Starting Failure Analysis ===")
    model.eval()

    errors = []
    seq_lengths = []
    audio_energies = []

    with torch.no_grad():
        for skels, audios, labels, lengths in val_loader:
            skels = skels.to(device)
            audios = audios.to(device)

            # Forward pass
            logits = model(skels, audios, lengths)

            batch_size = logits.size(0)
            for i in range(batch_size):
                length = int(lengths[i].item())

                # Decode Prediction
                seq_logits = logits[i, :length, :].cpu().numpy()
                pred_seq = decode_predictions(seq_logits)

                # Get Target
                raw_target = labels[i, :length].cpu().tolist()
                target_seq = [x for x in raw_target if x != LABEL_MAP["background"]]

                # Compute Distance
                dist = compute_levenshtein(pred_seq, target_seq)

                # Normalize error by target length (handle div by zero)
                norm_error = dist / len(target_seq) if len(target_seq) > 0 else dist

                errors.append(norm_error)
                seq_lengths.append(length)

                # Simple audio feature: Mean energy of the sequence
                # Audio shape: (Batch, Time, MelBins)
                audio_energy = audios[i, :length, :].mean().item()
                audio_energies.append(audio_energy)

    # Compute Correlations
    if len(errors) > 1:
        corr_len, _ = pearsonr(errors, seq_lengths)
        corr_audio, _ = pearsonr(errors, audio_energies)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Audio Energy): {corr_audio:.4f}")
    else:
        print("Insufficient data for correlation analysis.")
    print("=== Failure Analysis Complete ===\n")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission for test set...")

    # Load Test Data
    test_dataset = GestureDataset(split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    model.eval()
    results = []

    # Map ID back to sample name for CSV
    # The dataset metadata has 'sample_id' which corresponds to the file name (e.g. Session00001)
    # We need to access this. The loader shuffles? No, shuffle=False.
    # We can iterate the metadata directly or trust the order.
    # Safer to rely on the loader order matching the dataset order.

    sample_ids = test_dataset.metadata["sample_id"].tolist()
    current_idx = 0

    with torch.no_grad():
        for skels, audios, _, lengths in test_loader:
            skels = skels.to(device)
            audios = audios.to(device)

            logits = model(skels, audios, lengths)

            batch_size = logits.size(0)
            for i in range(batch_size):
                length = int(lengths[i].item())

                seq_logits = logits[i, :length, :].cpu().numpy()
                pred_seq = decode_predictions(seq_logits)

                # Format: SessionID,label1,label2...
                # Get ID
                sid = sample_ids[current_idx]
                current_idx += 1

                # Join labels
                labels_str = ",".join(map(str, pred_seq))

                # Construct line
                # Format example: Session00001,2,12,3
                line = f"{sid},{labels_str}"
                results.append(line)

    # Save
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {submission_path}")


def main():
    # 1. Setup
    set_seed()

    # 2. Train
    # Using full dataset (debug_subset_size=None) as it is small.
    # Limited epochs for fast baseline.
    print("Starting training pipeline...")
    best_metric = train_model(debug_subset_size=None, epochs=30)

    # 3. Report Metric
    print(f"Final Validation Metric: {best_metric}")

    # 4. Load Best Model
    model = PCA_IIN().to(DEVICE)
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Error: Best model checkpoint not found.")
        return

    # 5. Failure Analysis
    val_dataset = GestureDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    run_failure_analysis(model, val_loader, DEVICE)

    # 6. Submission
    THRESHOLD = 0.0765306122
    if best_metric < THRESHOLD:
        generate_submission(model, DEVICE)
    else:
        print(
            f"Validation metric ({best_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
