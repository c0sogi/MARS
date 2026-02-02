import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.utils import (
    seed_everything,
    smooth_predictions,
    decode_sequence,
    compute_normalized_levenshtein,
    compute_levenshtein,
)
from library.data_loader import GestureDataset, collate_fn
from library.trainer import Trainer


def main():
    # 1. Setup and Reproducibility
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Configuration
    # Reduced epochs to 15 for a fast baseline execution within time limits
    config = {
        "input_dim": 85,  # 72 (Skeleton) + 13 (Audio)
        "num_classes": 21,  # 20 Gestures + 1 Background
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 15,  # Limited for speed
        "patience": 5,
        "noise_std": 0.05,  # Regularization
    }

    # 3. Data Loading
    print("Loading datasets...")
    # We use load_cached_data=True to speed up loading if cache exists
    train_dataset = GestureDataset(
        metadata_file="./metadata/train.csv", load_cached_data=True
    )
    val_dataset = GestureDataset(
        metadata_file="./metadata/val.csv", load_cached_data=True
    )
    test_dataset = GestureDataset(
        metadata_file="./metadata/test.csv", load_cached_data=True, is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # 4. Training
    print("Initializing Trainer...")
    trainer = Trainer(config, device=device)

    print("Starting training...")
    trainer.fit(
        train_loader, val_loader, epochs=config["epochs"], patience=config["patience"]
    )

    # 5. Evaluation & Failure Analysis
    print("Running comprehensive validation analysis...")

    # Load the best model saved during training
    if os.path.exists(trainer.checkpoint_path):
        trainer.model.load_state_dict(
            torch.load(trainer.checkpoint_path, map_location=device)
        )
        print("Loaded best model checkpoint.")

    trainer.model.eval()

    all_preds = []
    all_targets = []
    sample_errors = []
    sample_lengths = []  # Feature: Sequence length in frames

    with torch.no_grad():
        for batch_idx, (features, labels, lengths, mask) in enumerate(val_loader):
            features = features.to(device)
            mask = mask.to(device)
            padding_mask = ~mask

            # Inference
            outputs = trainer.model(features, src_key_padding_mask=padding_mask)

            # Greedy decoding
            probs = torch.softmax(outputs, dim=2)
            preds = torch.argmax(probs, dim=2)  # (B, T)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            for i in range(len(lengths_np)):
                length = lengths_np[i]

                # Extract valid sequence (ignore padding)
                p_seq = preds_np[i, :length]
                t_seq = labels_np[i, :length]

                # Post-processing: Smooth and Decode
                p_smooth = smooth_predictions(p_seq, window_size=5)
                p_decoded = decode_sequence(p_smooth, background_class_id=0)
                t_decoded = decode_sequence(t_seq, background_class_id=0)

                # Compute Error
                dist = compute_levenshtein(p_decoded, t_decoded)

                all_preds.append(p_decoded)
                all_targets.append(t_decoded)
                sample_errors.append(dist)
                sample_lengths.append(length)

    # Compute Final Metric
    final_metric = compute_normalized_levenshtein(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Sequence Length
    if len(sample_errors) > 1:
        # Using numpy for correlation
        correlation_matrix = np.corrcoef(sample_errors, sample_lengths)
        correlation = correlation_matrix[0, 1]
        print(
            f"Correlation between Error (Levenshtein) and Sequence Length (Frames): {correlation:.4f}"
        )
    else:
        print("Insufficient data for correlation analysis.")

    # 6. Submission Generation
    # Threshold check as per requirements
    threshold = 0.424
    if final_metric < threshold:
        print(
            f"Validation metric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        trainer.predict(test_loader, output_file=submission_path)
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
