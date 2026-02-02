import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.train import run_training, generate_submission
from library.model import AudioClassifier
from library.dataset import AudioDataset
from library.utils import calculate_lwlrap, set_seed


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast baseline execution
    # We use the full dataset but reduce epochs to fit within the time limit
    Config.epochs = 6
    Config.batch_size = 32

    # Ensure reproducibility
    set_seed(Config.seed)

    print(f"Configuration:")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch Size: {Config.batch_size}")
    print(f"  Device: {Config.device}")

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n--- Starting Training Pipeline ---")
    run_training()

    # =========================================================================
    # 3. Validation & Metrics
    # =========================================================================
    print("\n--- Starting Validation & Failure Analysis ---")

    device = Config.device

    # Load the best model saved during training
    if not os.path.exists(Config.model_save_path):
        print(f"Error: Model checkpoint not found at {Config.model_save_path}")
        return

    model = AudioClassifier()
    state_dict = torch.load(Config.model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Initialize Validation Loader
    val_dataset = AudioDataset(Config.val_csv_path, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []
    all_losses = []

    # Loss function for failure analysis (error magnitude)
    # We use reduction='none' to get loss per sample
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    print("Running inference on validation set...")
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            logits = model(data)
            probs = torch.sigmoid(logits)

            # Calculate error magnitude (mean BCE across classes for each sample)
            loss = criterion(logits, target).mean(dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_losses.append(loss.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_losses = np.concatenate(all_losses, axis=0)

    # Calculate and Print Final Metric
    metric = calculate_lwlrap(all_targets, all_preds)
    print(f"Final Validation Metric: {metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n--- Performing Failure Analysis ---")

    # Collect metadata features: Duration and Number of Labels
    durations = []
    num_labels = []

    # We iterate through the dataframe to get metadata
    # Note: Reading headers for ~5000 files is relatively fast
    print("Extracting metadata features...")
    for idx, row in val_dataset.df.iterrows():
        # Number of labels
        if pd.notna(row["labels"]):
            n_lbl = len(str(row["labels"]).split(","))
        else:
            n_lbl = 0
        num_labels.append(n_lbl)

        # Audio Duration
        full_path = os.path.join(Config.input_root, row["filepath"])
        try:
            # Only read header
            info = torchaudio.info(full_path)
            # Calculate duration in seconds
            dur = info.num_frames / info.sample_rate
            durations.append(dur)
        except Exception as e:
            # Fallback if file read fails
            durations.append(0.0)

    durations = np.array(durations)
    num_labels = np.array(num_labels)

    # Calculate Correlations
    # We use numpy for correlation

    # 1. Error vs Duration
    if len(durations) == len(all_losses):
        # np.corrcoef returns a matrix, we want [0, 1]
        corr_dur = np.corrcoef(all_losses, durations)[0, 1]
        print(f"Correlation (Error Magnitude vs Audio Duration): {corr_dur:.6f}")
    else:
        print("Mismatch in lengths for duration analysis.")

    # 2. Error vs Number of Labels
    if len(num_labels) == len(all_losses):
        corr_lbl = np.corrcoef(all_losses, num_labels)[0, 1]
        print(f"Correlation (Error Magnitude vs Number of Labels): {corr_lbl:.6f}")
    else:
        print("Mismatch in lengths for label count analysis.")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    print("\n--- Submission Check ---")
    threshold = 0.7043669848725875

    if metric > threshold:
        print(
            f"Metric {metric} exceeds threshold {threshold}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Metric {metric} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
