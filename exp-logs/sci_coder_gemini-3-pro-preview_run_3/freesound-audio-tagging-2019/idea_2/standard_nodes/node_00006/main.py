import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import AudioDataset
from library.model import AudioClassifier
from library.engine import train_model, validate, predict


def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error (BCE loss) and input features.
    """
    model.eval()
    all_losses = []
    all_num_labels = []
    all_sig_magnitudes = []

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    print("\n--- Failure Analysis ---")
    with torch.no_grad():
        for data, target, _ in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            # Calculate loss per sample (average over classes) to represent error magnitude
            # Shape: (Batch, Num_Classes) -> (Batch,)
            loss = criterion(output, target).mean(dim=1)

            # Feature 1: Number of active labels (Complexity)
            num_labels = target.sum(dim=1)

            # Feature 2: Signal Magnitude (Mean of spectrogram)
            # data shape: (Batch, 1, Freq, Time)
            sig_mag = data.mean(dim=(1, 2, 3))

            all_losses.append(loss.cpu().numpy())
            all_num_labels.append(num_labels.cpu().numpy())
            all_sig_magnitudes.append(sig_mag.cpu().numpy())

    all_losses = np.concatenate(all_losses)
    all_num_labels = np.concatenate(all_num_labels)
    all_sig_magnitudes = np.concatenate(all_sig_magnitudes)

    # Calculate Correlations
    if len(all_losses) > 1:
        corr_labels = np.corrcoef(all_losses, all_num_labels)[0, 1]
        corr_mag = np.corrcoef(all_losses, all_sig_magnitudes)[0, 1]

        print(f"Correlation (Error vs Label Count): {corr_labels:.6f}")
        print(f"Correlation (Error vs Signal Magnitude): {corr_mag:.6f}")

        if abs(corr_labels) > 0.1:
            print("Observation: Error is correlated with the number of labels.")
        if abs(corr_mag) > 0.1:
            print("Observation: Error is correlated with signal magnitude.")
    else:
        print("Insufficient data for correlation analysis.")
    print("------------------------\n")


def run():
    # 1. Configuration for Fast Baseline
    # We override Config defaults to ensure quick execution on A100
    Config.epochs = 5
    Config.batch_size = 128  # EfficientNet-B0 is small, A100 can handle large batches
    Config.num_workers = 4

    # Set seeds for reproducibility
    set_seed(Config.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    # We use the full dataset but fewer epochs as the dataset size (20k) is manageable
    train_dataset = AudioDataset(Config.train_metadata, mode="train")
    val_dataset = AudioDataset(Config.val_metadata, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = AudioClassifier(
        num_classes=Config.num_classes, pretrained=Config.pretrained
    )
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 5. Training
    print(f"Starting training for {Config.epochs} epochs...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.epochs,
    )

    # 6. Final Validation & Metrics
    print("Calculating Final Validation Metric...")
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_score = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    # Threshold from task description
    THRESHOLD = 0.43188462409377254

    if val_score > THRESHOLD:
        print(
            f"Validation Score ({val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = AudioDataset(Config.test_metadata, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        predict(model, test_loader, device)
    else:
        print(
            f"Validation Score ({val_score}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
