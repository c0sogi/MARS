import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import random

from library.config import Config
from library.trainer import Trainer
from library.data_loader import get_dataloaders
from library.inference import generate_submission


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metric(model, loader, device):
    """
    Computes the KL Divergence metric on the given loader.
    """
    model.eval()
    # reduction='batchmean' aligns with the competition metric definition for averaged KL
    criterion = nn.KLDivLoss(reduction="batchmean")
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            probs = model(inputs)
            # KLDivLoss requires log-probabilities as input
            log_probs = torch.log(probs + 1e-7)

            # loss.item() is the mean over the batch
            # We multiply by batch size to accumulate the weighted sum
            batch_size = inputs.size(0)
            loss = criterion(log_probs, targets)

            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def failure_analysis(model, loader, device):
    """
    Performs failure analysis on the validation set by correlating error with metadata features.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            probs = model(inputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate row-wise KL Divergence manually for analysis
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log P) - sum(P * log Q)
    epsilon = 1e-7
    preds = np.clip(preds, epsilon, 1.0)

    # Term 1: P * log P (handling 0 * log 0 = 0)
    term1 = np.where(targets > epsilon, targets * np.log(targets), 0.0)

    # Term 2: P * log Q
    term2 = targets * np.log(preds)

    # Sum over classes (axis 1) to get scalar error per sample
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Get metadata from the dataset
    df = loader.dataset.df.copy()

    # Ensure alignment
    if len(df) != len(kl_per_sample):
        print("Warning: Metadata length mismatch in failure analysis.")
        return

    df["error_magnitude"] = kl_per_sample

    print("\n=== Failure Analysis ===")
    features = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]

    for feat in features:
        if feat in df.columns:
            vals = df[feat].fillna(0).values
            corr, _ = pearsonr(vals, kl_per_sample)
            print(f"Correlation between Error and {feat}: {corr:.10f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Configure for Fast Baseline
    # We reduce epochs to 3 for speed while maintaining full dataset coverage (Debug=False)
    # This ensures the submission file contains predictions for ALL test IDs.
    Config.EPOCHS = 3
    Config.BATCH_SIZE = 64
    Config.DEBUG = False

    # 3. Training and Submission Generation
    # Trainer.fit() handles training, validation monitoring, model saving,
    # and automatically generates the submission file on the test set.
    print("Starting training pipeline...")
    trainer = Trainer(config=Config)
    trainer.fit(debug=Config.DEBUG)

    # 4. Load Best Model for Validation Assessment
    device = torch.device(Config.DEVICE)
    model = trainer.model
    # Ensure we have the best weights loaded (Trainer.fit does this, but being explicit is safer)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 5. Validation Metric
    # Load validation data
    _, val_loader, _ = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
    )

    metric = calculate_metric(model, val_loader, device)
    print(f"Final Validation Metric: {metric:.16f}")

    # 6. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 7. Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission successfully generated at {Config.SUBMISSION_PATH}")
    else:
        print("Warning: Submission file missing. Generating now...")
        generate_submission(debug=Config.DEBUG)


if __name__ == "__main__":
    main()
