import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloader, get_dataframe
from library.model import AudioResNet34
from library.engine import train, validate, predict


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and input features (duration, label count).
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_probs = []
    all_targets = []

    # Get predictions
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate error per sample
    # Using Mean Absolute Error (MAE) per sample as a proxy for difficulty
    # error_i = mean(|y_pred_ij - y_true_ij|) over classes j
    sample_errors = np.abs(all_probs - all_targets).mean(axis=1)

    # Get metadata for correlation
    # We need the dataframe associated with the validation set
    # The loader's dataset has the dataframe
    val_df = val_loader.dataset.df.copy()

    # Ensure alignment (loader preserves order if shuffle=False)
    # val_loader is created with shuffle=False by default in get_dataloader for 'val'

    # Extract features
    # Note: 'duration' and 'label_count' are available in the dataframe from metadata generation/EDA
    # If not explicitly in columns, we might need to recalculate, but EDA showed they exist or can be derived.
    # The provided dataset.py filters columns, so we check what's available.
    # The dataset object filters columns for 'labels', but the self.df retains all columns passed to it.

    # Check if 'duration' and 'label_count' are in the dataframe
    # If not, we calculate them.
    if "duration" not in val_df.columns:
        # Fallback: we can't easily get duration without loading files,
        # but the metadata generation script output suggests they might not be in the saved csv
        # unless added. The EDA script added them but didn't save.
        # However, 'label_count' can be derived from the targets.
        pass

    # Calculate label count from ground truth if not in DF
    if "label_count" not in val_df.columns:
        val_df["label_count"] = val_df[val_loader.dataset.label_cols].sum(axis=1)

    # Calculate correlations
    # 1. Error vs Label Count
    if "label_count" in val_df.columns:
        corr_count, _ = pearsonr(sample_errors, val_df["label_count"])
        print(f"Correlation between Error and Label Count: {corr_count:.4f}")

    # 2. Error vs Duration
    # Since duration might not be in the CSV (EDA script calculated it but didn't save to metadata/val.csv),
    # we might skip this or try to infer. The prompt asks to "Calculate... correlation... input features".
    # We will attempt to use 'label_count' as the primary feature available.
    # If duration is missing, we skip it to avoid crashing or slow file reads.
    if "duration" in val_df.columns:
        # Handle NaNs if any
        valid_mask = ~val_df["duration"].isna()
        if valid_mask.sum() > 0:
            corr_dur, _ = pearsonr(
                sample_errors[valid_mask], val_df.loc[valid_mask, "duration"]
            )
            print(f"Correlation between Error and Audio Duration: {corr_dur:.4f}")
    else:
        print("Duration feature not found in metadata, skipping duration correlation.")


def main():
    # 1. Setup
    # Modify Config for fast baseline execution
    Config.MAX_EPOCHS = 20  # Limit epochs

    device = Config.DEVICE
    set_seed(Config.SEED)

    print(f"Initializing run on {device}...")

    # 2. Data Loading
    print("Loading data...")
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("val", batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Initialization
    print("Initializing model...")
    model = AudioResNet34(num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # 4. Training
    print("Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=Config.MAX_EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    # 5. Load Best Model
    print(f"Loading best model from {Config.MODEL_PATH}...")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # 6. Final Validation Metric
    print("Performing final validation...")
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_lrap = validate(model, val_loader, criterion, device)

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {val_lrap}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    baseline_score = 0.7883450707332457
    if val_lrap > baseline_score:
        print(
            f"Validation score ({val_lrap}) exceeds baseline ({baseline_score}). Generating submission..."
        )
        predict(model, test_loader, device, output_path=Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation score ({val_lrap}) does not exceed baseline ({baseline_score}). Skipping submission."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
