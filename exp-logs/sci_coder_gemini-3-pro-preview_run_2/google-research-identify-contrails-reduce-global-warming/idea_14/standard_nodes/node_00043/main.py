import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, dice_coef_metric
from library.dataset import ContrailDataset
from library.model import ConvNeXtHyperUNet
from library.loss import HybridLoss
from library.engine import train_model, valid_one_epoch, make_submission


def perform_failure_analysis(model, loader, device):
    """
    Analyzes model errors on the validation set and correlates them with metadata.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    dice_scores = []

    # 1. Compute per-sample Dice scores
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(images)
            preds = torch.sigmoid(logits)

            # Iterate through batch
            for i in range(images.size(0)):
                # dice_coef_metric handles single sample tensors (C, H, W)
                d = dice_coef_metric(preds[i], masks[i], threshold=Config.THRESHOLD)
                dice_scores.append(d)

    dice_scores = np.array(dice_scores)
    errors = 1.0 - dice_scores

    # 2. Extract Metadata
    # The loader preserves order, so we can map directly to the dataframe
    df = loader.dataset.df.copy()

    # Ensure lengths match
    if len(df) != len(errors):
        print(
            f"Warning: Metadata length ({len(df)}) != Predictions length ({len(errors)})"
        )
        return

    # Feature Engineering for Analysis
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"] = df["timestamp_dt"].dt.hour
    df["month"] = df["timestamp_dt"].dt.month

    # Define features to correlate with Error
    features = {
        "Hour": df["hour"],
        "Month": df["month"],
        "Row_Min": df["row_min"],
        "Col_Min": df["col_min"],
    }

    print("Correlation between Error Magnitude (1 - Dice) and Input Features:")
    for name, series in features.items():
        # Handle potential NaNs just in case
        valid_idx = ~series.isna()
        if valid_idx.sum() > 1:
            corr, _ = pearsonr(errors[valid_idx], series[valid_idx])
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (Insufficient data)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Override Config for Fast Baseline
    # We limit epochs and training samples to ensure execution within 2 hours
    EPOCHS = 5
    TRAIN_SAMPLE_SIZE = 5000
    # Cite debug_lesson_7: Explicitly override batch size to handle potential stale module cache
    Config.BATCH_SIZE = 8

    # 2. Data Loading
    print("Initializing Datasets...")

    # Training Data
    train_dataset = ContrailDataset(split="train", load_cached_data=True)
    # Subsample training data for speed
    if len(train_dataset.df) > TRAIN_SAMPLE_SIZE:
        print(
            f"Subsampling training data from {len(train_dataset.df)} to {TRAIN_SAMPLE_SIZE}..."
        )
        train_dataset.df = train_dataset.df.sample(
            n=TRAIN_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Data (Full)
    valid_dataset = ContrailDataset(split="validation", load_cached_data=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Data (Full)
    test_dataset = ContrailDataset(split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing Model: {Config.MODEL_NAME}...")
    model = ConvNeXtHyperUNet()
    model.to(device)

    # 4. Training Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = HybridLoss()

    # 5. Training Loop
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS,
        patience=3,  # Strict patience for baseline
        save_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    )

    # 6. Evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(
        torch.load(
            os.path.join(Config.WORKING_DIR, "best_model.pth"), map_location=device
        )
    )
    model.to(device)
    model.eval()

    # Calculate Final Metric on Validation Set
    _, global_dice = valid_one_epoch(model, valid_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {global_dice}")

    # 7. Failure Analysis
    perform_failure_analysis(model, valid_loader, device)

    # 8. Submission
    THRESHOLD_SCORE = 0.5910660985501295

    if global_dice > THRESHOLD_SCORE:
        print(
            f"\nValidation score ({global_dice}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        make_submission(
            model=model,
            loader=test_loader,
            output_path=Config.SUBMISSION_PATH,
            device=device,
        )
    else:
        print(
            f"\nValidation score ({global_dice}) did not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
