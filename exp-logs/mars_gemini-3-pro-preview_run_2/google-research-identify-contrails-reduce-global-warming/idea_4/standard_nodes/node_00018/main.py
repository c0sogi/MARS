import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.model import ResnetUNet
from library.loss import GlobalBatchDiceLoss
from library.train import train_model, validate
from library.predict import predict


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify systematic errors.
    Calculates the correlation between error magnitude (1 - Dice) and input features.
    """
    print("Starting failure analysis...")

    # Access the dataframe from the dataset
    # Note: val_loader.dataset is the ContrailDataset
    meta_df = val_loader.dataset.df.copy()

    # Store per-sample dice scores
    per_sample_dice = []

    model.eval()
    with torch.no_grad():
        for i, (images, masks, record_ids) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            pred_masks = outputs["mask"]

            # Binarize predictions
            preds = (pred_masks > Config.THRESHOLD).float()

            # Compute Dice per sample for analysis
            # Intersection and Cardinality per image in the batch
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            cardinality = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            # Add epsilon to avoid division by zero for empty mask + empty pred
            # If both are empty, dice is 1.0. If one is empty, dice is 0.0.
            # Using a small epsilon handles the 0/0 case correctly if logic matches metric
            # Here we use standard formula
            dice_scores = (2.0 * intersection) / (cardinality + 1e-6)

            # Handle the case where cardinality is 0 (both empty) -> Dice should be 1
            # The formula gives 0 if intersection is 0.
            # We fix this: if cardinality is effectively 0, score is 1.
            is_empty = cardinality < 1e-5
            dice_scores[is_empty] = 1.0

            per_sample_dice.extend(dice_scores.cpu().numpy())

    # Add error metric to dataframe
    # Ensure length matches
    if len(per_sample_dice) != len(meta_df):
        print(
            f"Warning: Mismatch in failure analysis samples. Preds: {len(per_sample_dice)}, DF: {len(meta_df)}"
        )
        # Truncate to match the smaller one (though they should match)
        min_len = min(len(per_sample_dice), len(meta_df))
        per_sample_dice = per_sample_dice[:min_len]
        meta_df = meta_df.iloc[:min_len]

    meta_df["dice"] = per_sample_dice
    meta_df["error"] = 1.0 - meta_df["dice"]

    # Feature Engineering for Correlation
    if "timestamp" in meta_df.columns:
        meta_df["datetime"] = pd.to_datetime(meta_df["timestamp"], unit="s")
        meta_df["hour"] = meta_df["datetime"].dt.hour
        meta_df["month"] = meta_df["datetime"].dt.month

    # Features to check
    features = ["timestamp", "row_min", "col_min", "hour", "month"]
    correlations = {}

    print("Correlation between Error (1-Dice) and Features:")
    for feat in features:
        if feat in meta_df.columns:
            # Drop NaNs just in case
            valid_df = meta_df[[feat, "error"]].dropna()
            if not valid_df.empty:
                corr = valid_df["error"].corr(valid_df[feat])
                correlations[feat] = corr
                print(f"  {feat}: {corr}")
            else:
                print(f"  {feat}: Insufficient data")
        else:
            print(f"  {feat}: Not found in metadata")


def main():
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Training
    # Cite solution_lesson_node_00002: Training on full dataset (removed max_samples)
    print("=== Starting Training Phase ===")
    train_model(epochs=Config.EPOCHS)

    # 3. Validation
    print("\n=== Starting Validation Phase ===")
    # Load the best model saved during training
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Best model not found. Training may have failed.")

    model = ResnetUNet(in_channels=Config.IN_CHANNELS, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load FULL validation set
    val_dataset = ContrailDataset(split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    criterion = GlobalBatchDiceLoss()

    # Compute metrics
    _, global_dice = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {global_dice}")

    # 4. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 5. Submission
    # Threshold check
    THRESHOLD_SCORE = 0.5454606988733747

    if global_dice > THRESHOLD_SCORE:
        print(
            f"\nValidation metric ({global_dice}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        # predict() function handles loading the best model and generating the CSV
        predict()
    else:
        print(
            f"\nValidation metric ({global_dice}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
