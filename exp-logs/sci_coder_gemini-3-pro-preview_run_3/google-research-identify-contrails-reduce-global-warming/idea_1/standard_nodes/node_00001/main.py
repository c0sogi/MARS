import os
import torch
import pandas as pd
import numpy as np
import warnings
from library.config import Config
from library.data import get_dataloaders
from library.model import LightUNet
from library.train import run_training
from library.inference import generate_submission
from library.utils import set_seed


def main():
    # 1. Setup
    set_seed(Config.SEED)
    warnings.filterwarnings("ignore")

    # Ensure working directory exists (handled by Config.setup usually, but good to ensure)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Training
    # We limit epochs to 5 to ensure a fast baseline execution within the 2-hour limit.
    # The A100 GPU is fast enough to process the dataset, but we want to be conservative.
    print("Starting training pipeline...")
    best_model_path = run_training(
        epochs=5, batch_size=Config.BATCH_SIZE, debug=False  # Use full dataset
    )

    # 3. Validation & Metric Calculation
    print("Starting validation assessment...")

    # Load validation data
    # get_dataloaders returns (train, val, test). We need val.
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=False
    )

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = LightUNet().to(device)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(
            "Error: Best model not found. Using random weights for validation (invalid results expected)."
        )

    model.eval()

    # Variables for Global Dice calculation
    total_intersection = 0.0
    total_union = 0.0

    # Store per-sample errors for failure analysis
    error_records = []

    with torch.no_grad():
        for images, masks, record_ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(images)
            preds = (outputs > Config.THRESHOLD).float()

            # Update Global Dice accumulators
            # Flatten to vectors for set operations
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            total_intersection += (preds_flat * masks_flat).sum().item()
            total_union += preds_flat.sum().item() + masks_flat.sum().item()

            # --- Per-Sample Analysis for Failure Analysis ---
            # Calculate Dice per image in the batch
            # Shape: (Batch, 1, H, W) -> flatten to (Batch, H*W)
            b_preds = preds.view(preds.size(0), -1)
            b_masks = masks.view(masks.size(0), -1)

            b_inter = (b_preds * b_masks).sum(dim=1)
            b_union = b_preds.sum(dim=1) + b_masks.sum(dim=1)

            # Dice = 2*inter / union. If union is 0 (both empty), Dice is 1.0.
            b_dice = torch.zeros_like(b_inter)
            non_empty = b_union > 0
            b_dice[non_empty] = (2.0 * b_inter[non_empty]) / b_union[non_empty]
            b_dice[~non_empty] = 1.0

            # Error = 1 - Dice
            b_errors = 1.0 - b_dice.cpu().numpy()

            for rid, err in zip(record_ids, b_errors):
                error_records.append({"record_id": str(rid), "error": err})

    # Calculate Final Global Dice
    # Add epsilon to prevent division by zero if dataset is empty or all black
    final_metric = (2.0 * total_intersection) / (total_union + 1e-6)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")

    # Load validation metadata to get features
    val_metadata = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    val_metadata["record_id"] = val_metadata["record_id"].astype(str)

    # Create DataFrame from errors
    error_df = pd.DataFrame(error_records)

    # Merge errors with metadata
    analysis_df = pd.merge(error_df, val_metadata, on="record_id", how="left")

    # Calculate correlations
    features_to_analyze = ["timestamp", "row_min", "col_min"]

    print("Correlation between Error Magnitude (1-Dice) and Input Features:")
    for feature in features_to_analyze:
        if feature in analysis_df.columns:
            # Drop NaNs just in case
            valid_data = analysis_df[[feature, "error"]].dropna()
            if not valid_data.empty:
                corr = valid_data[feature].corr(valid_data["error"])
                print(f"Correlation with {feature}: {corr}")
            else:
                print(f"Correlation with {feature}: Insufficient data")
        else:
            print(f"Correlation with {feature}: Feature not found")

    # 5. Submission
    print("Generating submission file...")
    generate_submission(
        model_path=best_model_path, batch_size=Config.BATCH_SIZE, debug=False
    )
    print("Process complete.")


if __name__ == "__main__":
    main()
