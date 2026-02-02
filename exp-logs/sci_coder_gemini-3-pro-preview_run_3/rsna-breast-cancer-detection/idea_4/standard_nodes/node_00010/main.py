import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, apply_analytical_correction
from library.dataset import get_dataloaders
from library.model import BreastMILModel
from library.train_eval import train_one_epoch, evaluate, predict_and_submit


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline execution
    Config.NUM_EPOCHS = 5

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # Use cached data to speed up initialization
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = BreastMILModel(pretrained=True)
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    best_pf1 = -1.0

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val pF1: {val_pf1:.6f}"
        )

        # Checkpoint
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # -------------------------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    # Required Output Format
    print(f"Final Validation Metric: {best_pf1}")

    # Load best model for analysis and inference
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()

    print("\n=== Failure Analysis ===")
    # Generate predictions on validation set to correlate errors
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for images, labels, ids in val_loader:
            images = [img.to(device) for img in images]
            logits = model(images)
            # Apply calibration to get probabilities
            probs = apply_analytical_correction(logits)

            val_preds.extend(probs.cpu().numpy().flatten())
            val_targets.extend(labels.numpy().flatten())
            val_ids.extend(ids)

    # Create results DataFrame
    df_res = pd.DataFrame(
        {"prediction_id": val_ids, "prob": val_preds, "target": val_targets}
    )
    df_res["error"] = (df_res["target"] - df_res["prob"]).abs()

    # Load metadata to get features
    df_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Reconstruct prediction_id in metadata
    if "prediction_id" not in df_meta.columns:
        df_meta["prediction_id"] = (
            df_meta["patient_id"].astype(str) + "_" + df_meta["laterality"]
        )

    # Features to analyze
    features = ["age", "implant", "density", "machine_id", "site_id"]
    features = [f for f in features if f in df_meta.columns]

    # Aggregate metadata to bag level (take first value)
    df_meta_bag = df_meta.groupby("prediction_id")[features].first().reset_index()

    # Merge results with metadata
    df_analysis = pd.merge(df_res, df_meta_bag, on="prediction_id", how="left")

    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        if feat in df_analysis.columns:
            # Drop NaNs
            tmp = df_analysis[[feat, "error"]].dropna()
            if len(tmp) > 0:
                # Encode categorical if necessary
                if tmp[feat].dtype == "object":
                    try:
                        tmp[feat] = tmp[feat].astype("category").cat.codes
                    except:
                        continue

                corr = tmp[feat].corr(tmp["error"])
                print(f"Feature '{feat}': {corr:.5f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.06310755014419556

    if best_pf1 > THRESHOLD:
        print(
            f"\nValidation metric ({best_pf1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({best_pf1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
