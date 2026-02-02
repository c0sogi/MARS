import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import MCSINModel, train_one_epoch, evaluate, predict_and_submit

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")

    # Modify Config for Fast Baseline
    Config.EPOCHS = 2  # Reduce epochs for speed (Baseline requirement)
    # Config.DEBUG = True # Uncomment if strictly testing pipeline, but we want a valid score

    device = torch.device(Config.DEVICE)
    logger.info(f"Device: {device}")

    # 2. Data Loading
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    logger.info(f"Initializing Model: {Config.MODEL_NAME}")
    model = MCSINModel(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 4. Optimizer, Scheduler, Loss
    # Note: pos_weight must be on device
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scaler = GradScaler(enabled=Config.USE_AMP)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # 5. Training Loop
    best_pf1 = 0.0
    logger.info("Starting Training...")

    # Remove existing best model to ensure we don't load stale state
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate (Standard evaluation for checkpointing)
        val_pf1 = evaluate(model, val_loader, device, return_preds=False)
        logger.info(f"Epoch {epoch+1} | Val pF1: {val_pf1}")

        # Save Best
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved! (pF1: {val_pf1})")

    # 6. Final Evaluation & Failure Analysis
    logger.info("Loading best model for final evaluation and analysis...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning("No best model found. Using current state.")

    # Get predictions on validation set
    final_pf1, val_preds_df = evaluate(model, val_loader, device, return_preds=True)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_pf1}")

    # --- Failure Analysis ---
    logger.info("Performing Failure Analysis...")
    try:
        # Load validation metadata to get features
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

        # Reconstruct prediction_id in metadata to merge
        # Logic matches BreastCancerDataset: f"{patient_id}_{laterality}"
        val_meta["prediction_id"] = (
            val_meta["patient_id"].astype(str) + "_" + val_meta["laterality"]
        )

        # Merge predictions with metadata
        # val_preds_df has columns: [prediction_id, prob, target]
        # We aggregate metadata by prediction_id (taking first/max since patient-level feats are constant)
        meta_agg = val_meta.groupby("prediction_id").first().reset_index()

        analysis_df = pd.merge(val_preds_df, meta_agg, on="prediction_id", how="inner")

        # Calculate Error
        analysis_df["error"] = (analysis_df["target"] - analysis_df["prob"]).abs()

        # Preprocess features for correlation
        # Density: A->1, B->2, C->3, D->4
        density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
        analysis_df["density_encoded"] = analysis_df["density"].map(density_map)

        # Calculate Correlations
        correlations = {}

        # Age
        if "age" in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df["age"])
            correlations["Age"] = corr

        # Density
        if "density_encoded" in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df["density_encoded"])
            correlations["Density"] = corr

        # Machine ID
        if "machine_id" in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df["machine_id"])
            correlations["Machine_ID"] = corr

        print("\n=== Failure Analysis: Error Correlations ===")
        for feature, corr in correlations.items():
            print(f"{feature}: {corr:.4f}")
        print("============================================\n")

    except Exception as e:
        logger.error(f"Failure analysis failed: {e}")

    # 7. Submission
    THRESHOLD = 0.044888656586408615
    if final_pf1 > THRESHOLD:
        logger.info(
            f"Validation score ({final_pf1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        logger.warning(
            f"Validation score ({final_pf1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
