import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from provided library files
from library.config import Config
from library.utils import get_logger, set_seed, get_device
from library.data import get_dataloaders
from library.model import MTSIN
from library.train import MultiTaskLoss, train_one_epoch, validate, predict_and_submit


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    logger = get_logger(name="runfile")

    logger.info("Initializing Fast Baseline Run...")

    # Override Config for Fast Baseline
    # Limiting epochs to 2 to ensure runtime < 2 hours while allowing convergence
    Config.NUM_EPOCHS = 2

    # 2. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    model = MTSIN()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    criterion = MultiTaskLoss(device)

    # 4. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_cancer_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val pF1: {val_pf1:.10f}"
        )

        # Save Best
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with pF1: {best_pf1:.10f}")

    # 5. Final Validation & Failure Analysis
    logger.info("Loading best model for analysis...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    # --- Collect Validation Predictions for Analysis ---
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["target_cancer"].to(device)

            outputs = model(images, meta)
            probs = torch.sigmoid(outputs["cancer"]).cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets_np)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Error
    errors = np.abs(val_preds - val_targets)

    # Access Validation Metadata
    # val_loader.dataset.df corresponds to the data in order (shuffle=False)
    val_df = val_loader.dataset.df.copy()
    val_df["error"] = errors
    val_df["prediction"] = val_preds

    # --- Failure Analysis: Correlation ---
    logger.info("Performing Failure Analysis...")

    # Select numerical/ordinal columns for correlation
    analysis_cols = ["age", "density_enc", "implant", "laterality_enc", "view_enc"]
    # Filter columns that actually exist in the dataframe
    analysis_cols = [c for c in analysis_cols if c in val_df.columns]

    correlations = val_df[analysis_cols].corrwith(val_df["error"])

    print("\n=== Failure Analysis: Error Correlations ===")
    print(correlations)
    print("============================================\n")

    # Print Final Metric as required
    print(f"Final Validation Metric: {best_pf1}")

    # 6. Submission Logic
    THRESHOLD = 0.044888656586408615

    if best_pf1 > THRESHOLD:
        logger.info(
            f"Validation metric ({best_pf1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device)
    else:
        logger.warning(
            f"Validation metric ({best_pf1}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
