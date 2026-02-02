import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import OffsetGuidedDualStreamModel
from library.train import train_one_epoch, validate
from library.inference import inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override defaults for a fast but effective baseline
    Config.EPOCHS = 3

    # Initialize Config (creates directories)
    Config.setup(debug=False)

    # Set reproducible seeds
    seed_everything(Config.SEED)

    # Logger and Device
    logger = get_logger("runfile")
    device = torch.device(Config.DEVICE)

    logger.info(f"Initialized. Device: {device}, Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=True
    )
    logger.info("DataLoaders ready.")

    # -------------------------------------------------------------------------
    # 3. Model & Optimization
    # -------------------------------------------------------------------------
    model = OffsetGuidedDualStreamModel(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    logger.info("Starting training loop...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler, device, logger
        )

        # Validate
        val_loss = validate(model, val_loader, device, logger)

        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model.")

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    # REQUIRED: Print the final metric in full precision
    print(f"Final Validation Metric: {best_val_loss}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Starting Failure Analysis...")

    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_errors = []

    # Compute per-sample KL Divergence
    with torch.no_grad():
        for spec, eeg, guidance, targets in val_loader:
            spec = spec.to(device)
            eeg = eeg.to(device)
            guidance = guidance.to(device)
            targets = targets.to(device)

            logits = model(spec, eeg, guidance)
            log_probs = F.log_softmax(logits, dim=1)

            # KL Div reduction='none' gives element-wise. Sum over classes (dim=1) for sample error.
            # F.kl_div(input, target) -> target * (log(target) - input)
            batch_errors = F.kl_div(log_probs, targets, reduction="none").sum(dim=1)
            val_errors.extend(batch_errors.cpu().numpy())

    val_errors = np.array(val_errors)

    # Load Validation Metadata to correlate errors
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (handle potential edge cases, though loaders preserve order)
    if len(val_errors) != len(val_df):
        min_len = min(len(val_errors), len(val_df))
        val_errors = val_errors[:min_len]
        val_df = val_df.iloc[:min_len]

    val_df["kl_error"] = val_errors

    print("Failure Analysis - Correlation with Error:")
    features_to_check = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]

    for feat in features_to_check:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["kl_error"])
            print(f"Correlation between Error and {feat}: {corr}")

    # -------------------------------------------------------------------------
    # 7. Conditional Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.7327804565429688

    if best_val_loss < THRESHOLD:
        logger.info(
            f"Validation metric {best_val_loss} meets threshold ({THRESHOLD}). Generating submission..."
        )
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        inference(model, test_loader, device, submission_path)
    else:
        logger.info(
            f"Validation metric {best_val_loss} did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
