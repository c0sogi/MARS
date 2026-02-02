import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    setup_logger,
    get_device,
    save_checkpoint,
    probabilistic_f1,
    AverageMeter,
)
from library.data import get_dataloaders
from library.model import DSGEHNModel

# We import train_one_epoch and validate to reuse logic,
# but we will orchestrate them here to control the flow for the baseline.
from library.train import train_one_epoch, validate


def run_failure_analysis(val_df, preds, targets):
    """
    Analyzes the correlation between model error and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude
    # preds are probabilities, targets are 0/1
    errors = np.abs(targets - preds)

    # Add error to dataframe (ensure alignment)
    # The val_loader iterates sequentially, and val_df should match if not shuffled (shuffle=False in utils)
    analysis_df = val_df.copy()
    if len(analysis_df) != len(errors):
        print(
            f"Warning: Validation DF length ({len(analysis_df)}) != Predictions length ({len(errors)}). Skipping detailed analysis."
        )
        return

    analysis_df["error"] = errors

    # Features to analyze
    features = ["age", "density", "machine_id", "site_id", "laterality_idx", "view_idx"]

    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        if feat not in analysis_df.columns:
            # Try to find encoded version if original not present
            if f"{feat}_idx" in analysis_df.columns:
                feat = f"{feat}_idx"
            else:
                continue

        # Drop NaNs for correlation
        tmp_df = analysis_df[[feat, "error"]].dropna()

        if len(tmp_df) < 2:
            continue

        # Ensure numeric
        if not pd.api.types.is_numeric_dtype(tmp_df[feat]):
            continue

        corr, pval = stats.spearmanr(tmp_df[feat], tmp_df["error"])
        print(f"  {feat}: Correlation={corr:.4f}, p-value={pval:.4f}")


def generate_submission(model, test_loader, device, threshold_met):
    """
    Generates submission file if threshold is met.
    """
    if not threshold_met:
        print("Validation metric did not meet threshold. Skipping submission.")
        return

    print("\nGenerating Submission...")
    model.eval()

    # We need to map back to prediction_id.
    test_df = test_loader.dataset.df.copy()
    raw_preds = np.zeros(len(test_df))

    with torch.no_grad():
        # Iterate without tqdm for cleaner output
        for batch in test_loader:
            imgs = batch["image"].to(device)
            cats = batch["categorical"].to(device)
            conts = batch["continuous"].to(device)
            indices = batch["idx"].cpu().numpy()

            # Forward (Main head only)
            logits, _ = model(imgs, cats, conts)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Assign to correct index
            raw_preds[indices] = probs

    test_df["cancer_prob"] = raw_preds

    # Aggregation: Group by prediction_id and take MAX
    submission_df = test_df.groupby("prediction_id")["cancer_prob"].max().reset_index()
    submission_df.rename(columns={"cancer_prob": "cancer"}, inplace=True)

    # Save
    sub_path = Config.SUBMISSION_PATH
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(submission_df.head())


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 1  # Limit to 1 epoch for speed

    set_seed(Config.SEED)
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    logger = setup_logger(os.path.join(Config.WORK_DIR, "run.log"))
    device = get_device()
    logger.info(f"Running on device: {device}")

    # 2. Data Loading
    logger.info("Loading Data...")
    # Use cached data as requested
    train_loader, val_loader, test_loader, feature_meta = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = DSGEHNModel(feature_meta, pretrained=Config.PRETRAINED)
    model.to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    # Loss Function
    pos_weight = torch.tensor(Config.POS_WEIGHT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Training Loop (Fast Baseline: 1 Epoch)
    logger.info("Starting Training (Fast Baseline)...")

    best_pf1 = 0.0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scheduler
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val pF1={val_pf1:.6f}"
        )

        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_pf1": best_pf1,
                },
                is_best=True,
            )

    # 6. Final Validation & Metric Printing
    # Load best model for final evaluation
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded best model for final validation.")

    # Run validation again to get predictions for analysis
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            cats = batch["categorical"].to(device)
            conts = batch["continuous"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            final_logits, _ = model(imgs, cats, conts)
            probs = torch.sigmoid(final_logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets)

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    final_metric = probabilistic_f1(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    run_failure_analysis(val_loader.dataset.df, all_preds, all_targets)

    # 8. Submission
    threshold = 0.044888656586408615
    generate_submission(model, test_loader, device, final_metric > threshold)


if __name__ == "__main__":
    main()
