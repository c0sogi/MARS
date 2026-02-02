import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import SLHDAN
from library.train import train_one_epoch, loss_fn, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def analyze_failures(val_df, preds, targets):
    """
    Performs failure analysis by correlating errors with features.
    """
    # Calculate absolute error
    val_df = val_df.copy()
    val_df["Pred_FVC"] = preds
    val_df["Abs_Error"] = np.abs(val_df["FVC"] - val_df["Pred_FVC"])

    print("\nFailure Analysis (Correlation with Absolute Error):")
    features = ["Age", "Percent", "Weeks"]

    # Encode categorical if needed, but standard correlation works on numericals
    # Sex and Smoking are categorical, we'll focus on continuous vars for simple correlation
    for feat in features:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["Abs_Error"])
            print(f"Correlation with {feat}: {corr:.4f}")

    # Check bias
    bias = np.mean(val_df["Pred_FVC"] - val_df["FVC"])
    print(f"Mean Prediction Bias: {bias:.4f}")


def validate_and_analyze(model, loader, device):
    """
    Runs inference on validation set, computes metric, and returns data for analysis.
    """
    model.eval()
    all_targets = []
    all_fvc_preds = []
    all_sigma_preds = []

    # We need to align predictions with the dataframe for analysis.
    # The loader is shuffle=False, so order is preserved.

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, time_delta, baseline_fvc
            )

            all_targets.extend(target.cpu().numpy())
            all_fvc_preds.extend(fvc_pred.cpu().numpy())
            all_sigma_preds.extend(sigma_pred.cpu().numpy())

    # Calculate metric
    score = score_function(all_targets, all_fvc_preds, all_sigma_preds)

    return score, np.array(all_fvc_preds), np.array(all_targets)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    # 1100 samples is small, so 30 epochs is very fast on A100 (~5 mins)
    Config.EPOCHS = 30

    # 2. Data Loading
    # We load data. The library handles caching.
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False,  # Use full dataset for valid baseline
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    model = SLHDAN().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model_runfile.pth")

    # Simple training loop
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate (using the local function to get score)
        val_score, _, _ = validate_and_analyze(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Save Best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Run validation inference
    final_score, val_preds, val_targets = validate_and_analyze(
        model, val_loader, device
    )

    # REQUIRED PRINT
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    # Load validation metadata to get original features
    val_df = pd.read_csv(Config.VAL_CSV)
    # Ensure length matches (drop if debug mode was used, though we set debug=False)
    if len(val_df) != len(val_preds):
        # Fallback if sizes mismatch (e.g. drop_last or other loader quirks, though unlikely with defaults)
        val_df = val_df.iloc[: len(val_preds)]

    analyze_failures(val_df, val_preds, val_targets)

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = -6.510164260864258

    if final_score > THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation score {final_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
