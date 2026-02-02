import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE,
    SEED,
    PRED_LEN,
    VAL_PARQUET,
)
from library.utils import seed_everything, calculate_mcrmse
from library.loss import MaskedMSELoss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_ids = []
    all_losses = []

    criterion = MaskedMSELoss()

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(sequence, loop_type, pair_dist)

            # Calculate loss per sample
            # preds: (B, Seq, Targets), targets: (B, Seq, Targets)
            # MaskedMSELoss usually reduces to mean, we need per-sample error here.
            # We manually compute MSE on scored positions for each sample in the batch.

            preds_scored = preds[:, :PRED_LEN, :]
            targets_scored = targets[:, :PRED_LEN, :]

            # Squared error: (B, PRED_LEN, Targets)
            sq_error = (preds_scored - targets_scored) ** 2

            # Mean over positions and targets -> (B,)
            mse_per_sample = sq_error.mean(dim=(1, 2))

            all_losses.extend(mse_per_sample.cpu().numpy())
            all_ids.extend(ids)

    # Create a DataFrame for errors
    df_errors = pd.DataFrame({"id": all_ids, "mse_loss": all_losses})

    # Load validation metadata to get features
    if not os.path.exists(VAL_PARQUET):
        print("Validation parquet not found, skipping detailed metadata correlation.")
        return

    df_val = pd.read_parquet(VAL_PARQUET)

    # Merge errors with metadata
    df_analysis = pd.merge(df_val, df_errors, on="id", how="inner")

    # Features to analyze
    # We derive some features if they exist or calculate them
    if "signal_to_noise" in df_analysis.columns:
        corr, _ = pearsonr(df_analysis["mse_loss"], df_analysis["signal_to_noise"])
        print(f"Correlation (MSE vs Signal_to_Noise): {corr:.4f}")

    if "SN_filter" in df_analysis.columns:
        corr, _ = pearsonr(df_analysis["mse_loss"], df_analysis["SN_filter"])
        print(f"Correlation (MSE vs SN_filter): {corr:.4f}")

    # Analyze sequence composition correlations
    df_analysis["count_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))
    df_analysis["count_G"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
    df_analysis["count_U"] = df_analysis["sequence"].apply(lambda x: x.count("U"))
    df_analysis["count_C"] = df_analysis["sequence"].apply(lambda x: x.count("C"))

    for nuc in ["A", "G", "U", "C"]:
        corr, _ = pearsonr(df_analysis["mse_loss"], df_analysis[f"count_{nuc}"])
        print(f"Correlation (MSE vs Count_{nuc}): {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Device: {DEVICE}")

    # 2. Data Loading
    # Using load_cached_data=True as requested
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAModel().to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = MaskedMSELoss()

    best_mcrmse = float("inf")

    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_mcrmse = validate(model, val_loader, DEVICE)

        scheduler.step()

        # Simple logging
        print(
            f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    final_val_metric = validate(model, val_loader, DEVICE)
    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, DEVICE)

    # 8. Conditional Submission
    THRESHOLD = 0.6199890971183777
    if final_val_metric < THRESHOLD:
        print(
            f"Validation metric {final_val_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, DEVICE, SUBMISSION_FILE)
    else:
        print(
            f"Validation metric {final_val_metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
