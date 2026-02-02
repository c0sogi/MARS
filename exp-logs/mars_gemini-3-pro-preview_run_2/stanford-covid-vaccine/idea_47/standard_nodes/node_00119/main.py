import os
import sys
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss, GlobalMCRMSE, format_submission
from library.data import get_dataloaders
from library.model import PFDRN
from library.train import train_epoch, validate, predict


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Correlates model error with input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Get Predictions on Validation Set
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            pairing_mask = batch["pairing_mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Forward pass (returns only preds_2 in eval mode)
            preds = model(inputs, partner_indices, pairing_mask)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    preds_arr = np.concatenate(all_preds, axis=0)  # (N, L, 5)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, L, 5)

    # 2. Calculate Per-Sample RMSE
    # Filter to scored columns and positions
    scored_len = Config.SEQ_SCORED
    scored_cols = Config.SCORED_INDICES

    p_scored = preds_arr[:, :scored_len, scored_cols]
    t_scored = targets_arr[:, :scored_len, scored_cols]

    # MSE per sample: Mean over Length and Columns
    mse_per_sample = np.mean((p_scored - t_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 3. Load Metadata for Features
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Ensure alignment (ids should match, but we'll map just in case)
    # The loader order is deterministic (shuffle=False), so we can assume alignment
    # if the CSV hasn't changed. To be safe, we create a map.
    id_to_rmse = dict(zip(all_ids, rmse_per_sample))

    val_df["rmse"] = val_df["id"].map(id_to_rmse)

    # Drop rows where id might not be in the loader (shouldn't happen)
    val_df = val_df.dropna(subset=["rmse"])

    # 4. Feature Engineering for Correlation
    # Base counts
    val_df["count_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
    val_df["count_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
    val_df["count_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
    val_df["count_U"] = val_df["sequence"].apply(lambda x: x.count("U"))

    features = [
        "signal_to_noise",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
        "seq_length",
    ]

    print(f"Correlation with Model Error (RMSE) across {len(val_df)} samples:")
    print(f"{'Feature':<20} {'Correlation':<10}")
    print("-" * 35)

    for feat in features:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["rmse"])
            print(f"{feat:<20} {corr:.4f}")

    print("-" * 35)


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 25  # Reduced from 50 to ensure quick completion

    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)

    print(f"Running PF-DRN Pipeline on {device}")
    print(f"Training for {Config.EPOCHS} epochs.")

    # 2. Data
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = PFDRN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 4. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("\nStarting Training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f} | Time: {elapsed:.1f}s"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            # print("  >>> Saved Best Model")

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Conditional Submission
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions
        test_preds, test_ids = predict(model, test_loader, device)

        # Save
        save_path = "./submission/submission.csv"
        format_submission(test_ids, test_preds, save_path=save_path)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
