import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.layers import RNAModel
from library.engine import train_fn, eval_fn, inference_fn
from library.model import generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Get Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            neighbor_indices = batch["neighbor_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(inputs, neighbor_indices, pair_masks)

            # Slice to scored length
            preds = preds[:, : Config.SEQ_SCORED, :]
            targets = targets[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate Error per Sample (Mean RMSE across scored columns/positions)
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # MSE per sample: (N, L, C) -> (N,)
    mse_per_sample = torch.mean((preds_scored - targets_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata for Correlation
    # We need to map IDs to metadata features
    val_df = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id", how="left")

    # 4. Feature Engineering for Correlation
    # Extract sequence properties
    merged_df["pct_A"] = merged_df["sequence"].apply(lambda s: s.count("A") / len(s))
    merged_df["pct_G"] = merged_df["sequence"].apply(lambda s: s.count("G") / len(s))
    merged_df["pct_C"] = merged_df["sequence"].apply(lambda s: s.count("C") / len(s))
    merged_df["pct_U"] = merged_df["sequence"].apply(lambda s: s.count("U") / len(s))
    merged_df["pct_unpaired"] = merged_df["structure"].apply(
        lambda s: s.count(".") / len(s)
    )

    # Select features to correlate
    features = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_unpaired",
    ]

    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 45)

    correlations = {}
    for feat in features:
        if feat in merged_df.columns:
            # Ensure numeric
            try:
                corr = merged_df[feat].astype(float).corr(merged_df["error"])
                correlations[feat] = corr
                print(f"{feat:<20} | {corr:.4f}")
            except Exception:
                pass

    return correlations


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAModel().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_mcrmse = eval_fn(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    final_metric = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission Generation
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        preds, ids = inference_fn(model, test_loader, device)

        # Reshape for submission: (N, 107, 5)
        # inference_fn returns concatenated numpy array of shape (N, 107, 5)

        generate_submission(preds, ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
