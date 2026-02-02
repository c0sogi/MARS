import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Suppress tqdm progress bars from library modules
import tqdm

tqdm.tqdm = lambda x, **kwargs: x

# Import library components
from library.config import Config
from library.dataset import load_data
from library.model import WideResBiGRU
from library.utils import mcrmse_loss
from library.train import train_one_epoch, validate
from library.predict import predict


def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates it with metadata features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["distance"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)

            # Slice to scored length
            preds_scored = preds[:, : Config.PRED_LEN, :]
            targets_scored = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Per-Sample Error (Mean of Column RMSEs)
    # Shape: (N, 68, 3)
    # MSE per sample per column: mean over dim 1 (sequence)
    mse_per_col = np.mean((all_preds - all_targets) ** 2, axis=1)  # (N, 3)
    rmse_per_col = np.sqrt(mse_per_col)  # (N, 3)
    # Mean over columns
    sample_errors = np.mean(rmse_per_col, axis=1)  # (N,)

    # 3. Load Metadata
    val_df = pd.read_parquet(Config.VAL_FILE)
    # Ensure alignment by ID
    val_df = val_df.set_index("id").loc[all_ids].reset_index()

    # 4. Extract Features for Correlation
    analysis_df = pd.DataFrame(
        {
            "error": sample_errors,
            "signal_to_noise": (
                val_df["signal_to_noise"].values
                if "signal_to_noise" in val_df.columns
                else 0
            ),
            "SN_filter": (
                val_df["SN_filter"].values if "SN_filter" in val_df.columns else 0
            ),
            "len_A": val_df["sequence"].apply(lambda x: x.count("A")).values,
            "len_G": val_df["sequence"].apply(lambda x: x.count("G")).values,
            "len_C": val_df["sequence"].apply(lambda x: x.count("C")).values,
            "len_U": val_df["sequence"].apply(lambda x: x.count("U")).values,
        }
    )

    # 5. Compute Correlations
    print("-" * 40)
    print(f"{'Feature':<20} | {'Correlation (r)':<15}")
    print("-" * 40)

    features = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]
    for feat in features:
        if feat in analysis_df.columns and analysis_df[feat].std() > 0:
            corr, _ = pearsonr(analysis_df[feat], analysis_df["error"])
            print(f"{feat:<20} | {corr:.4f}")
        else:
            print(f"{feat:<20} | N/A (Constant or Missing)")
    print("-" * 40)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Use Config defaults
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Load Data
    print("Loading datasets...")
    train_dataset = load_data(mode="train", load_cached_data=True)
    val_dataset = load_data(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = WideResBiGRU().to(device)
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mcrmse = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Calculate final metric on full validation set
    final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Conditional Submission
    THRESHOLD = 0.6199890971183777
    if final_val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_val_metric:.6f}) meets threshold ({THRESHOLD:.6f})."
        )
        print("Generating submission...")
        # Use the provided predict function which handles test loading and formatting
        predict(device=Config.DEVICE)
    else:
        print(
            f"\nValidation metric ({final_val_metric:.6f}) did not meet threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
