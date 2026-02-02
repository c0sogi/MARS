import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse, format_submission
from library.dataset import RNADataset
from library.model import TokenAdaptiveWideResBiGRU
from library.engine import train_fn, eval_fn, inference_fn


def run_failure_analysis(model, val_loader, val_dataset, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n=== Running Failure Analysis ===")
    model.eval()

    all_preds = []
    all_targets = []

    # 1. Get Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(sequences, loop_types, pair_dists)

            # Slice to scored length
            outputs_scored = outputs[:, : Config.PRED_LEN, :]
            targets_scored = targets[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 2. Compute Error per Sample (Mean RMSE across columns and positions)
    # Shape: (N, 68, 3) -> (N,)
    mse_per_sample = np.mean((preds - targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 3. Extract Metadata and Features
    df_val = val_dataset.df.copy()

    # Ensure alignment
    if len(df_val) != len(rmse_per_sample):
        print("Warning: Mismatch in validation set size during analysis.")
        return

    # Add Error to DataFrame
    df_val["model_error_rmse"] = rmse_per_sample

    # Extract/Create Features for Correlation
    # Feature 1: Signal to Noise (if available)
    if "signal_to_noise" not in df_val.columns:
        # Try to parse from SN_filter or ignore
        pass

    # Feature 2: Sequence Composition
    df_val["count_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["count_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["count_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["count_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # Select numeric columns for correlation
    cols_to_corr = [
        "model_error_rmse",
        "signal_to_noise",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]
    # Filter only existing columns
    cols_to_corr = [c for c in cols_to_corr if c in df_val.columns]

    # Compute Correlation
    corr_matrix = df_val[cols_to_corr].corr()
    error_correlations = corr_matrix["model_error_rmse"].drop("model_error_rmse")

    print("Correlation between Model Error (RMSE) and Features:")
    print(error_correlations)

    return float(np.mean(rmse_per_sample))


def main():
    # 1. Configuration Overrides for Fast Baseline
    # Reducing complexity and epochs to satisfy "fast baseline" requirement
    Config.EPOCHS = 10
    Config.NUM_LAYERS = 4
    Config.HIDDEN_DIM = 256

    # Set Seed
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    print("Loading Datasets...")
    train_dataset = RNADataset(mode="train", load_cached_data=True)
    val_dataset = RNADataset(mode="val", load_cached_data=True)

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
    print("Initializing Model...")
    model = TokenAdaptiveWideResBiGRU()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = float("inf")

    print(f"Starting Training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_score = eval_fn(model, val_loader, device)

        scheduler.step()

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training finished. Best Val MCRMSE: {best_score}")

    # 5. Final Validation & Metric
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Compute final metric on full validation set
    final_metric = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, val_dataset, device)

    # 7. Submission Logic
    THRESHOLD = 0.6209375959946717

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        test_dataset = RNADataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        preds, test_ids = inference_fn(model, test_loader, device)
        format_submission(preds, test_ids, Config.SUBMISSION_FILE)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
