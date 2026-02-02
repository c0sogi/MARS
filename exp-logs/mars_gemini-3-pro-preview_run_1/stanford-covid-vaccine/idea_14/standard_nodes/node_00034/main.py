import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse, save_submission
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import train_one_epoch, validate, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    set_seed(Config.SEED)

    # Use full epochs for convergence
    # Config.EPOCHS is 50 in config.py

    # Set specific submission path as requested
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = RNAModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MaskedMSELoss()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_mcrmse = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation Step
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"Epoch {epoch}: New best MCRMSE: {best_mcrmse:.6f}")

    # -------------------------------------------------------------------------
    # 5. Final Validation & Metric
    # -------------------------------------------------------------------------
    print("Loading best model for final evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current state.")

    model.eval()

    # Collect full validation predictions for metric and failure analysis
    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, dist)

            # Slice to scored length (68)
            preds_sliced = preds[:, : Config.PRED_LEN, :]
            targets_sliced = targets[:, : Config.PRED_LEN, :]

            all_val_preds.append(preds_sliced.cpu().numpy())
            all_val_targets.append(targets_sliced.cpu().numpy())

    val_preds_arr = np.concatenate(all_val_preds, axis=0)
    val_targets_arr = np.concatenate(all_val_targets, axis=0)

    final_metric = mcrmse(val_targets_arr, val_preds_arr)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample: sqrt(mean((y-y_pred)^2))
    # Shape is (N_samples, 68, 3), average over seq_len and channels
    mse_per_sample = np.mean((val_targets_arr - val_preds_arr) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to correlate errors with features
    # Note: val_loader is shuffle=False, so order matches the parquet file
    try:
        df_val = pd.read_parquet(Config.VAL_FILE)

        # Attach error metric
        df_val["error_rmse"] = rmse_per_sample

        # Feature Engineering for correlation
        df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
        df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
        df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
        df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))
        df_val["GC_content"] = (df_val["len_G"] + df_val["len_C"]) / df_val[
            "sequence"
        ].str.len()

        # Define features to check
        features_to_check = [
            "signal_to_noise",
            "SN_filter",
            "len_A",
            "len_G",
            "len_C",
            "len_U",
            "GC_content",
        ]

        print("Correlation between Error (RMSE) and Features:")
        for feat in features_to_check:
            if feat in df_val.columns:
                # Ensure numeric
                if pd.api.types.is_numeric_dtype(df_val[feat]):
                    corr = df_val[feat].corr(df_val["error_rmse"])
                    print(f"  {feat}: {corr:.4f}")

    except Exception as e:
        print(f"Failure analysis skipped due to error: {e}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6226052641868591

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        test_ids, test_preds = predict(model, test_loader, device)
        save_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
