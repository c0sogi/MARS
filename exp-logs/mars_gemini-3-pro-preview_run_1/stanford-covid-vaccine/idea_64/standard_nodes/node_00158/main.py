import os
import time
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, save_checkpoint, mcrmse_metric
from library.data import get_dataloaders
from library.model import StructureInjectedWideResBiLSTM
from library.engine import train_one_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device, val_df_path):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates it with metadata features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_ids = []
    all_preds = []
    all_targets = []

    # 1. Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(seq, loop, dist)

            # Slice to scored positions
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # 2. Calculate Per-Sample Error (MCRMSE per sample)
    # y_true/pred shape: (N_samples, 68, 3)
    # MSE per sample per target: (N_samples, 3)
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=1)
    # RMSE per sample per target: (N_samples, 3)
    rmse_per_sample = np.sqrt(mse_per_sample)
    # Mean RMSE across targets per sample: (N_samples,)
    sample_errors = np.mean(rmse_per_sample, axis=1)

    # Create Error DataFrame
    df_errors = pd.DataFrame({"id": all_ids, "error": sample_errors})

    # 3. Load Metadata
    df_val = pd.read_parquet(val_df_path)

    # Merge
    df_analysis = pd.merge(df_val, df_errors, on="id", how="inner")

    # 4. Feature Engineering for Correlation
    # We want to check correlations with: signal_to_noise, SN_filter, base counts, structure counts
    df_analysis["len_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))
    df_analysis["len_G"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
    df_analysis["len_U"] = df_analysis["sequence"].apply(lambda x: x.count("U"))
    df_analysis["len_C"] = df_analysis["sequence"].apply(lambda x: x.count("C"))
    df_analysis["struct_open"] = df_analysis["structure"].apply(lambda x: x.count("("))

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_U",
        "len_C",
        "struct_open",
    ]

    print("Correlation between Model Error and Features:")
    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(df_analysis[feat]):
                corr, _ = pearsonr(df_analysis[feat], df_analysis["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not numeric, skipping.")
        else:
            print(f"  {feat}: Not found in metadata.")


def main():
    # 1. Configuration & Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Limit epochs for fast baseline execution
    MAX_EPOCHS = 15

    print(f"Initializing run on {device}...")

    # 2. Data Loaders
    # Using load_cached_data=True to leverage preprocessed files
    train_loader = get_dataloaders(
        split="train", batch_size=Config.BATCH_SIZE, load_cached_data=True
    )
    val_loader = get_dataloaders(
        split="val", batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    model = StructureInjectedWideResBiLSTM()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS  # Adjust T_max to actual epochs
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {MAX_EPOCHS} epochs...")

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved")

    print("Training complete.")

    # 6. Final Evaluation
    print(f"Final Validation Metric: {best_mcrmse}")

    # Load best model for analysis
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint)

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device, Config.VAL_FILE)

    # 8. Conditional Submission
    THRESHOLD = 0.6176461577
    if best_mcrmse < THRESHOLD:
        print(
            f"\nValidation score ({best_mcrmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(best_model_path, device)
    else:
        print(
            f"\nValidation score ({best_mcrmse}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
