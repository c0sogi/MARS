import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import HybridResNetBiGRU
from library.loss import MaskedHuberLoss
from library.train import train_one_epoch, validate, generate_submission
from library.utils import mcrmse_metric


def main():
    # 1. Configuration & Setup
    # Override defaults for a fast baseline execution
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 32
    # Ensure we use the full dataset (it is small enough)
    Config.DEBUG = False

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = HybridResNetBiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss Function
    criterion = MaskedHuberLoss(delta=Config.HUBER_DELTA)

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_mcrmse:.4f}"
        )

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation Assessment
    print(f"Loading best model from {best_model_path} for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-run validation to get predictions and IDs for analysis
    model.eval()
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            structure = batch["structure"].to(device)
            loop_type = batch["predicted_loop_type"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            preds = model(sequence, structure, loop_type)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric on Scored Positions (First 68 bases)
    scored_preds = val_preds[:, : Config.SCORED_LEN, :]
    scored_targets = val_targets[:, : Config.SCORED_LEN, :]

    final_metric = mcrmse_metric(scored_targets, scored_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate RMSE per sample
    # Shape: (N, 68, 5) -> (N,)
    diff = scored_preds - scored_targets
    mse_per_sample = np.mean(diff**2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Map errors to dataframe
    # We map by ID to ensure correct alignment
    error_map = dict(zip(val_ids, rmse_per_sample))
    df_val["rmse_error"] = df_val["id"].map(error_map)

    # Feature Engineering for Correlation
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))

    # Select features
    features = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_U", "len_C"]
    # Filter out columns that might not exist
    features = [f for f in features if f in df_val.columns]

    # Compute correlation
    correlations = (
        df_val[features + ["rmse_error"]].corr()["rmse_error"].drop("rmse_error")
    )
    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission Logic
    THRESHOLD = 0.7462618350982666
    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
