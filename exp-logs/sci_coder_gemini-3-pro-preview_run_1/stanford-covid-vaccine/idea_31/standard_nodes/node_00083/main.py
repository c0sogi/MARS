import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import train_one_epoch, validate, generate_submission


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set by correlating
    error magnitude with input features.
    """
    print("\n=== Starting Failure Analysis ===")
    model.eval()

    all_ids = []
    all_preds = []
    all_targets = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequences"].to(device)
            loop_type = batch["loop_types"].to(device)
            pair_dist = batch["pair_dists"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            # Forward pass
            preds = model(sequence, loop_type, pair_dist)

            # Slice to scored length
            preds_sliced = preds[:, : Config.PRED_LEN, :]

            all_ids.extend(ids)
            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate
    y_pred = np.vstack(all_preds)  # (N, 68, 3)
    y_true = np.vstack(all_targets)  # (N, 68, 3)

    # 2. Calculate Error Per Sample
    # Metric is Mean Columnwise RMSE. We calculate this per sample.
    # RMSE per column for each sample: sqrt(mean((y-y_hat)^2, axis=1)) -> shape (N, 3)
    mse_per_sample_col = np.mean((y_true - y_pred) ** 2, axis=1)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # Average across columns -> shape (N,)
    error_per_sample = np.mean(rmse_per_sample_col, axis=1)

    # Create Error DataFrame
    df_error = pd.DataFrame({"id": all_ids, "error_magnitude": error_per_sample})

    # 3. Load Metadata for Features
    # We use the validation parquet file directly
    df_val_meta = pd.read_parquet(Config.VAL_FILE)

    # Merge
    df_analysis = pd.merge(df_error, df_val_meta, on="id", how="inner")

    # 4. Feature Engineering for Analysis
    # Nucleotide counts
    df_analysis["count_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))
    df_analysis["count_G"] = df_analysis["sequence"].apply(lambda x: x.count("G"))
    df_analysis["count_C"] = df_analysis["sequence"].apply(lambda x: x.count("C"))
    df_analysis["count_U"] = df_analysis["sequence"].apply(lambda x: x.count("U"))

    # 5. Compute Correlations
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]

    print("Correlation between Error Magnitude and Features:")
    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Drop NaNs if any (though data should be clean)
            valid_data = df_analysis[[feat, "error_magnitude"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error_magnitude"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature not found")

    print("=== Failure Analysis Complete ===\n")


def main():
    # 1. Setup
    Config.setup_environment()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using the full dataset as it is small enough for fast execution
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = RNAModel(config=Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MaskedMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

        # Save Best
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training finished. Best MCRMSE: {best_mcrmse:.5f}")

    # 6. Final Evaluation & Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Re-calculate metric on full validation set to ensure accuracy and print required format
    _, final_metric = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.6199890971183777

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
