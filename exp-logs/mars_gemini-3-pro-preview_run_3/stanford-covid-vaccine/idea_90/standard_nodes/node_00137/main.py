import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import HighCapacityRNAnet
from library.engine import train_model, validate, generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating model error with input features.
    """
    print("Running Failure Analysis...")

    # Load validation metadata to access features
    if not os.path.exists(Config.VAL_METADATA):
        print(f"Metadata file not found at {Config.VAL_METADATA}. Skipping analysis.")
        return

    val_df = pd.read_parquet(Config.VAL_METADATA)

    model.eval()

    # Indices for scored targets: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    pred_len = Config.PRED_LEN

    sample_errors = []
    sample_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            # Forward pass
            outputs = model(inputs, pair_indices, pair_masks)

            # Slice to scored length (68) and scored columns
            outputs_sliced = outputs[:, :pred_len, scored_indices]
            targets_sliced = targets[:, :pred_len, scored_indices]

            # Calculate MCRMSE per sample
            # 1. MSE per column per sample: Mean over sequence length (dim 1)
            mse_per_col = torch.mean((outputs_sliced - targets_sliced) ** 2, dim=1)
            # 2. RMSE per column per sample
            rmse_per_col = torch.sqrt(mse_per_col)
            # 3. Mean across columns (MCRMSE per sample)
            mcrmse_per_sample = torch.mean(rmse_per_col, dim=1)

            sample_errors.extend(mcrmse_per_sample.cpu().numpy())
            sample_ids.extend(ids)

    # Create a DataFrame for correlation analysis
    error_df = pd.DataFrame({"id": sample_ids, "error": sample_errors})

    # Merge with metadata
    merged_df = pd.merge(error_df, val_df, on="id")

    # Feature Engineering for correlation
    # Calculate nucleotide content and paired percentage
    merged_df["pct_A"] = merged_df["sequence"].apply(lambda x: x.count("A") / len(x))
    merged_df["pct_G"] = merged_df["sequence"].apply(lambda x: x.count("G") / len(x))
    merged_df["pct_C"] = merged_df["sequence"].apply(lambda x: x.count("C") / len(x))
    merged_df["pct_U"] = merged_df["sequence"].apply(lambda x: x.count("U") / len(x))
    merged_df["pct_paired"] = merged_df["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )

    # List of features to check
    features = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_paired",
    ]

    print("\nCorrelation between Model Error (MCRMSE) and Input Features:")
    for feat in features:
        if feat in merged_df.columns:
            # Drop NaNs just in case
            valid_data = merged_df[[feat, "error"]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feat].corr(valid_data["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Column not found in metadata")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Fast Baseline Configuration
    # Reducing epochs to ensure quick execution while using full data
    Config.NUM_EPOCHS = 15
    print(
        f"Configuration: Device={device}, Epochs={Config.NUM_EPOCHS}, BatchSize={Config.BATCH_SIZE}"
    )

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    print("Initializing HighCapacityRNAnet...")
    model = HighCapacityRNAnet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 5. Training
    print("Starting Training...")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.NUM_EPOCHS,
        patience=5,
    )

    # 6. Validation & Metrics
    print("Loading best model for validation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current model state.")

    print("Calculating Final Validation Metric...")
    val_score = validate(model, val_loader, device)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.5884495377540588
    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation score ({val_score}) is above threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
