import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, evaluate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and metadata features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []
    scored_len = Config.SCORED_SEQ_LENGTH

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            structure_dist = batch["structure_dist"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            outputs = model(sequence, loop_type, structure_dist)

            # Slice to scored length
            outputs_scored = outputs[:, :scored_len, :]
            targets_scored = targets[:, :scored_len, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # 2. Calculate Sample-wise MSE (Mean over seq_len and targets)
    # Shape: (N_samples, Seq_Len, Targets) -> (N_samples,)
    sample_mse = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # 3. Load Metadata
    val_df = pd.read_parquet(Config.VAL_PATH)

    # Ensure alignment by ID
    analysis_df = pd.DataFrame({"id": all_ids, "error_rmse": sample_rmse})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id", how="left")

    # 4. Feature Engineering for Correlation
    # Calculate sequence properties
    merged_df["len_A"] = merged_df["sequence"].apply(lambda x: x.count("A"))
    merged_df["len_G"] = merged_df["sequence"].apply(lambda x: x.count("G"))
    merged_df["len_C"] = merged_df["sequence"].apply(lambda x: x.count("C"))
    merged_df["len_U"] = merged_df["sequence"].apply(lambda x: x.count("U"))
    merged_df["gc_content"] = (merged_df["len_G"] + merged_df["len_C"]) / merged_df[
        "seq_length"
    ]

    # 5. Compute Correlations
    features = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
        "gc_content",
    ]

    print("-" * 40)
    print("Correlation with Error (RMSE):")
    print("-" * 40)

    correlations = {}
    for feat in features:
        if feat in merged_df.columns:
            # Handle potential NaN in metadata
            valid_data = merged_df[[feat, "error_rmse"]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feat].corr(valid_data["error_rmse"])
                correlations[feat] = corr
                print(f"{feat:<20}: {corr:.4f}")
    print("-" * 40)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    # Using full training set (1728 samples) is fast enough (approx 1-2 mins per epoch on CPU, seconds on GPU)
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        debug=False,  # Use full dataset
    )

    # 3. Model Initialization
    model = RNAModel(Config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_mcrmse = evaluate(model, val_loader, device)

        scheduler.step()

        # Simple logging
        # print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_mcrmse:.4f}")

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print("Training complete.")

    # 6. Final Validation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_metric = evaluate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.6176461577
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        os.makedirs(submission_dir, exist_ok=True)
        generate_submission(model, test_loader, device, submission_path)
    else:
        print(
            f"Metric ({final_metric:.6f}) >= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
