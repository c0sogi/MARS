import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from library.config import Config
from library.utils import seed_everything, MetricTracker, format_submission
from library.loss import MCRMSELoss
from library.data import get_loaders, get_test_loader
from library.model import HybridNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    steps = 0

    for inputs, indices, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Move indices tensors to device
        indices = {k: v.to(device) for k, v in indices.items()}

        # Slice targets to scored length (first 68)
        targets_scored = targets[:, : Config.PRED_LEN, :]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, indices)

        # Slice outputs to scored length
        outputs_scored = outputs[:, : Config.PRED_LEN, :]

        loss = criterion(outputs_scored, targets_scored)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps += 1

    return running_loss / max(steps, 1)


def validate(model, loader, device):
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for inputs, indices, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            indices = {k: v.to(device) for k, v in indices.items()}

            outputs = model(inputs, indices)

            # Slice to scored region
            outputs_scored = outputs[:, : Config.PRED_LEN, :]
            targets_scored = targets[:, : Config.PRED_LEN, :]

            tracker.update(outputs_scored, targets_scored)

    return tracker.result()


def perform_failure_analysis(model, loader, device):
    print("\n==== Failure Analysis ====")
    model.eval()

    all_errors = []
    all_ids = []

    # 1. Calculate per-sample error
    scored_cols = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

    with torch.no_grad():
        for inputs, indices, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            indices = {k: v.to(device) for k, v in indices.items()}

            outputs = model(inputs, indices)

            # Slice
            p = outputs[:, : Config.PRED_LEN, :].cpu().numpy()
            t = targets[:, : Config.PRED_LEN, :].cpu().numpy()

            # Calculate RMSE per sample across scored columns and positions
            # Shape: (B, 68, 5)
            # Select scored columns
            p_scored = p[:, :, scored_cols]
            t_scored = t[:, :, scored_cols]

            # MSE per sample: mean over length and channels
            mse_per_sample = np.mean((p_scored - t_scored) ** 2, axis=(1, 2))
            rmse_per_sample = np.sqrt(mse_per_sample)

            all_errors.extend(rmse_per_sample)

            # We need IDs to map back to metadata.
            # The loader yields (inputs, indices, targets).
            # The RNADataset in library.config doesn't yield IDs in train/val mode by default.
            # However, the dataset object has .ids attribute.
            # Since shuffle=False for val_loader, we can match by index if we iterate carefully,
            # but let's load metadata separately and assume order is preserved (which it is for val_loader).

    # Load metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Verify length
    if len(all_errors) != len(val_df):
        print(
            f"Warning: Number of errors ({len(all_errors)}) does not match metadata length ({len(val_df)}). Skipping detailed correlation."
        )
        return

    val_df["model_rmse"] = all_errors

    # Features to correlate
    features = ["signal_to_noise", "mean_reactivity"]

    # Add sequence composition features
    val_df["count_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
    val_df["count_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
    val_df["count_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
    val_df["count_U"] = val_df["sequence"].apply(lambda x: x.count("U"))

    features.extend(["count_A", "count_G", "count_C", "count_U"])

    print(f"{'Feature':<20} | {'Correlation with Error':<25}")
    print("-" * 50)

    for feat in features:
        if feat in val_df.columns:
            # Handle NaNs if any
            valid_df = val_df[[feat, "model_rmse"]].dropna()
            if len(valid_df) > 0:
                corr, _ = stats.pearsonr(valid_df[feat], valid_df["model_rmse"])
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | N/A (No valid data)")
        else:
            print(f"{feat:<20} | Not found in metadata")


def generate_submission(model, device):
    print("\nGenerating submission...")
    test_loader = get_test_loader()
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, indices, ids in test_loader:
            inputs = inputs.to(device)
            indices = {k: v.to(device) for k, v in indices.items()}

            outputs = model(inputs, indices)

            # Outputs are (B, 107, 5)
            # We need predictions for all positions as per task description
            # "For each sample id in the test set, you must predict targets for each sequence position"

            preds_np = outputs.cpu().numpy()

            all_preds.extend(preds_np)
            all_ids.extend(ids)

    # Format and save
    format_submission(all_preds, all_ids, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data
    train_loader, val_loader = get_loaders()

    # 3. Model
    model = HybridNet().to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_metric = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metric = validate(model, val_loader, device)

        scheduler.step(val_metric)

        # Save best model
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Final Validation & Output
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    final_metric = validate(model, val_loader, device)

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    # Threshold from task description
    THRESHOLD = 0.5421870350837708

    if final_metric < THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
