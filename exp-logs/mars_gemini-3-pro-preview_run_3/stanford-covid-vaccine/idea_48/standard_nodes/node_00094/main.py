import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, compute_mcrmse_numpy
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, inference, generate_submission


def main():
    # 1. Setup
    # ---------------------------------------------------------
    # Adjust Config for a fast baseline run
    Config.epochs = 20  # Limit epochs for speed
    Config.debug_samples = None  # Use full dataset (it's small enough)

    seed_everything(Config.seed)
    Config.setup()
    device = torch.device(Config.device)

    print(f"Running on device: {device}")

    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_samples=Config.debug_samples, load_cached_data=True
    )

    # 3. Model Initialization
    # ---------------------------------------------------------
    model = RNAModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = MCRMSELoss()

    # 4. Training Loop
    # ---------------------------------------------------------
    print(f"Starting training for {Config.epochs} epochs...")
    best_score = float("inf")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging (minimal)
        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f} | Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_score < (best_score - Config.min_delta):
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)
            # Reset patience would go here, but we run fixed epochs for baseline

    # 5. Final Evaluation & Metric
    # ---------------------------------------------------------
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.eval()

    # Re-run validation to get exact predictions and score
    # We need sample-wise predictions for failure analysis
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices, pair_masks)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_ids.extend(ids)

    val_preds_np = np.concatenate(val_preds, axis=0)
    val_targets_np = np.concatenate(val_targets, axis=0)

    final_metric = compute_mcrmse_numpy(val_preds_np, val_targets_np)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to get features
    df_val = pd.read_parquet(Config.val_metadata_path)
    if Config.debug_samples:
        df_val = df_val.iloc[: Config.debug_samples]

    # Ensure alignment by ID
    # Create a map of ID -> Error
    # Calculate MCRMSE per sample
    # Slice to scored length and columns
    pred_len = Config.pred_len
    scored_indices = Config.scored_classes_indices

    # Slice
    vp_sliced = val_preds_np[:, :pred_len, :][:, :, scored_indices]
    vt_sliced = val_targets_np[:, :pred_len, :][:, :, scored_indices]

    # Compute RMSE per sample (average over columns and length)
    # Shape: (N, L, C) -> (N,)
    mse_per_sample = np.mean((vp_sliced - vt_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": val_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(df_val, on="id", how="left")

    # Feature Engineering for Correlation
    # 1. Signal to Noise
    # 2. SN Filter
    # 3. GC Content (from sequence)
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )

    # Calculate correlations
    correlations = {}
    features_to_check = ["signal_to_noise", "SN_filter", "gc_content"]

    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[feat])
            correlations[feat] = corr

    print("Correlation between Error and Input Features:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # 7. Conditional Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Inference on Test Set
        test_preds, test_ids = inference(model, test_loader, device)

        # Generate CSV
        generate_submission(test_preds, test_ids, Config.submission_file)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
