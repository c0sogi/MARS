import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import library modules
from library import config, utils, data, model, train, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override config for fast baseline execution
    config.NUM_EPOCHS = 35
    config.PATIENCE = 7

    # Set seeds for reproducibility
    utils.seed_everything(config.SEED)

    # Device configuration
    device = torch.device(config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load dataloaders with caching enabled
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    # Initialize model
    net = model.PartnerAwareHybridNet()
    net.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Loss Function
    criterion = utils.MCRMSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        # Training step
        train_loss = train.train_fn(net, train_loader, optimizer, criterion, device)

        # Validation step
        val_loss = train.eval_fn(net, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            break

    # -------------------------------------------------------------------------
    # 4. Validation Assessment
    # -------------------------------------------------------------------------
    # Load best model for final evaluation
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    # Calculate final metric on the full validation set
    final_val_metric = train.eval_fn(net, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    net.eval()
    val_ids = []
    val_rmses = []

    # We need to compute error per sample to correlate with metadata
    # The validation loader shuffles=False, so order is preserved relative to dataset.ids

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)

            # Slice to scored sequence length (68)
            if outputs.shape[1] > targets.shape[1]:
                outputs = outputs[:, : targets.shape[1], :]

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate RMSE per sample (averaged over columns and sequence positions)
    # Shape: (N_samples, 68, 5)
    # MSE per sample: mean over axes 1 and 2
    mse_per_sample = np.mean((all_preds - all_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to get features
    val_df = pd.read_csv(config.VAL_CSV)

    # Ensure alignment (val_loader ids should match val_df ids if sorted or preserved)
    # The data loader uses ids from cache which came from the dataframe order.
    # We can double check using the ids in the dataset
    loader_ids = val_loader.dataset.ids

    # Create a dataframe for analysis
    analysis_df = pd.DataFrame({"id": loader_ids, "rmse": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Calculate correlations
    # 1. Correlation with Signal to Noise
    if "signal_to_noise" in analysis_df.columns:
        corr_sn, _ = pearsonr(analysis_df["rmse"], analysis_df["signal_to_noise"])
        print(f"Correlation between Error (RMSE) and Signal_to_Noise: {corr_sn:.4f}")

    # 2. Correlation with Mean Reactivity (proxy for signal magnitude)
    if "mean_reactivity" in analysis_df.columns:
        corr_react, _ = pearsonr(analysis_df["rmse"], analysis_df["mean_reactivity"])
        print(f"Correlation between Error (RMSE) and Mean Reactivity: {corr_react:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6477736930052439

    if final_val_metric < THRESHOLD:
        predict.generate_submission()


if __name__ == "__main__":
    main()
