import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import provided library modules
from library import config, utils, data, model, train


def run_pipeline():
    # 1. Setup and Configuration Overrides
    # Limit epochs for fast baseline execution as requested
    config.NUM_EPOCHS = 15

    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # Device configuration
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing HC_HIDN model...")
    net = model.HC_HIDN().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_metric = float("inf")
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # Train one epoch
        train_loss = train.train_epoch(net, train_loader, optimizer, device)

        # Validate
        val_metric = train.validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step(val_metric)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_metric:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save best model
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(net.state_dict(), config.MODEL_PATH)
            print(f"  New best model saved! (MCRMSE: {best_metric:.6f})")

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    net.eval()

    # Re-run validation to get detailed predictions for analysis
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for inputs, partner_indices, targets, sample_ids in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass (get refined prediction y2)
            _, y2 = net(inputs, partner_indices)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(sample_ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metric = utils.calculate_global_mcrmse(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Calculate error per sample
    # Filter for scored positions and columns
    preds_scored = all_preds[:, : config.SEQ_SCORED, :][
        :, :, config.SCORED_TARGET_INDICES
    ]
    targets_scored = all_targets[:, : config.SEQ_SCORED, :][
        :, :, config.SCORED_TARGET_INDICES
    ]

    # MSE per sample (averaging over sequence length and targets)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to correlate with features
    val_df = pd.read_csv(config.VAL_CSV)

    # Ensure alignment by ID
    # Create a map from ID to error
    error_map = dict(zip(all_ids, rmse_per_sample))
    val_df["model_error"] = val_df["id"].map(error_map)

    # Drop rows where error might be missing (shouldn't happen if loaders are correct)
    val_df = val_df.dropna(subset=["model_error"])

    # Features to check
    analysis_features = ["signal_to_noise", "mean_reactivity", "seq_length"]
    # Add sequence composition features
    val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
    val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
    val_df["pct_paired"] = val_df["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )

    analysis_features.extend(["pct_A", "pct_G", "pct_paired"])

    print("Correlation between Model Error (RMSE) and Input Features:")
    for feat in analysis_features:
        if feat in val_df.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(val_df[feat]):
                corr, _ = pearsonr(val_df["model_error"], val_df[feat])
                print(f"  {feat}: {corr:.4f}")

    # 6. Conditional Submission
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        train.generate_submission(net, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
