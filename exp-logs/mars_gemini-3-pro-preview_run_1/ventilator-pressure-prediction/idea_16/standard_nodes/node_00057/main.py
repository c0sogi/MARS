import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

# Import library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.dataset as dataset
import library.model as model_lib
import library.loss as loss_lib
import library.engine as engine


def run_pipeline():
    # 1. Setup
    print("Setting up environment...")
    utils.seed_everything(config.SEED)

    # Configuration Overrides for Fast Baseline
    # Reducing epochs and training data size to ensure execution within time limits
    config.EPOCHS = 6
    TRAIN_SAMPLE_BREATHS = 15000  # Subsample to ~1.2M rows for speed

    print(
        f"Configuration Overrides: EPOCHS={config.EPOCHS}, TRAIN_SAMPLE_BREATHS={TRAIN_SAMPLE_BREATHS}"
    )

    # 2. Data Loading
    print("Loading and preparing datasets...")
    # Load full datasets (cached if available)
    train_df, val_df, test_df = features.prepare_datasets(load_cached_data=True)

    # Subsample Training Data
    print(f"Subsampling training data to {TRAIN_SAMPLE_BREATHS} breaths...")
    unique_breaths = train_df["breath_id"].unique()
    if len(unique_breaths) > TRAIN_SAMPLE_BREATHS:
        # Deterministic slicing
        selected_breaths = unique_breaths[:TRAIN_SAMPLE_BREATHS]
        train_df = train_df[train_df["breath_id"].isin(selected_breaths)].copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Instantiate Datasets
    # We use the full validation and test sets for accurate metrics and submission
    train_ds = dataset.VentilatorDataset(train_df, is_test=False)
    val_ds = dataset.VentilatorDataset(val_df, is_test=False)
    test_ds = dataset.VentilatorDataset(test_df, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    device = config.DEVICE
    model = model_lib.VentilatorModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler needs correct steps_per_epoch based on subsampled data
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
        pct_start=config.PCT_START,
    )

    loss_fn = loss_lib.MaskedL1Loss(aux_weight=config.AUX_WEIGHT)

    # 4. Training Loop
    print(f"Starting training for {config.EPOCHS} epochs...")
    best_val_mae = float("inf")

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn
        )

        # Evaluate on full validation set
        val_mae = engine.evaluate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved (MAE: {best_val_mae:.6f})")

    print("Training complete.")

    # 5. Final Validation
    print("\n=== Final Validation ===")
    # Load the best model state
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()

    # Compute metric on the entire hold-out validation set
    final_metric = engine.evaluate(model, val_loader, device, loss_fn)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    print("Calculating correlation between error magnitude and input features...")

    # Map feature names to indices for extraction
    feat_map = {name: i for i, name in enumerate(config.FEATURE_NAMES)}
    target_feats = ["time_step", "u_in", "R", "C"]
    target_indices = [feat_map[f] for f in target_feats]

    all_errors = []
    all_features = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Inference
            pred, _ = model(x)
            if pred.dim() > y.dim():
                pred = pred.squeeze(-1)

            # Calculate Absolute Error
            error = torch.abs(pred - y)

            # Mask: Only consider inspiratory phase (u_out == 0)
            mask = (1 - u_out).bool()

            # Extract valid errors
            valid_errors = error[mask].cpu().numpy()

            # Extract corresponding features
            # Flatten batch and sequence dims, then apply mask
            x_flat = x.view(-1, x.shape[-1])
            mask_flat = mask.view(-1)
            valid_feats = x_flat[mask_flat][:, target_indices].cpu().numpy()

            all_errors.append(valid_errors)
            all_features.append(valid_feats)

    # Concatenate results
    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features)

    # Compute Correlation
    analysis_df = pd.DataFrame(all_features, columns=target_feats)
    analysis_df["error"] = all_errors

    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation with Error Magnitude:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.2164510190486908

    if final_metric < THRESHOLD:
        print(
            f"\nFinal Metric {final_metric} is lower than {THRESHOLD}. Generating submission..."
        )

        ids, preds = engine.predict(model, test_loader, device)

        submission_df = pd.DataFrame({"id": ids, "pressure": preds})

        # Ensure sorted by ID
        submission_df.sort_values("id", inplace=True)

        submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
    else:
        print(
            f"\nFinal Metric {final_metric} is NOT lower than {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
