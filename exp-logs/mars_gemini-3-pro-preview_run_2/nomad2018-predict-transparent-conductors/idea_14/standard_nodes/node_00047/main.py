import sys
import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, rmsle
from library.data import get_loaders
from library.model import CGCNN_IB
from library.train import train_one_epoch, validate, generate_submission


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline run
    Config.NUM_EPOCHS = 30
    Config.BATCH_SIZE = 64  # Increase batch size for speed

    set_seed(Config.RANDOM_SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Initializing DataLoaders...")
    # Load cached data if available to save processing time
    train_loader, val_loader, test_loader, scaler = get_loaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = CGCNN_IB(config=Config).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_val_score = float("inf")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device, scaler)

        # Scheduler Step
        scheduler.step(val_score)

        epoch_time = time.time() - start_time

        # Save best model
        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        print(
            f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Val RMSLE: {val_score:.6f}"
        )

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    print("Loading best model for assessment...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Collect predictions and targets for the entire validation set
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)

            # Inverse transform to get original units (eV)
            preds_original = scaler.inverse_transform(preds)
            targets_original = scaler.inverse_transform(batch.y)

            val_preds.append(preds_original.cpu().numpy())
            val_targets.append(targets_original.cpu().numpy())
            val_ids.extend(batch.id.cpu().numpy().flatten())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric
    final_metric = rmsle(torch.tensor(val_targets), torch.tensor(val_preds))
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate Mean Absolute Error per sample (averaged over the two targets)
    abs_errors = np.abs(val_preds - val_targets)
    mean_abs_error = np.mean(abs_errors, axis=1)

    # Load metadata to get feature values
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"id": val_ids, "error_magnitude": mean_abs_error})

    # Merge with metadata on ID
    full_analysis_df = pd.merge(analysis_df, val_meta, on="id")

    # Calculate correlations between error magnitude and numeric features
    numeric_cols = full_analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["id", "error_magnitude"] + Config.TARGET_COLS
    features = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for feat in features:
        if full_analysis_df[feat].std() > 0:  # Avoid constant columns
            corr = full_analysis_df["error_magnitude"].corr(full_analysis_df[feat])
            correlations[feat] = corr

    # Sort correlations by absolute value
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features (Top 10):")
    for feat, corr in sorted_corrs[:10]:
        print(f"  {feat:<30}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 7. Conditional Submission
    # -------------------------------------------------------------------------
    threshold = 0.05085437756413089

    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric} < {threshold}. Generating submission..."
        )
        generate_submission(model, test_loader, device, scaler, Config.SUBMISSION_PATH)
    else:
        print(f"\nValidation metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
