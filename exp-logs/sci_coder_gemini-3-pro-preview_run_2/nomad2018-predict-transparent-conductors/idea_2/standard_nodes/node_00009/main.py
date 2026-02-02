import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import CrystalGraphConvNet
from library.train import train_one_epoch, evaluate


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Use load_cached_data=True to speed up if available
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    model = CrystalGraphConvNet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.MSELoss()

    # 5. Training Loop
    best_val_metric = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_t = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        # evaluate returns (avg_loss, metrics_dict)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, scaler)

        # Metric for early stopping: RMSLE Mean
        current_metric = val_metrics["rmsle_mean"]

        # Scheduler step
        scheduler.step(val_loss)

        # Checkpointing
        if current_metric < best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # 6. Final Validation Assessment
    print("Loading best model for final assessment...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Re-evaluate to get precise final metric
    _, final_metrics = evaluate(model, val_loader, criterion, device, scaler)
    final_metric_value = final_metrics["rmsle_mean"]

    print(f"Final Validation Metric: {final_metric_value}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Get predictions on validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            val_preds.append(outputs.cpu())
            val_targets.append(batch.y.cpu())

    val_preds_tensor = torch.cat(val_preds, dim=0)
    val_targets_tensor = torch.cat(val_targets, dim=0)

    # Inverse transform
    val_preds_orig = scaler.inverse_transform(val_preds_tensor)
    val_targets_orig = scaler.inverse_transform(val_targets_tensor)

    # Clamp
    val_preds_orig = torch.clamp(val_preds_orig, min=0.0)
    val_targets_orig = torch.clamp(val_targets_orig, min=0.0)

    # Calculate error magnitude per sample (Mean Squared Log Error per sample)
    # log(p+1) - log(t+1)
    log_diff = torch.log1p(val_preds_orig) - torch.log1p(val_targets_orig)
    # Average squared log error across the 2 targets for each sample
    sample_errors = torch.mean(log_diff**2, dim=1).numpy()

    # Load validation metadata to get features
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure lengths match
    if len(val_meta_df) != len(sample_errors):
        print("Warning: Mismatch in validation set size for analysis.")
    else:
        # Select numerical features for correlation
        # We exclude id, file_path, and targets
        feature_cols = [
            col
            for col in val_meta_df.columns
            if col
            not in [
                "id",
                "file_path",
                "formation_energy_ev_natom",
                "bandgap_energy_ev",
                "spacegroup",
            ]
            and np.issubdtype(val_meta_df[col].dtype, np.number)
        ]

        # Add error column
        analysis_df = val_meta_df[feature_cols].copy()
        analysis_df["error_magnitude"] = sample_errors

        # Compute correlations
        correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

        print("Correlation between Error Magnitude (MSLE) and Features:")
        print(correlations.sort_values(key=abs, ascending=False).to_string())

    # 8. Conditional Submission
    THRESHOLD = 0.053007537912991315

    if final_metric_value < THRESHOLD:
        print(f"\nMetric {final_metric_value} < {THRESHOLD}. Generating submission...")

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                test_preds.append(outputs.cpu())

        # Concatenate
        test_preds_tensor = torch.cat(test_preds, dim=0)

        # Inverse transform
        test_preds_orig = scaler.inverse_transform(test_preds_tensor)
        test_preds_orig = torch.clamp(test_preds_orig, min=0.0)
        test_preds_np = test_preds_orig.numpy()

        # Create submission dataframe
        test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)
        submission_df = pd.DataFrame(
            {
                "id": test_meta_df["id"],
                "formation_energy_ev_natom": test_preds_np[:, 0],
                "bandgap_energy_ev": test_preds_np[:, 1],
            }
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric_value} >= {THRESHOLD}. Submission NOT generated."
        )


if __name__ == "__main__":
    main()
