import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

# Import from provided library modules
from library.config import (
    CHECKPOINT_DIR,
    CACHE_DIR,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TARGET_COLS,
    RANDOM_SEED,
)
from library.utils import set_seed, get_scaler
from library.data import get_train_val_datasets, get_test_dataset
from library.model import MH_RA_CGN
from library.train import train_epoch, validate


def main():
    # 1. Setup and Configuration
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Hyperparameters for fast baseline
    BATCH_SIZE = 48
    NUM_EPOCHS = 50  # Limited epochs for speed while ensuring convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # 2. Data Loading
    print("Loading datasets...")
    # Load cached datasets if available to save time
    train_dataset, val_dataset = get_train_val_datasets(load_cached=True)

    # 3. Scaler Preparation
    print("Preparing target scaler...")
    # We fit the scaler on the training targets
    all_train_targets = torch.cat([d.y for d in train_dataset], dim=0)
    scaler_path = os.path.join(CACHE_DIR, "target_scaler.npz")
    scaler = get_scaler(all_train_targets, scaler_path, load_cached_data=True)

    # Move scaler statistics to device
    if isinstance(scaler.mean, torch.Tensor):
        scaler.mean = scaler.mean.to(device)
        scaler.std = scaler.std.to(device)

    # 4. Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Model Initialization
    model = MH_RA_CGN().to(device)

    # 6. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=8, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # 7. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_val_rmsle = float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_runfile.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_mse, val_rmsle = validate(model, val_loader, criterion, scaler, device)

        scheduler.step(val_mse)

        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            torch.save(model.state_dict(), best_model_path)

        # Print progress periodically
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Val RMSLE: {val_rmsle:.4f}"
            )

    print("Training complete.")

    # 8. Final Validation & Failure Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Compute Final Metric on Validation Set
    _, final_metric = validate(model, val_loader, criterion, scaler, device)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("Performing Failure Analysis on Validation Set...")
    val_metadata = pd.read_csv(VAL_METADATA_PATH)

    # Collect predictions and targets
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            preds_raw = scaler.inverse_transform(out)
            val_preds_list.append(preds_raw.cpu().numpy())
            val_targets_list.append(data.y.cpu().numpy())

    val_preds_arr = np.concatenate(val_preds_list, axis=0)
    val_targets_arr = np.concatenate(val_targets_list, axis=0)

    # Calculate mean absolute error per sample (averaged over the two targets)
    # This gives a scalar error value for each material in the validation set
    sample_errors = np.mean(np.abs(val_preds_arr - val_targets_arr), axis=1)

    # Assign errors to metadata
    # Note: val_loader is not shuffled, so order matches val_metadata
    if len(val_metadata) == len(sample_errors):
        val_metadata["error"] = sample_errors

        # Calculate correlations with numeric features
        numeric_cols = val_metadata.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "error"]
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        correlations = (
            val_metadata[feature_cols]
            .corrwith(val_metadata["error"])
            .sort_values(key=abs, ascending=False)
        )
        print("Top 5 features correlated with prediction error:")
        print(correlations.head(5))
    else:
        print(
            "Warning: Mismatch between validation set size and metadata size. Skipping correlation analysis."
        )

    # 9. Submission Generation
    THRESHOLD = 0.049412816762924194

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = get_test_dataset(load_cached=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_preds_list = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                preds_raw = scaler.inverse_transform(out)
                test_preds_list.append(preds_raw.cpu().numpy())

        test_preds_arr = np.concatenate(test_preds_list, axis=0)

        # Create Submission DataFrame
        test_meta = pd.read_csv(TEST_METADATA_PATH)

        # Ensure lengths match
        if len(test_meta) != len(test_preds_arr):
            print("Error: Test metadata length does not match predictions length.")
        else:
            submission_df = pd.DataFrame(
                {
                    "id": test_meta["id"],
                    "formation_energy_ev_natom": test_preds_arr[:, 0],
                    "bandgap_energy_ev": test_preds_arr[:, 1],
                }
            )

            submission_path = os.path.join("submission", "submission.csv")
            # Ensure directory exists (handled by config but good practice)
            os.makedirs(os.path.dirname(submission_path), exist_ok=True)

            submission_df.to_csv(submission_path, index=False)
            print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric {final_metric} did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
