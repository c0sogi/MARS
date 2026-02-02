import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_loader import process_and_cache_data, get_global_stats, IcebergDataset
from library.model import CRWBN
from library.train_eval import Trainer


def main():
    # 1. Initialize Environment
    set_seed(Config.SEED)

    # Override Config for fast baseline execution
    # Reducing epochs to 50 to ensure the run completes quickly while allowing sufficient convergence
    Config.EPOCHS = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load and Prepare Data
    # load_cached_data=True allows using pre-processed .npz files if they exist
    data = process_and_cache_data(load_cached_data=True)
    X_full = data["X_train_full"]
    y_full = data["y_train_full"]
    angles_full = data["angles_train_full"]
    ids_full = data["ids_train_full"]

    # Compute global statistics for normalization (Min-Max scaling)
    stats = get_global_stats(X_full)

    # 3. Stratified Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Array to store Out-Of-Fold (OOF) predictions for global metric calculation
    oof_preds = np.zeros(len(y_full))

    print(f"Starting Stratified {Config.NUM_FOLDS}-Fold CV on {len(y_full)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold} ---")

        # Split Data
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        angles_train, angles_val = angles_full[train_idx], angles_full[val_idx]
        ids_train, ids_val = ids_full[train_idx], ids_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_train, angles_train, y_train, ids_train, transform=True, stats=stats
        )
        val_ds = IcebergDataset(
            X_val, angles_val, y_val, ids_val, transform=False, stats=stats
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize Model
        model = CRWBN().to(device)

        # Setup Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Initialize Trainer
        # Logger is set to None to use standard print output
        trainer = Trainer(model, device, criterion, optimizer, scheduler, logger=None)

        # Train the model
        trainer.fit(train_loader, val_loader, Config.EPOCHS, Config.PATIENCE, fold)

        # Save the best model for this fold
        save_path = Config.MODEL_PATH_TEMPLATE.format(fold)
        torch.save(model.state_dict(), save_path)

        # Validation Inference
        # The trainer loads the best model state automatically at the end of fit()
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for inputs, angles, _ in val_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        # Store predictions
        oof_preds[val_idx] = np.array(fold_preds)

    # 4. Compute Metrics and Analysis
    # Clip predictions to strictly [epsilon, 1-epsilon] to ensure log_loss stability
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_full, oof_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(y_full - oof_preds)

    # Extract simple features for correlation analysis
    # Calculate mean intensity for Band 1 and Band 2
    b1_mean = np.mean(X_full[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_full[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_full,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
            "target": y_full,
        }
    )

    # Calculate correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission Generation
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        X_test = data["X_test"]
        angles_test = data["angles_test"]
        ids_test = data["ids_test"]

        # Create Test Dataset
        test_ds = IcebergDataset(
            X_test, angles_test, labels=None, ids=ids_test, transform=False, stats=stats
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        ensemble_preds = []

        # Iterate over all folds for ensemble prediction
        for fold in range(Config.NUM_FOLDS):
            model_path = Config.MODEL_PATH_TEMPLATE.format(fold)

            # Load Model
            model = CRWBN().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for inputs, angles, _ in test_loader:
                    inputs = inputs.to(device)
                    angles = angles.to(device)
                    outputs = model(inputs, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            ensemble_preds.append(np.array(fold_preds))

        # Average predictions across folds
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

        # Save to CSV
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
