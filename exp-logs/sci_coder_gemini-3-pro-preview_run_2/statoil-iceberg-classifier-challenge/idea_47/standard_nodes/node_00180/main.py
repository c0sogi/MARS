import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.data_handling import load_data, IcebergDataset, get_global_stats
from library.model import PCWBN
from library.training import train_one_epoch, validate, EarlyStopping


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Load Data
    # Using load_cached_data=True to utilize pre-processed data
    data = load_data(load_cached_data=True)

    # Combine Train and Val for full Stratified K-Fold
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    inc_full = np.concatenate([data["inc_train"], data["inc_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    # Compute global stats for normalization (using full training set statistics)
    stats = get_global_stats(X_full, inc_full)

    # 3. Training Configuration
    n_folds = Config.NUM_FOLDS
    # Limit epochs for fast baseline execution while ensuring convergence on small data
    epochs = 50
    batch_size = Config.BATCH_SIZE

    # Storage for Out-Of-Fold (OOF) predictions
    oof_preds = np.zeros(len(y_full))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    print(f"Starting {n_folds}-Fold Cross-Validation on {len(y_full)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n{'='*10} Fold {fold+1}/{n_folds} {'='*10}")

        # Split Data
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        inc_train, inc_val = inc_full[train_idx], inc_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        # Datasets & Loaders
        train_ds = IcebergDataset(X_train, inc_train, y_train, stats, transform=True)
        val_ds = IcebergDataset(X_val, inc_val, y_val, stats, transform=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = PCWBN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        # Early Stopping
        model_save_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        early_stopping = EarlyStopping(patience=Config.PATIENCE, path=model_save_path)

        # Training Loop
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            # Silent progress, only print occasional status or if needed
            # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if early_stopping.early_stop:
                break

        # Load best model for inference
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        # Generate predictions for validation set
        fold_preds = []
        with torch.no_grad():
            for inputs, incs, _ in val_loader:
                inputs = inputs.to(device)
                incs = incs.to(device)
                outputs = model(inputs, incs)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0).flatten()
        oof_preds[val_idx] = fold_preds

        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Evaluation & Failure Analysis
    final_metric = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_full - oof_preds)

    # Compute simple features for correlation analysis
    # X_full is (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_means = np.mean(X_full[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X_full[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_full,
            "band_1_mean": b1_means,
            "band_2_mean": b2_means,
            "target": y_full,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # 5. Submission
    threshold = 0.14772333549413377
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        X_test = data["X_test"]
        inc_test = data["inc_test"]

        # Create Test Loader
        y_test_dummy = np.zeros(len(X_test))
        test_ds = IcebergDataset(X_test, inc_test, y_test_dummy, stats, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        ensemble_preds = np.zeros((len(X_test), 1))

        for fold in range(n_folds):
            model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
            model = PCWBN().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for inputs, incs, _ in test_loader:
                    inputs = inputs.to(device)
                    incs = incs.to(device)
                    outputs = model(inputs, incs)
                    probs = torch.sigmoid(outputs)
                    fold_preds.append(probs.cpu().numpy())

            ensemble_preds += np.concatenate(fold_preds, axis=0)

        avg_preds = ensemble_preds / n_folds

        # Save Submission
        test_meta = pd.read_csv(Config.TEST_META_PATH)
        submission = pd.DataFrame(
            {"id": test_meta["id"], "is_iceberg": avg_preds.flatten()}
        )

        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
