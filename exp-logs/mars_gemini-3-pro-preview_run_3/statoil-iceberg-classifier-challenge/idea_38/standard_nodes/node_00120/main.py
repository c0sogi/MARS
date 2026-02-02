import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library import utils
from library import dataset
from library import model as model_lib
from library import train as train_lib


def run():
    # 1. Setup
    utils.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("Preparing data...")
    # Load cached data
    data = dataset.prepare_data(load_cached_data=True)

    # Combine Train and Val for 5-Fold CV
    # The library splits are 80/20, but we want to perform CV on the whole labeled set
    X_train_part = data["train"]["X"]
    y_train_part = data["train"]["y"]
    angle_train_part = data["train"]["angle"]

    X_val_part = data["val"]["X"]
    y_val_part = data["val"]["y"]
    angle_val_part = data["val"]["angle"]

    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    angle_full = np.concatenate([angle_train_part, angle_val_part], axis=0)

    # Test Data
    X_test = data["test"]["X"]
    ids_test = data["test"]["ids"]
    angle_test = data["test"]["angle"]

    print(f"Full Training Data Shape: {X_full.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(y_full))
    test_preds_accum = np.zeros(len(ids_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n{'='*10} Starting Fold {fold} {'='*10}")

        # Split Data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        ang_tr, ang_va = angle_full[train_idx], angle_full[val_idx]

        # Create Datasets
        train_ds = dataset.IcebergDataset(
            X_tr, ang_tr, labels=y_tr, transform=dataset.get_transforms("train")
        )
        val_ds = dataset.IcebergDataset(
            X_va, ang_va, labels=y_va, transform=dataset.get_transforms("test")
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = model_lib.DualPolarityDropBlockSECNN().to(device)

        # Optimizer & Loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = train_lib.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                Config.NUM_EPOCHS,
            )

            # Validate
            val_loss, val_acc = train_lib.validate(model, val_loader, criterion, device)

            # Checkpoint
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                utils.save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "fold": fold,
                    },
                    is_best=True,
                    fold=fold,
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Fold {fold} Best Loss: {best_loss:.5f}")

        # 4. Inference for OOF and Test
        # Load best model
        utils.load_checkpoint(model, None, fold, load_best=True, device=device)
        model.eval()

        # OOF Inference
        val_preds_fold = []
        with torch.no_grad():
            for inputs, angles, _ in val_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                probs = torch.sigmoid(outputs)
                val_preds_fold.append(probs.cpu().numpy())

        oof_preds[val_idx] = np.concatenate(val_preds_fold).flatten()

        # Test Inference
        test_ds = dataset.IcebergDataset(
            X_test, angle_test, ids=ids_test, transform=dataset.get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds_fold = []
        with torch.no_grad():
            for inputs, angles, _ in test_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                probs = torch.sigmoid(outputs)
                test_preds_fold.append(probs.cpu().numpy())

        test_preds_accum += np.concatenate(test_preds_fold).flatten()

    # 5. Final Metrics and Analysis
    final_metric = log_loss(y_full, oof_preds)
    # Print the full precision metric as requested
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis on Validation Set:")
    errors = np.abs(y_full - oof_preds)

    # Compute features for correlation
    # Band 1 Mean (Channel 0)
    b1_mean = np.mean(X_full[:, 0, :, :], axis=(1, 2))
    # Band 2 Mean (Channel 1)
    b2_mean = np.mean(X_full[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Incidence Angle": angle_full,
            "Band 1 Mean": b1_mean,
            "Band 2 Mean": b2_mean,
        }
    )

    correlations = analysis_df.corr()["Error"].drop("Error")
    print("Correlation with Error Magnitude:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.1806015565870406
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.5f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        avg_test_preds = test_preds_accum / Config.NUM_FOLDS

        submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric:.5f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
