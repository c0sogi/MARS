import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_and_process_data, IcebergDataset
from library.model import DSN_CNN
from library.trainer import train_one_epoch, validate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load cached data (or process if not available)
    # The loader returns separate train/val/test splits based on metadata
    (train_data, val_data, test_data) = load_and_process_data(load_cached_data=True)

    X_train_part, y_train_part, angle_train_part = train_data
    X_val_part, y_val_part, angle_val_part = val_data
    X_test, ids_test, angle_test = test_data

    # Cite solution_lesson_node_00044: The Pessimism of Out-Of-Fold (OOF) Metrics.
    # We use a Hold-Out Ensemble strategy to match the threshold evaluation methodology.
    # X_train_part is used for 5-Fold CV Training.
    # X_val_part is used as the fixed Hold-Out set for Ensemble Evaluation.

    print(f"Training Samples: {len(y_train_part)}")
    print(f"Hold-Out Validation Samples: {len(y_val_part)}")
    print(f"Test Samples: {len(ids_test)}")

    # Prepare Hold-Out Validation Loader
    val_ds_holdout = IcebergDataset(
        X_val_part, angle_val_part, y_val_part, transform=None
    )
    val_loader_holdout = DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Prepare Test Loader
    test_ds = IcebergDataset(X_test, angle_test, y=None, transform=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. K-Fold Training on Training Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store ensemble predictions
    val_preds_accum = np.zeros(len(X_val_part))
    test_preds_accum = np.zeros(len(X_test))

    # Augmentation for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    print("\nStarting 5-Fold Cross-Validation on Training Split...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_part, y_train_part)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split data (Inner CV split)
        X_tr, y_tr, ang_tr = (
            X_train_part[train_idx],
            y_train_part[train_idx],
            angle_train_part[train_idx],
        )
        X_va, y_va, ang_va = (
            X_train_part[val_idx],
            y_train_part[val_idx],
            angle_train_part[val_idx],
        )

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None)

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
        model = DSN_CNN().to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    break

        print(f"Fold {fold+1} Best Inner Val Loss: {best_val_loss:.6f}")

        # Load best model for inference
        model.load_state_dict(best_model_state)
        model.eval()

        # Generate Predictions on Hold-Out Validation Set
        fold_val_preds = predict(model, val_loader_holdout, device)
        val_preds_accum += fold_val_preds

        # Generate Test Predictions
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds

        # Cleanup
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Evaluation (Ensemble)
    avg_val_preds = val_preds_accum / Config.N_FOLDS
    final_log_loss = log_loss(y_val_part, avg_val_preds)
    print(f"\nFinal Validation Metric (Hold-Out Ensemble): {final_log_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error per sample
    errors = np.abs(y_val_part - avg_val_preds)

    # Calculate image statistics for correlation
    # X_val_part shape is (N, 75, 75, 3).
    b1_mean = np.mean(X_val_part[:, :, :, 0], axis=(1, 2))
    b1_std = np.std(X_val_part[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(X_val_part[:, :, :, 1], axis=(1, 2))
    b2_std = np.std(X_val_part[:, :, :, 1], axis=(1, 2))

    features = {
        "Incidence Angle": angle_val_part,
        "Band 1 Mean": b1_mean,
        "Band 1 Std": b1_std,
        "Band 2 Mean": b2_mean,
        "Band 2 Std": b2_std,
    }

    print("Correlation between Absolute Error and Input Features:")
    for name, values in features.items():
        corr, _ = pearsonr(errors, values)
        print(f"  {name}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.18145903282502943

    if final_log_loss < THRESHOLD:
        print(f"\nMetric {final_log_loss} is lower than threshold {THRESHOLD}.")
        print("Generating submission file...")

        # Average predictions across folds
        avg_test_preds = test_preds_accum / Config.N_FOLDS

        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(f"\nMetric {final_log_loss} did not meet threshold {THRESHOLD}.")
        print("Submission skipped.")


if __name__ == "__main__":
    main()
