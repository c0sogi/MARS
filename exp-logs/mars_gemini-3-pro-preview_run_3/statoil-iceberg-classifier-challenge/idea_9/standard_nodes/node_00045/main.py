import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from provided library files
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import load_and_process_data, IcebergDataset
from library.model import SimpleCNN
from library.train import run_fold, predict


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust Config for fast baseline execution
    Config.EPOCHS = 30  # Increased slightly for SimpleCNN convergence

    print("Loading and processing data...")
    # 2. Data Loading
    data = load_and_process_data(load_cached_data=True)

    # Cite Lesson 00044: Do not directly compare Out-Of-Fold (OOF) metrics with Hold-Out Ensemble metrics.
    # We strictly separate Train and Val to perform Ensemble Evaluation on Val.
    X_train = data["X_train"]
    y_train = data["y_train"]
    angles_train = data["angles_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    angles_val = data["angles_val"]

    # Test Data
    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    print(f"Training samples: {len(X_train)}")
    print(f"Hold-out Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 3. 5-Fold Cross-Validation on TRAIN set
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for Ensemble Predictions
    val_preds_sum = np.zeros((len(X_val), 1))
    test_preds_sum = np.zeros((len(X_test), 1))

    # Augmentation for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Fixed Loaders for Evaluation
    val_ds = IcebergDataset(X_val, angles_val, y_val, transform=None)
    val_loader_fixed = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_ds = IcebergDataset(X_test, angles_test, y=None, transform=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    print("\nStarting 5-Fold Cross-Validation (Ensemble Training)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold} ---")

        # Split Data (Internal CV on Training Set)
        X_tr, X_v = X_train[train_idx], X_train[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]
        ang_tr, ang_v = angles_train[train_idx], angles_train[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_v, ang_v, y_v, transform=None)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Train
        run_fold(fold, train_loader, val_loader, device)

        # Load Best Model for Inference
        model = SimpleCNN().to(device)
        ckpt_path = Config.get_checkpoint_path(fold)
        load_checkpoint(ckpt_path, model, device=Config.DEVICE)

        # Predict on Fixed Hold-out Validation Set (Ensembling)
        # Cite Lesson 00016: Variance Reduction via Cross-Validation Ensembling
        fold_val_preds = predict(model, val_loader_fixed, device)
        val_preds_sum += fold_val_preds

        # Predict on Test Set
        fold_test_preds = predict(model, test_loader, device)
        test_preds_sum += fold_test_preds

        # Cleanup
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 4. Evaluation (Ensemble)
    avg_val_preds = val_preds_sum / Config.N_FOLDS

    # Calculate Metric
    final_metric = log_loss(y_val, avg_val_preds)
    print(f"Final Validation Metric (Ensemble on Hold-out): {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val - avg_val_preds.flatten())

    # Extract features for correlation analysis from the holdout set (X_val)
    # X is (N, 3, 75, 75). Band 1 is index 0, Band 2 is index 1.
    b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))
    inc_angles = angles_val

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_angles,
            "b1_mean": b1_mean,
            "b2_mean": b2_mean,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.18145903282502943
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Average predictions across 5 folds
        avg_test_preds = test_preds_sum / Config.N_FOLDS

        submission_df = pd.DataFrame(
            {"id": ids_test, "is_iceberg": avg_test_preds.flatten()}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
