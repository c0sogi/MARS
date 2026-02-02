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

    print("Loading and processing data...")
    # 2. Data Loading
    # load_and_process_data returns splits based on metadata/train.csv and metadata/val.csv
    data = load_and_process_data(load_cached_data=True)

    X_train_split = data["X_train"]
    y_train_split = data["y_train"]
    angles_train_split = data["angles_train"]

    X_val_split = data["X_val"]
    y_val_split = data["y_val"]
    angles_val_split = data["angles_val"]

    # Combine for 5-Fold CV strategy (as per Idea)
    # We concatenate [Train, Val] so we can perform CV on the full available labeled data.
    # We track the split point to extract hold-out metrics later.
    X_full = np.concatenate([X_train_split, X_val_split], axis=0)
    y_full = np.concatenate([y_train_split, y_val_split], axis=0)
    angles_full = np.concatenate([angles_train_split, angles_val_split], axis=0)

    # Test Data
    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    print(f"Total labeled samples: {len(X_full)}")
    print(f"Test samples: {len(X_test)}")

    # 3. 5-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF predictions (to calculate metric) and Test predictions (for submission)
    oof_preds = np.zeros(len(y_full))
    test_preds_sum = np.zeros((len(X_test), 1))

    # Augmentation for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Test Loader (Fixed)
    test_ds = IcebergDataset(X_test, angles_test, y=None, transform=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    print("\nStarting 5-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold} ---")

        # Split Data
        X_tr, X_v = X_full[train_idx], X_full[val_idx]
        y_tr, y_v = y_full[train_idx], y_full[val_idx]
        ang_tr, ang_v = angles_full[train_idx], angles_full[val_idx]

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

        # Train (run_fold handles training loop and saving best model)
        run_fold(fold, train_loader, val_loader, device)

        # Load Best Model for Inference
        model = SimpleCNN().to(device)
        ckpt_path = Config.get_checkpoint_path(fold)
        load_checkpoint(ckpt_path, model, device=Config.DEVICE)

        # Predict on Validation Fold (OOF)
        fold_val_preds = predict(model, val_loader, device)
        oof_preds[val_idx] = fold_val_preds.flatten()

        # Predict on Test Set
        fold_test_preds = predict(model, test_loader, device)
        test_preds_sum += fold_test_preds

        # Cleanup to save memory
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 4. Evaluation
    # We must report the metric on the "hold-out validation set".
    # Since we concatenated [Train, Val], the hold-out set corresponds to the indices starting from len(X_train_split).
    val_start_idx = len(X_train_split)
    holdout_preds = oof_preds[val_start_idx:]
    holdout_targets = y_full[val_start_idx:]

    # Calculate Metric
    final_metric = log_loss(holdout_targets, holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(holdout_targets - holdout_preds)

    # Extract features for correlation analysis from the holdout set (X_val_split)
    # X is (N, 3, 75, 75). Band 1 is index 0, Band 2 is index 1.
    b1_mean = np.mean(X_val_split[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_val_split[:, 1, :, :], axis=(1, 2))
    inc_angles = angles_val_split

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
