import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library functions
from library.dataset import load_data
from library.utils import set_seed, get_device, IcebergDataset, predict_test, BHA_ResNet
from library.train import fit_fold


def main():
    # 1. Environment Setup
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load processed numpy arrays. We use ./working/cache as the base directory.
    # We load the full dataset (dataset_size=None) as the dataset is small (~1600 samples).
    print("Loading and processing data...")
    X_all, y_all, angle_all, X_test, ids_test, angle_test = load_data(
        base_dir="./working/cache", load_cached_data=False, dataset_size=None
    )

    # Load Metadata to reconstruct the specific Train/Val split
    print("Loading metadata...")
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")

    # Extract indices to partition the loaded arrays
    train_indices = train_meta["original_index"].values
    val_indices = val_meta["original_index"].values

    # Create the Train set (for CV) and the Hold-out Val set
    X_train_cv = X_all[train_indices]
    y_train_cv = y_all[train_indices]
    angle_train_cv = angle_all[train_indices]

    X_holdout = X_all[val_indices]
    y_holdout = y_all[val_indices]
    angle_holdout = angle_all[val_indices]

    print(f"Training Set Size (for CV): {len(X_train_cv)}")
    print(f"Hold-out Validation Set Size: {len(X_holdout)}")

    # 3. Cross-Validation Training
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Define augmentation for training
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    # Training Hyperparameters
    # 50 epochs is sufficient for convergence on this small dataset while keeping runtime low.
    EPOCHS = 50
    BATCH_SIZE = 32
    PATIENCE = 10
    LR = 1e-3
    CHECKPOINT_DIR = "./working/checkpoints"

    model_paths = []

    print(f"\nStarting {n_folds}-Fold Cross-Validation...")

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_cv, y_train_cv)):
        # Inner split for monitoring training progress
        X_tr, X_val = X_train_cv[tr_idx], X_train_cv[val_idx]
        y_tr, y_val = y_train_cv[tr_idx], y_train_cv[val_idx]
        angle_tr, angle_val = angle_train_cv[tr_idx], angle_train_cv[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, angle_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_val, angle_val, y_val, transform=None)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Train the fold
        best_model_path, best_loss = fit_fold(
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            patience=PATIENCE,
            lr=LR,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        model_paths.append(best_model_path)

    # 4. Hold-out Evaluation
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    # Load all trained models for the ensemble
    models = []
    for path in model_paths:
        model = BHA_ResNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    # Create Hold-out Loader
    holdout_ds = IcebergDataset(X_holdout, angle_holdout, y_holdout, transform=None)
    holdout_loader = DataLoader(
        holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Generate predictions (averaging across ensemble)
    holdout_preds = predict_test(models, holdout_loader, device)

    # Compute Metric
    final_metric = log_loss(y_holdout, holdout_preds)
    print("Final Validation Metric:", final_metric)

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_holdout - holdout_preds)

    # Calculate simple image statistics for correlation analysis
    # X_holdout is (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_means = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angle_holdout,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    # Compute correlations
    corr_angle = analysis_df["error"].corr(analysis_df["inc_angle"])
    corr_b1 = analysis_df["error"].corr(analysis_df["b1_mean"])
    corr_b2 = analysis_df["error"].corr(analysis_df["b2_mean"])

    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.4f}")
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2:.4f}")

    # 6. Submission
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Create Test Loader
        test_ds = IcebergDataset(X_test, angle_test, y=None, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Generate Test Predictions
        test_preds = predict_test(models, test_loader, device)

        # Save Submission
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        sub_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds})
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
