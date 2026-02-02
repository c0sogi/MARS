import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided libraries
from library.config import Config
from library.data_utils import load_data, IcebergDataset, set_seed
from library.model_utils import CSNet
from library.train_utils import train_fold


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Data
    # load_data returns dictionaries for train (from train.csv), val (from val.csv), and test
    print("Loading data...")
    train_data, val_data, test_data = load_data(Config, load_cached_data=True)

    # Prepare data arrays for Cross-Validation
    X_train_full = train_data["images"]
    ang_train_full = train_data["angles"]
    y_train_full = train_data["labels"]

    # 3. Stratified 5-Fold Cross-Validation Training
    # We split the 'train.csv' data into 5 folds to train 5 ensemble models.
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    print(f"\nStarting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, inner_val_idx) in enumerate(
        skf.split(X_train_full, y_train_full)
    ):
        print(f"\n--- Fold {fold} ---")

        # Split data for this fold
        X_fold_train = X_train_full[train_idx]
        ang_fold_train = ang_train_full[train_idx]
        y_fold_train = y_train_full[train_idx]

        X_fold_val = X_train_full[inner_val_idx]
        ang_fold_val = ang_train_full[inner_val_idx]
        y_fold_val = y_train_full[inner_val_idx]

        # Create Datasets
        # Apply augmentation (transform=True) only to the training set of the fold
        train_ds = IcebergDataset(
            X_fold_train, ang_fold_train, y_fold_train, transform=True
        )
        val_ds = IcebergDataset(X_fold_val, ang_fold_val, y_fold_val, transform=False)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False,
        )

        # Initialize Model, Optimizer, Scheduler
        model = CSNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Train the fold
        # train_fold saves the best model to disk and returns the model instance
        _, best_loss = train_fold(
            model, train_loader, val_loader, optimizer, scheduler, device, Config, fold
        )

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Evaluation on Hold-out Validation Set (Ensemble)
    print("\n--- Evaluation on Hold-out Validation Set ---")

    # Dataset for the fixed hold-out validation set (from val.csv)
    holdout_ds = IcebergDataset(
        val_data["images"], val_data["angles"], val_data["labels"], transform=False
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    ensemble_preds = []

    # Iterate through all trained fold models
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.MODEL_DIR, f"csnet_fold_{fold}.pth")

        # Load Model
        model = CSNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in holdout_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.append(probs)

        # Concatenate predictions for this fold
        ensemble_preds.append(np.concatenate(fold_preds))

        del model
        torch.cuda.empty_cache()

    # Average predictions across folds
    avg_preds = np.mean(ensemble_preds, axis=0).flatten()

    # Calculate Metric
    final_log_loss = log_loss(val_data["labels"], avg_preds)
    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_data["labels"] - avg_preds)

    # Compute simple signal strength proxy (mean of Band 1) for correlation analysis
    # images shape: (N, 6, 75, 75). Band 0 is HH.
    signal_strength = np.mean(val_data["images"][:, 0, :, :], axis=(1, 2))

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": val_data["angles"],
            "signal_strength": signal_strength,
        }
    )

    # Calculate correlations
    corr_angle = df_analysis["error"].corr(df_analysis["inc_angle"])
    corr_signal = df_analysis["error"].corr(df_analysis["signal_strength"])

    print(f"Correlation (Error vs Inc Angle): {corr_angle}")
    print(f"Correlation (Error vs Signal Strength): {corr_signal}")

    # 6. Submission
    THRESHOLD = 0.16676861786296204

    if final_log_loss < THRESHOLD:
        print("\nMetric threshold met. Generating submission...")

        test_ds = IcebergDataset(
            test_data["images"], test_data["angles"], labels=None, transform=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False,
        )

        test_ensemble_preds = []

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.MODEL_DIR, f"csnet_fold_{fold}.pth")

            model = CSNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    fold_preds.append(probs)

            test_ensemble_preds.append(np.concatenate(fold_preds))
            del model
            torch.cuda.empty_cache()

        # Average predictions
        avg_test_preds = np.mean(test_ensemble_preds, axis=0).flatten()

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"id": test_data["ids"], "is_iceberg": avg_test_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_log_loss} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
