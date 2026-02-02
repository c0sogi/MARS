import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model as model_lib
from library import engine


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Metadata and Data
    # Load metadata
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    val_meta = pd.read_csv(config.VAL_META_PATH)

    # Combine for CV
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load cached data arrays
    print("Loading training data...")
    data_dict = dataset.process_and_cache_data("train", load_cached_data=True)

    # Calculate global mean angle for imputation
    valid_angles = data_dict["angle"][~np.isnan(data_dict["angle"])]
    angle_fill_value = float(np.mean(valid_angles))
    print(f"Angle fill value: {angle_fill_value:.4f}")

    # 3. Cross-Validation Initialization
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Array to store Out-Of-Fold predictions
    oof_preds = np.zeros(len(full_meta))
    # Array to store targets for metric calculation
    oof_targets = np.zeros(len(full_meta))

    # Store model paths for ensemble inference
    model_paths = []

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_meta, full_meta["is_iceberg"])
    ):
        print(f"\n--- Fold {fold + 1}/{config.NUM_FOLDS} ---")

        # Split metadata
        train_fold_meta = full_meta.iloc[train_idx].copy()
        val_fold_meta = full_meta.iloc[val_idx].copy()

        # Create Datasets
        train_dataset = dataset.IcebergDataset(
            metadata_df=train_fold_meta,
            full_data_dict=data_dict,
            transform=dataset.get_transforms("train"),
            angle_fill_value=angle_fill_value,
            mode="train",
        )

        val_dataset = dataset.IcebergDataset(
            metadata_df=val_fold_meta,
            full_data_dict=data_dict,
            transform=dataset.get_transforms("val"),
            angle_fill_value=angle_fill_value,
            mode="val",
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = model_lib.SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.BCELoss()

        # Training
        best_val_loss = float("inf")
        checkpoint_dir = os.path.join(config.WORKING_DIR, f"fold_{fold}")

        for epoch in range(config.NUM_EPOCHS):
            train_loss = engine.train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = engine.evaluate(model, val_loader, criterion, device)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Save checkpoint
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_score": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                checkpoint_dir,
            )

            # Simple early stopping check (optional, but good for speed)
            # Given the fixed epoch count is small (50), we run full duration
            # or rely on the save_checkpoint logic to keep the best.

        # Load best model for OOF prediction
        best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
        model_paths.append(best_model_path)
        utils.load_checkpoint(best_model_path, model, device=device)

        # Generate OOF predictions (No TTA for validation speed/standardization)
        model.eval()
        fold_preds = []
        fold_targets = []
        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = model(images, angles)
                fold_preds.extend(outputs.cpu().numpy().flatten())
                fold_targets.extend(targets.numpy().flatten())

        oof_preds[val_idx] = fold_preds
        oof_targets[val_idx] = fold_targets

    # 5. Validation Assessment
    final_metric = log_loss(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(oof_targets - oof_preds)

    # Extract features for correlation analysis
    # We need to access the raw data corresponding to the full_meta indices
    indices = full_meta["original_index"].values

    # Get Incidence Angles (filled)
    angles = data_dict["angle"][indices]
    mask = np.isnan(angles)
    angles[mask] = angle_fill_value

    # Get Band Means
    # X shape is (N, 75, 75, 3). Band 0 is HH, Band 1 is HV.
    X_subset = data_dict["X"][indices]
    b1_means = np.mean(X_subset[:, :, :, 0], axis=(1, 2))
    b2_means = np.mean(X_subset[:, :, :, 1], axis=(1, 2))

    # Compute correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "inc_angle": angles, "b1_mean": b1_means, "b2_mean": b2_means}
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.18145903282502943
    if final_metric < THRESHOLD:
        print("\nMetric condition met. Generating submission...")

        # Load Test Data
        print("Loading test data...")
        test_data_dict = dataset.process_and_cache_data("test", load_cached_data=True)
        test_meta = pd.read_csv(config.TEST_META_PATH)

        test_dataset = dataset.IcebergDataset(
            metadata_df=test_meta,
            full_data_dict=test_data_dict,
            transform=dataset.get_transforms("test"),  # No augs, TTA handled in engine
            angle_fill_value=angle_fill_value,
            mode="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Inference
        ensemble_preds = {}
        # Initialize dictionary with 0s
        for uid in test_meta["id"].values:
            ensemble_preds[uid] = 0.0

        for fold_idx, model_path in enumerate(model_paths):
            print(f"Inference with model fold {fold_idx + 1}...")
            model = model_lib.SimpleCNN().to(device)
            utils.load_checkpoint(model_path, model, device=device)

            # Predict with TTA
            preds = engine.predict_with_tta(model, test_loader, device)

            # Accumulate
            for uid, prob in preds.items():
                ensemble_preds[uid] += prob

        # Average
        for uid in ensemble_preds:
            ensemble_preds[uid] /= config.NUM_FOLDS

        # Save
        engine.save_submission(ensemble_preds, config.SUBMISSION_PATH)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
