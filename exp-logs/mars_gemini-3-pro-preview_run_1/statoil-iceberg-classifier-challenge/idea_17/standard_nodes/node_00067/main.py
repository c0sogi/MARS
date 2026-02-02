import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader

# Import provided library modules
from library.configuration import Config
from library.utilities import set_seed
from library.data_loader import get_data_arrays, IcebergDataset
from library.architecture import IcebergResNet
from library.optimization import get_optimizer, get_scheduler, SoftBCELoss
from library.training_engine import train_one_epoch, validate
from library.experiment_manager import ExperimentManager


def main():
    # =========================================================================
    # 1. Initialization and Configuration
    # =========================================================================
    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    # Override Configuration for Robust Calibration
    # Increasing epochs to ensure convergence with lower LR (Cite 00063)
    Config.MAX_EPOCHS_PHASE_1 = 75
    Config.NUM_ENSEMBLE_MODELS = 5
    Config.SWA_EPOCHS = 12

    print("Configuration configured for robust calibration.")
    print(f"Phase 1 Epochs: {Config.MAX_EPOCHS_PHASE_1}")
    print(f"Ensemble Size: {Config.NUM_ENSEMBLE_MODELS}")

    # Initialize Experiment Manager
    manager = ExperimentManager()

    # =========================================================================
    # 2. Phase 1: Calibration (Global Epoch Selection)
    # =========================================================================
    # This uses 5-Fold CV on the full training data to find the optimal epoch.
    print("\n>>> Starting Phase 1: Calibration...")
    best_epoch = manager.run_calibration_phase()
    print(f">>> Calibration Complete. Best Convergence Epoch: {best_epoch}")

    # =========================================================================
    # 3. Validation Assessment (Strict Hold-out)
    # =========================================================================
    # We must evaluate on the specific hold-out set defined in ./metadata/val_metadata.csv
    print("\n>>> Starting Validation Assessment on Hold-out Set...")

    # Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)

    # Get indices for split
    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values

    # Load cached data arrays
    # Note: These arrays contain ALL training data ordered by train.json
    train_imgs_all, train_angles_all, train_labels_all, _, _, _ = get_data_arrays(
        load_cached_data=True
    )

    # Create Split Arrays
    X_train_sub = train_imgs_all[train_indices]
    a_train_sub = train_angles_all[train_indices]
    y_train_sub = train_labels_all[train_indices]

    X_val_sub = train_imgs_all[val_indices]
    a_val_sub = train_angles_all[val_indices]
    y_val_sub = train_labels_all[val_indices]

    # Define Transforms (Replicating logic from ExperimentManager)
    train_transform = A.Compose(
        [
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose([ToTensorV2()])

    # Create Datasets
    train_ds_sub = IcebergDataset(
        X_train_sub, a_train_sub, y_train_sub, transform=train_transform, mode="train"
    )
    val_ds_sub = IcebergDataset(
        X_val_sub, a_val_sub, y_val_sub, transform=val_transform, mode="val"
    )

    # Create DataLoaders
    train_loader_sub = DataLoader(
        train_ds_sub,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )
    val_loader_sub = DataLoader(
        val_ds_sub,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model for Validation Run
    val_model = IcebergResNet().to(Config.DEVICE)
    optimizer = get_optimizer(val_model)
    scheduler = get_scheduler(optimizer)
    criterion = SoftBCELoss()

    # Train for the determined best_epoch
    print(f"Training validation model for {best_epoch} epochs...")
    for epoch in range(1, best_epoch + 1):
        train_loss = train_one_epoch(
            val_model, train_loader_sub, optimizer, criterion, Config.DEVICE, epoch
        )
        # Validate to step scheduler
        val_loss = validate(val_model, val_loader_sub, Config.DEVICE)
        scheduler.step(val_loss)

    # Generate Predictions on Hold-out Set
    val_model.eval()
    val_preds = []
    val_targets = []
    val_angles = []

    with torch.no_grad():
        for images, angles, labels in val_loader_sub:
            images = images.to(Config.DEVICE)
            angles_gpu = angles.to(Config.DEVICE)

            logits = val_model(images, angles_gpu)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(labels.numpy().flatten())
            val_angles.extend(angles.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_angles = np.array(val_angles)

    # Calculate Metric
    final_metric = log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n>>> Performing Failure Analysis...")
    errors = np.abs(val_targets - val_preds)

    # Correlation with Incidence Angle
    corr_angle = np.corrcoef(errors, val_angles)[0, 1]
    print(f"Correlation between Error and Incidence Angle: {corr_angle:.6f}")

    # Correlation with Image Mean Intensity
    # Compute mean intensity for validation images
    # X_val_sub is (N, 224, 224, 3)
    img_means = X_val_sub.mean(axis=(1, 2, 3))
    corr_img_mean = np.corrcoef(errors, img_means)[0, 1]
    print(f"Correlation between Error and Image Mean Intensity: {corr_img_mean:.6f}")

    # =========================================================================
    # 5. Phase 2: Production & Submission
    # =========================================================================
    TARGET_THRESHOLD = 0.16918645240183008

    if final_metric < TARGET_THRESHOLD:
        print(
            f"\n>>> Metric {final_metric:.6f} passed threshold {TARGET_THRESHOLD:.6f}."
        )
        print(">>> Proceeding to Production Training and Submission Generation...")

        # Train Ensemble on Full Data
        models = manager.run_production_phase(best_epoch)

        # Generate Submission
        manager.run_inference(models)

    else:
        print(
            f"\n>>> Metric {final_metric:.6f} did not pass threshold {TARGET_THRESHOLD:.6f}."
        )
        print(">>> Submission generation skipped.")


if __name__ == "__main__":
    main()
