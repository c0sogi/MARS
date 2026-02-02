import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import library modules
import library.config as config
import library.utils as utils
import library.data as data_lib
import library.model as model_lib
import library.engine as engine_lib
import library.sam as sam_lib

# Define constants for this run
PHASE1_EPOCHS = 25
SWA_EPOCHS = 6
BATCH_SIZE = 32
THRESHOLD = 0.16918645240183008


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data and Metadata
    print("Loading data...")
    # Load cached numpy arrays (contains all train.json data)
    data_container = data_lib.load_data(load_cached_data=True)
    all_train_b1, all_train_b2, all_train_angles, all_train_labels, all_train_ids = (
        data_container["train"]
    )
    stats = data_container["stats"]

    # Load metadata to split into Train/Val
    df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(config.VAL_META_PATH)

    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values

    # Slice data
    train_b1 = all_train_b1[train_indices]
    train_b2 = all_train_b2[train_indices]
    train_angles = all_train_angles[train_indices]
    train_labels = all_train_labels[train_indices]
    train_ids = all_train_ids[train_indices]

    val_b1 = all_train_b1[val_indices]
    val_b2 = all_train_b2[val_indices]
    val_angles = all_train_angles[val_indices]
    val_labels = all_train_labels[val_indices]
    val_ids = all_train_ids[val_indices]

    print(f"Train set size: {len(train_indices)}")
    print(f"Val set size: {len(val_indices)}")

    # 3. Create Datasets and Loaders
    # Transforms
    train_transform = data_lib.get_transforms("train")
    val_transform = data_lib.get_transforms("val")

    train_dataset = data_lib.IcebergDataset(
        train_b1,
        train_b2,
        train_angles,
        labels=train_labels,
        ids=train_ids,
        transform=train_transform,
        stats=stats,
    )
    val_dataset = data_lib.IcebergDataset(
        val_b1,
        val_b2,
        val_angles,
        labels=val_labels,
        ids=val_ids,
        transform=val_transform,
        stats=stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model, Optimizer, Scheduler (Phase 1)
    print("\n--- Phase 1: Calibration & Validation ---")
    model = model_lib.IcebergResNet(
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
        gem_p=config.GEM_P_INIT,
        gem_trainable=config.GEM_P_TRAINABLE,
    ).to(device)

    # Base optimizer for SAM
    base_optimizer = torch.optim.AdamW
    optimizer = sam_lib.SAM(
        model.parameters(),
        base_optimizer,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        rho=0.05,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.base_optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
        verbose=False,
    )

    # 5. Train Phase 1
    # We use SWA here to get the best possible validation metric for the report
    swa_start = PHASE1_EPOCHS - SWA_EPOCHS
    if swa_start < 1:
        swa_start = 1

    trained_model = engine_lib.fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=PHASE1_EPOCHS,
        patience=config.EARLY_STOPPING_PATIENCE,
        use_swa=True,
        swa_start_epoch=swa_start,
        save_dir=config.CHECKPOINT_DIR,
        fold_idx="val_split",
    )

    # 6. Validation Assessment
    print("\n--- Validation Assessment ---")
    criterion = engine_lib.BCEWithLogitsLossLabelSmoothing()
    val_loss, val_acc = engine_lib.validate_tta(
        val_loader, trained_model, criterion, device
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    trained_model.eval()
    all_preds = []
    all_targets = []

    # Get predictions
    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)

            # TTA Preds
            out1 = torch.sigmoid(trained_model(images, angles))
            out2 = torch.sigmoid(trained_model(torch.flip(images, [3]), angles))
            out3 = torch.sigmoid(trained_model(torch.flip(images, [2]), angles))
            out4 = torch.sigmoid(trained_model(torch.rot90(images, 2, [2, 3]), angles))
            avg_preds = (out1 + out2 + out3 + out4) / 4.0

            all_preds.append(avg_preds.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    # Calculate errors
    errors = np.abs(all_preds - all_targets)

    # Features for correlation
    b1_means = np.mean(val_b1, axis=(1, 2))
    b2_means = np.mean(val_b2, axis=(1, 2))
    b1_stds = np.std(val_b1, axis=(1, 2))
    b2_stds = np.std(val_b2, axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": val_angles,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
            "b1_std": b1_stds,
            "b2_std": b2_stds,
        }
    )

    print("Correlation between Error and Features:")
    corrs = analysis_df.corr()["error"].drop("error")
    print(corrs)

    # 8. Submission (Conditional)
    if val_loss < THRESHOLD:
        print("\n--- Phase 2: Production Training & Submission ---")
        # Train on FULL dataset
        # Create Full Dataset
        full_train_dataset = data_lib.IcebergDataset(
            all_train_b1,
            all_train_b2,
            all_train_angles,
            labels=all_train_labels,
            ids=all_train_ids,
            transform=train_transform,
            stats=stats,
        )
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # Re-initialize model
        prod_model = model_lib.IcebergResNet(
            backbone_name=config.BACKBONE,
            pretrained=config.PRETRAINED,
            dropout_rate=config.DROPOUT_RATE,
            num_classes=config.NUM_CLASSES,
            gem_p=config.GEM_P_INIT,
            gem_trainable=config.GEM_P_TRAINABLE,
        ).to(device)

        prod_optimizer = sam_lib.SAM(
            prod_model.parameters(),
            base_optimizer,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            rho=0.05,
        )

        prod_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            prod_optimizer.base_optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
            verbose=False,
        )

        # Train
        prod_model_trained = engine_lib.fit_model(
            model=prod_model,
            train_loader=full_train_loader,
            val_loader=None,  # No validation in production
            optimizer=prod_optimizer,
            scheduler=prod_scheduler,
            device=device,
            epochs=PHASE1_EPOCHS,
            use_swa=True,
            swa_start_epoch=swa_start,
            save_dir=config.CHECKPOINT_DIR,
            fold_idx="full",
        )

        # Predict on Test
        print("Generating predictions...")
        test_loader = data_lib.get_dataloaders(mode="test", load_cached_data=True)

        preds, ids = engine_lib.predict_tta(test_loader, prod_model_trained, device)

        # Save
        sub_df = pd.DataFrame({"id": ids, "is_iceberg": preds})
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_loss} is not better than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
