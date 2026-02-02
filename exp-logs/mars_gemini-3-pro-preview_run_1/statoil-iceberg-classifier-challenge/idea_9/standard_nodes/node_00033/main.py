import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, save_checkpoint, calculate_log_loss
from library.data_loader import load_and_process_data, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.engine import train_one_epoch, validate_with_tta, predict_tta


def run_teacher_stage(X_train, ang_train, y_train, device):
    """
    Trains a K-Fold ensemble on the training split to generate OOF predictions.
    """
    print("\n=== Stage 1: Teacher Ensemble (K-Fold) ===")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Data Splits
        X_tr, ang_tr, y_tr = (
            X_train[train_idx],
            ang_train[train_idx],
            y_train[train_idx],
        )
        X_val, ang_val, y_val = X_train[val_idx], ang_train[val_idx], y_train[val_idx]

        # Datasets & Loaders
        train_ds = IcebergDataset(
            X_tr, ang_tr, labels=y_tr, transform=get_transforms(mode="train")
        )
        val_ds = IcebergDataset(
            X_val, ang_val, labels=y_val, transform=get_transforms(mode="val")
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        # Model, Optimizer, Scheduler
        model = IcebergResNet18().to(device)
        optimizer = optim.AdamW(
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

        best_loss = float("inf")
        best_state = None

        # Training Loop
        for epoch in range(Config.NUM_EPOCHS):
            # Train (No distillation in teacher stage)
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch, use_distillation=False
            )

            # Validate (using TTA as per engine)
            val_loss = validate_with_tta(model, val_loader, device)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                # Save best model for this fold
                save_checkpoint(
                    best_state,
                    True,
                    Config.TEACHER_CHECKPOINT_DIR,
                    f"fold_{fold}_best.pth",
                )

        print(f"Fold {fold} Best Loss: {best_loss:.6f}")

        # Load best weights for OOF prediction
        model.load_state_dict(best_state)

        # Generate OOF predictions (using TTA for high quality soft targets)
        fold_preds = predict_tta(model, val_loader, device)
        oof_preds[val_idx] = fold_preds

    # Save OOF predictions
    np.save(Config.OOF_PREDICTIONS_PATH, oof_preds)
    print(f"OOF predictions saved to {Config.OOF_PREDICTIONS_PATH}")
    return oof_preds


def run_student_stage(X_train, ang_train, y_train, oof_preds, device):
    """
    Trains the Student model on the full training split using OOF predictions as soft targets.
    """
    print("\n=== Stage 2: Student Model (Distillation) ===")

    # Dataset & Loader
    train_ds = IcebergDataset(
        X_train,
        ang_train,
        labels=y_train,
        soft_targets=oof_preds,
        transform=get_transforms(mode="train"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model Setup
    model = IcebergResNet18().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, use_distillation=True
        )
        scheduler.step(loss)

    # Save final student model
    save_checkpoint(
        model.state_dict(), True, Config.STUDENT_CHECKPOINT_DIR, "student_final.pth"
    )
    return model


def failure_analysis(model, val_loader, device, X_val, ang_val, y_val):
    print("\n=== Failure Analysis ===")
    model.eval()

    # Get predictions
    preds = predict_tta(model, val_loader, device)

    # Calculate errors
    errors = np.abs(y_val - preds)
    log_loss_val = calculate_log_loss(y_val, preds)

    print(f"Final Validation Metric: {log_loss_val}")

    # Feature correlations
    # 1. Incidence Angle
    corr_angle, _ = pearsonr(errors, ang_val)
    print(f"Correlation (Error vs Inc_Angle): {corr_angle:.4f}")

    # 2. Image Stats (Mean of Band 1, Band 2)
    # X_val is (N, 75, 75, 3). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_means = np.mean(X_val[:, :, :, 0], axis=(1, 2))
    b2_means = np.mean(X_val[:, :, :, 1], axis=(1, 2))

    corr_b1, _ = pearsonr(errors, b1_means)
    corr_b2, _ = pearsonr(errors, b2_means)

    print(f"Correlation (Error vs Band1_Mean): {corr_b1:.4f}")
    print(f"Correlation (Error vs Band2_Mean): {corr_b2:.4f}")

    return log_loss_val


def main():
    seed_everything(Config.SEED)
    Config.create_directories()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data (All labeled data)
    X_all, ang_all, y_all, ids_all, X_test, ang_test, ids_test = load_and_process_data(
        load_cached_data=True
    )

    # 2. Split into Train/Val based on Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)

    train_ids_set = set(df_train_meta["id"].values)
    val_ids_set = set(df_val_meta["id"].values)

    # Create masks
    train_mask = np.array([uid in train_ids_set for uid in ids_all])
    val_mask = np.array([uid in val_ids_set for uid in ids_all])

    # Apply masks
    X_train = X_all[train_mask]
    ang_train = ang_all[train_mask]
    y_train = y_all[train_mask]

    X_val = X_all[val_mask]
    ang_val = ang_all[val_mask]
    y_val = y_all[val_mask]

    print(f"Training Set Size: {len(X_train)}")
    print(f"Validation Set Size: {len(X_val)}")

    # 3. Stage 1: Teacher Ensemble
    if os.path.exists(Config.OOF_PREDICTIONS_PATH):
        print("Loading existing OOF predictions...")
        oof_preds = np.load(Config.OOF_PREDICTIONS_PATH)
    else:
        oof_preds = run_teacher_stage(X_train, ang_train, y_train, device)

    # 4. Stage 2: Student Model
    student_model = run_student_stage(X_train, ang_train, y_train, oof_preds, device)

    # 5. Final Validation & Failure Analysis
    val_ds = IcebergDataset(
        X_val, ang_val, labels=y_val, transform=get_transforms(mode="val")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_metric = failure_analysis(
        student_model, val_loader, device, X_val, ang_val, y_val
    )

    # 6. Submission
    THRESHOLD = 0.17822679498532543
    if val_metric < THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")

        test_ds = IcebergDataset(
            X_test, ang_test, labels=None, transform=get_transforms(mode="test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        test_preds = predict_tta(student_model, test_loader, device)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
