import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torchvision import transforms
import warnings

# Import from provided library files
from library.config import (
    TRAIN_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_JSON,
    TEST_META_PATH,
    CHECKPOINT_DIR,
    SUBMISSION_FILE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_WORKERS,
    SEED,
    DEVICE,
    N_FOLDS,
    DROPBLOCK_START_PROB,
    DROPBLOCK_MAX_PROB,
)
from library.utils import set_seed, generate_submission_file
from library.data_loader import load_data_split, IcebergDataset, seed_worker
import library.model
import importlib

importlib.reload(library.model)
from library.model import DPDB_HSE_CNN
from library.train import train_one_epoch, validate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_training_fold(fold_idx, train_loader, val_loader, device):
    """
    Trains a single fold model.
    """
    model = DPDB_HSE_CNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Checkpoint path
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    # print(f"Fold {fold_idx}: Training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        # Train one epoch (handles DropBlock scheduling internally)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                # print(f"Fold {fold_idx}: Early stopping at epoch {epoch+1}")
                break

    return best_val_loss


def predict_ensemble(models, loader, device):
    """
    Generates averaged predictions from a list of models.
    """
    # Initialize accumulator
    total_preds = None
    sample_count = len(loader.dataset)

    for model in models:
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for images, angles, _, _ in loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_preds.extend(probs.cpu().numpy().flatten())

        fold_preds = np.array(fold_preds)
        if total_preds is None:
            total_preds = fold_preds
        else:
            total_preds += fold_preds

    avg_preds = total_preds / len(models)
    return avg_preds


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data Splits
    # We strictly use the metadata splits: Train for training, Val for hold-out evaluation
    X_train, ang_train, y_train, id_train = load_data_split(
        TRAIN_META_PATH, TRAIN_JSON, "train", load_cached_data=True
    )
    X_val, ang_val, y_val, id_val = load_data_split(
        VAL_META_PATH, TRAIN_JSON, "val", load_cached_data=True
    )
    X_test, ang_test, _, id_test = load_data_split(
        TEST_META_PATH, TEST_JSON, "test", load_cached_data=True
    )

    # 3. Imputation (Incidence Angle)
    # Use median from the training set to fill all splits
    angle_median = np.nanmedian(ang_train)

    ang_train = np.where(np.isnan(ang_train), angle_median, ang_train)
    ang_val = np.where(np.isnan(ang_val), angle_median, ang_val)
    ang_test = np.where(np.isnan(ang_test), angle_median, ang_test)

    # 4. Train 5-Fold Ensemble on TRAIN set
    # We split the 'train' set into K-Folds to train 5 diverse models
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Generator for reproducible DataLoaders
    g = torch.Generator()
    g.manual_seed(SEED)

    trained_models = []

    # print("Starting Ensemble Training...")

    for fold_idx, (idx_tr, idx_v) in enumerate(skf.split(X_train, y_train)):
        # Split internal train/val for this fold
        X_tr_fold, X_v_fold = X_train[idx_tr], X_train[idx_v]
        ang_tr_fold, ang_v_fold = ang_train[idx_tr], ang_train[idx_v]
        y_tr_fold, y_v_fold = y_train[idx_tr], y_train[idx_v]
        id_tr_fold, id_v_fold = id_train[idx_tr], id_train[idx_v]

        # Create Datasets
        ds_tr = IcebergDataset(
            X_tr_fold, ang_tr_fold, y_tr_fold, id_tr_fold, transform=train_transform
        )
        ds_v = IcebergDataset(X_v_fold, ang_v_fold, y_v_fold, id_v_fold, transform=None)

        # Create DataLoaders
        dl_tr = DataLoader(
            ds_tr,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )
        dl_v = DataLoader(
            ds_v,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )

        # Train Fold
        run_training_fold(fold_idx, dl_tr, dl_v, DEVICE)

        # Load Best Model
        model = DPDB_HSE_CNN().to(DEVICE)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        trained_models.append(model)

    # 5. Validation on Hold-out Set
    # print("Evaluating on Hold-out Validation Set...")
    ds_val = IcebergDataset(X_val, ang_val, y_val, id_val, transform=None)
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )

    # Get Ensemble Predictions
    val_preds = predict_ensemble(trained_models, dl_val, DEVICE)

    # Calculate Metric
    final_metric = log_loss(y_val, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # print("Performing Failure Analysis...")
    errors = np.abs(y_val - val_preds)

    # Compute simple image stats for correlation
    # X_val shape is (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_means = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X_val[:, 1, :, :], axis=(1, 2))

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_val,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 7. Conditional Submission
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        # print("Metric check passed. Generating submission...")
        ds_test = IcebergDataset(X_test, ang_test, None, id_test, transform=None)
        dl_test = DataLoader(
            ds_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )

        test_preds = predict_ensemble(trained_models, dl_test, DEVICE)
        generate_submission_file(test_preds, id_test, SUBMISSION_FILE)
    else:
        print(
            f"Validation metric {final_metric} is not lower than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
