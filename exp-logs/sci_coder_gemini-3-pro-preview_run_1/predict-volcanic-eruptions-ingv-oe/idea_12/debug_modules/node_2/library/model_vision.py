import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, mae_score
from library.dataset_vision import get_vision_loaders


class VolcanoEfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for 20-channel input (Dual-Resolution Spectrograms).
    """

    def __init__(self, backbone=Config.CNN_BACKBONE, pretrained=Config.CNN_PRETRAINED):
        super(VolcanoEfficientNet, self).__init__()
        # Create EfficientNet with modified input channels (20) and 1 output class (regression)
        # timm handles the weight initialization for the new input channels (usually by averaging original RGB weights)
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=1,
            global_pool="avg",
        )

    def forward(self, x):
        # Output shape: (Batch_Size, 1)
        return self.model(x)


def train_cnn_fold(
    train_df,
    val_df,
    fold_idx,
    load_cached_data=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=Config.CNN_EPOCHS,
    patience=Config.CNN_PATIENCE,
):
    """
    Trains the Vision Branch (CNN) for a single fold.

    Args:
        train_df (pd.DataFrame): Training metadata for this fold.
        val_df (pd.DataFrame): Validation metadata for this fold.
        fold_idx (int): Index of the current fold (for saving models).
        load_cached_data (bool): Whether to use cached spectrograms.
        device (str): Computation device.
        epochs (int): Maximum training epochs.
        patience (int): Early stopping patience.

    Returns:
        tuple: (best_val_mae, val_predictions_df)
    """
    seed_everything(Config.SEED + fold_idx)

    # 1. Prepare DataLoaders
    loaders = get_vision_loaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 2. Initialize Model, Optimizer, Scheduler, Loss
    model = VolcanoEfficientNet().to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.CNN_LR, weight_decay=Config.CNN_WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss function for log-scaled targets
    criterion = nn.L1Loss()

    # Mixed Precision Scaler
    scaler = GradScaler()

    # 3. Training Loop
    best_val_mae = float("inf")
    early_stop_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, f"cnn_fold_{fold_idx}.pth")

    print(f"\n[Fold {fold_idx}] Starting CNN Training on {device}...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_losses = []

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad()

            with autocast():
                preds = model(images)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        model.eval()
        val_preds_log = []
        val_targets_log = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).view(-1, 1)

                with autocast():
                    preds = model(images)

                val_preds_log.append(preds.cpu().numpy())
                val_targets_log.append(targets.cpu().numpy())

        # Concatenate
        val_preds_log = np.concatenate(val_preds_log, axis=0)
        val_targets_log = np.concatenate(val_targets_log, axis=0)

        # Inverse Transform: expm1(log1p(x)) -> x
        # We calculate MAE on the original time scale
        val_preds_original = np.expm1(val_preds_log)
        val_targets_original = np.expm1(val_targets_log)

        # Clip negative predictions to 0 (time cannot be negative)
        val_preds_original = np.maximum(val_preds_original, 0)

        current_val_mae = mae_score(val_targets_original, val_preds_original)

        scheduler.step()
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss (Log): {avg_train_loss:.6f} | "
            f"Val MAE (Original): {current_val_mae} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- Early Stopping & Checkpointing ---
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> Model Saved! Improved MAE: {best_val_mae}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # 4. Load Best Model and Generate Final Val Predictions
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    final_preds_log = []
    final_indices = (
        []
    )  # We assume loader preserves order, but robust way is to map via dataframe

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with autocast():
                preds = model(images)
            final_preds_log.append(preds.cpu().numpy())

    final_preds_log = np.concatenate(final_preds_log, axis=0)
    final_preds_original = np.expm1(final_preds_log)
    final_preds_original = np.maximum(final_preds_original, 0)

    # Create DataFrame for OOF
    oof_df = val_df[["segment_id"]].copy()
    oof_df["time_to_eruption_pred"] = final_preds_original.flatten()

    print(f"[Fold {fold_idx}] Finished. Best Val MAE: {best_val_mae}")
    return best_val_mae, oof_df


def inference_cnn(
    test_df,
    model_paths,
    load_cached_data=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generates predictions for the test set using an ensemble of trained models (folds).

    Args:
        test_df (pd.DataFrame): Test metadata.
        model_paths (list): List of paths to .pth model files.
        load_cached_data (bool): Whether to use cached spectrograms.
        device (str): Computation device.

    Returns:
        pd.DataFrame: Test DataFrame with 'time_to_eruption' predictions.
    """
    # Prepare Test Loader
    # Note: get_vision_loaders returns a dict. We pass None for train/val to skip them if possible,
    # but the function signature requires train/val. We pass dummy empty DFs or handle it.
    # Actually, get_vision_loaders requires train/val DFs.
    # To avoid overhead, we instantiate the dataset and loader directly here.

    from library.dataset_vision import VolcanoCNNDataset, generate_dataset_spectrograms
    from torch.utils.data import DataLoader

    test_dir = generate_dataset_spectrograms(
        test_df, "test", load_cached_data=load_cached_data
    )
    test_dataset = VolcanoCNNDataset(test_df, test_dir, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Array to store predictions from each fold: (Num_Samples, Num_Models)
    ensemble_preds = np.zeros((len(test_df), len(model_paths)))

    print(
        f"Running Inference on {len(test_df)} test samples with {len(model_paths)} models..."
    )

    for i, path in enumerate(model_paths):
        model = VolcanoEfficientNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        fold_preds_log = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device, non_blocking=True)
                with autocast():
                    preds = model(images)
                fold_preds_log.append(preds.cpu().numpy())

        fold_preds_log = np.concatenate(fold_preds_log, axis=0)
        # Inverse transform
        fold_preds_original = np.expm1(fold_preds_log)
        fold_preds_original = np.maximum(fold_preds_original, 0)

        ensemble_preds[:, i] = fold_preds_original.flatten()

    # Average predictions across folds
    avg_preds = np.mean(ensemble_preds, axis=1)

    submission_df = test_df[["segment_id"]].copy()
    submission_df["time_to_eruption"] = avg_preds

    return submission_df
