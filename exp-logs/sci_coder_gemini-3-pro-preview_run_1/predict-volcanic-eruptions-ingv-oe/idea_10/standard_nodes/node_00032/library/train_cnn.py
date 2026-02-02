import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.dataset import VolcanoTabularBuilder, VolcanoCNNDataset
from library.models import VolcanoEfficientNet
from library.utils import get_logger, seed_everything

# Initialize logger
logger = get_logger("train_cnn")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Validates the model. Returns Loss (Log-scale) and MAE (Original-scale).
    """
    model.eval()
    running_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Collect for MAE calculation
            preds_all.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    # Inverse Transform if Log Scaling was used
    if Config.TARGET_LOG_SCALE:
        preds_inv = np.expm1(preds_all)
        targets_inv = np.expm1(targets_all)
    else:
        preds_inv = preds_all
        targets_inv = targets_all

    epoch_mae = mean_absolute_error(targets_inv, preds_inv)

    return epoch_loss, epoch_mae


def predict_fn(model, loader, device):
    """
    Generates predictions for a dataloader. Returns predictions in Original-scale.
    """
    model.eval()
    preds_all = []

    with torch.no_grad():
        for batch in loader:
            # Handle case where dataset returns (input, target) or just (input)
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_all.append(outputs.cpu().numpy())

    preds_all = np.concatenate(preds_all)

    # Inverse Transform
    if Config.TARGET_LOG_SCALE:
        preds_inv = np.expm1(preds_all)
    else:
        preds_inv = preds_all

    return preds_inv.flatten()


def run_cnn_cv(debug=False):
    """
    Executes the training pipeline for Branch B: Contrast-Normalized Vision Model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    builder = VolcanoTabularBuilder()

    # Load Train and Val sets
    _, X_train_part, y_train_part = builder.get_data("train", load_cache=True)
    _, X_val_part, y_val_part = builder.get_data("val", load_cache=True)

    # Combine for CV
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    # Load Test set
    df_test, X_test, _ = builder.get_data("test", load_cache=True)

    if debug:
        logger.info("DEBUG mode enabled: Subsampling data.")
        indices = np.random.choice(
            len(X_full), size=min(100, len(X_full)), replace=False
        )
        X_full = X_full[indices]
        y_full = y_full[indices]

        test_indices = np.random.choice(
            len(X_test), size=min(50, len(X_test)), replace=False
        )
        X_test = X_test[test_indices]
        df_test = df_test.iloc[test_indices].reset_index(drop=True)

    logger.info(
        f"Training CNN on {len(X_full)} samples. Spectrogram shape: {X_full.shape[1:]}"
    )

    # ==========================================
    # 2. Cross-Validation Setup
    # ==========================================
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros(len(X_test))

    # Hyperparameters
    batch_size = Config.CNN_PARAMS.get("batch_size", 32)
    epochs = Config.CNN_PARAMS.get("epochs", 25)
    lr = Config.CNN_PARAMS.get("lr", 1e-3)

    if debug:
        epochs = 2

    fold_scores = []

    # ==========================================
    # 3. Training Loop
    # ==========================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        logger.info(f"\n--- Starting CNN Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Prepare Datasets
        train_dataset = VolcanoCNNDataset(X_full[train_idx], y_full[train_idx])
        val_dataset = VolcanoCNNDataset(X_full[val_idx], y_full[val_idx])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Scheduler
        model = VolcanoEfficientNet(pretrained=True)
        model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=Config.CNN_PARAMS.get("weight_decay", 1e-2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.CNN_PARAMS.get("t_max", epochs),
            eta_min=Config.CNN_PARAMS.get("eta_min", 1e-6),
        )
        criterion = nn.L1Loss()  # MAE Loss (applied on log-targets if configured)

        best_mae = float("inf")
        best_epoch = -1
        patience = 7  # Early stopping patience
        patience_counter = 0

        model_save_path = os.path.join(Config.WORKING_DIR, f"cnn_fold_{fold}.pth")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_mae = valid_one_epoch(model, val_loader, criterion, device)

            scheduler.step()

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val MAE: {val_mae}"
            )

            # Checkpoint
            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info(f"Fold {fold+1} Best MAE: {best_mae} at Epoch {best_epoch+1}")
        fold_scores.append(best_mae)

        # Load Best Model for Inference
        model.load_state_dict(torch.load(model_save_path, map_location=device))

        # OOF Predictions
        oof_pred = predict_fn(model, val_loader, device)
        oof_preds[val_idx] = oof_pred

        # Test Predictions
        test_dataset = VolcanoCNNDataset(X_test, targets=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        fold_test_pred = predict_fn(model, test_loader, device)
        test_preds_accum += fold_test_pred / Config.N_FOLDS

        # Clean up to save memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Results & Saving
    # ==========================================
    overall_mae = mean_absolute_error(y_full, oof_preds)
    logger.info(f"\nOverall CNN CV MAE: {overall_mae}")
    logger.info(f"Average Fold MAE: {np.mean(fold_scores)}")

    # Construct DataFrames
    # We need segment_ids for the OOF dataframe.
    # We concatenated train and val parts, so we need to concatenate their segment_ids in the same order.
    # We must reload metadata to get IDs or retrieve from builder if possible.
    # The builder returns df_tabular which has segment_id.
    df_train_tab, _, _ = builder.get_data("train", load_cache=True)
    df_val_tab, _, _ = builder.get_data("val", load_cache=True)

    if debug:
        # Re-apply sampling logic to get matching IDs if debug
        # Note: This relies on random seed consistency.
        # Ideally, we should have returned IDs from builder or passed them through.
        # For simplicity in this script, we assume the order is preserved from the load.
        # However, since we subsampled X_full using indices, we must subsample IDs too.
        # We need the original full list first.
        pass

    # Re-construct full ID list
    full_ids = pd.concat(
        [df_train_tab["segment_id"], df_val_tab["segment_id"]], axis=0
    ).reset_index(drop=True)

    if debug:
        full_ids = full_ids.iloc[indices].reset_index(drop=True)

    oof_df = pd.DataFrame(
        {"segment_id": full_ids, "time_to_eruption": y_full, "pred": oof_preds}
    )

    test_pred_df = df_test[["segment_id"]].copy()
    test_pred_df["time_to_eruption"] = test_preds_accum

    # Save
    oof_save_path = os.path.join(Config.WORKING_DIR, "cnn_oof.csv")
    test_save_path = os.path.join(Config.WORKING_DIR, "cnn_test.csv")

    oof_df.to_csv(oof_save_path, index=False)
    test_pred_df.to_csv(test_save_path, index=False)

    logger.info(f"Saved CNN OOF predictions to {oof_save_path}")
    logger.info(f"Saved CNN Test predictions to {test_save_path}")

    return oof_df, test_pred_df
