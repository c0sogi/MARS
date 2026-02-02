import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import seed_everything
from library.models import VolcanoEfficientNet
from library.data_processing import DataManager
from library.dataset import VolcanoSpectrogramDataset


def run_vision_training(debug=False):
    """
    Executes the training pipeline for the Vision Branch (Branch B).

    1. Loads spectrograms for Train, Val, and Test sets.
    2. Combines Train and Val for 5-Fold Cross-Validation.
    3. Trains 2D-CNN (EfficientNet) with Log-Target scaling.
    4. Generates OOF predictions and Test predictions (inverse transformed).

    Args:
        debug (bool): If True, runs on a subset of data.

    Returns:
        tuple: (df_oof, df_test_preds)
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing DataManager for Vision Training...")
    dm = DataManager()

    # Load Data
    # We need spectrograms (2nd return) and targets (3rd return)
    # The first return (tabular) is ignored here
    print("Loading Train data...")
    _, X_train_spec, y_train_part = dm.get_data(
        "train", load_cached_data=True, debug=debug
    )

    print("Loading Val data...")
    _, X_val_spec, y_val_part = dm.get_data("val", load_cached_data=True, debug=debug)

    print("Loading Test data...")
    X_test_tab, X_test_spec, _ = dm.get_data("test", load_cached_data=True, debug=debug)

    # Combine for CV
    X_full = np.concatenate([X_train_spec, X_val_spec], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    # We need segment_ids for the OOF dataframe.
    # We must reconstruct the segment_ids list in the same order as the concatenated data.
    # DataManager loads based on metadata CSV order.
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    test_meta = pd.read_csv(Config.TEST_METADATA)

    if debug:
        train_meta = train_meta.head(20)
        val_meta = val_meta.head(20)
        test_meta = test_meta.head(20)

    segment_ids_full = pd.concat(
        [train_meta["segment_id"], val_meta["segment_id"]], axis=0
    ).values
    test_segment_ids = test_meta["segment_id"].values

    print(f"Combined Spectrogram Shape: {X_full.shape}")
    print(f"Test Spectrogram Shape: {X_test_spec.shape}")

    # Prepare containers
    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros(len(X_test_spec))

    # Hyperparameters
    batch_size = Config.CNN_PARAMS["batch_size"]
    epochs = Config.CNN_PARAMS["epochs"]
    lr = Config.CNN_PARAMS["lr"]

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        print(f"\n--- Starting Vision Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split Data
        X_tr, y_tr = X_full[train_idx], y_full[train_idx]
        X_va, y_va = X_full[val_idx], y_full[val_idx]

        # Datasets
        # Train: Log targets for stability
        train_dataset = VolcanoSpectrogramDataset(X_tr, y_tr, log_target=True)
        # Val: Log targets for Loss calculation, but we will inverse for Metric
        val_dataset = VolcanoSpectrogramDataset(X_va, y_va, log_target=True)

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

        # Model Setup
        model = VolcanoEfficientNet(pretrained=True).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=Config.CNN_PARAMS["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.CNN_PARAMS["scheduler_T_max"],
            eta_min=Config.CNN_PARAMS["scheduler_eta_min"],
        )

        best_mae = float("inf")
        best_model_path = os.path.join(Config.WORKING_DIR, f"cnn_fold_{fold}.pth")

        # Training Loop
        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0

            for batch_spec, batch_target in train_loader:
                batch_spec = batch_spec.to(device)
                batch_target = batch_target.to(device)

                optimizer.zero_grad()
                output = model(batch_spec)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * batch_spec.size(0)

            scheduler.step()
            avg_train_loss = train_loss_sum / len(train_dataset)

            # Validation
            model.eval()
            val_preds_log = []
            val_targets_log = []

            with torch.no_grad():
                for batch_spec, batch_target in val_loader:
                    batch_spec = batch_spec.to(device)
                    output = model(batch_spec)

                    val_preds_log.append(output.cpu().numpy())
                    val_targets_log.append(batch_target.numpy())

            val_preds_log = np.concatenate(val_preds_log, axis=0).flatten()
            val_targets_log = np.concatenate(val_targets_log, axis=0).flatten()

            # Calculate Loss on Log Scale (for monitoring convergence)
            val_loss_log = mean_absolute_error(val_targets_log, val_preds_log)

            # Calculate MAE on Original Scale (for Model Selection)
            val_preds_orig = np.expm1(val_preds_log)
            val_targets_orig = np.expm1(val_targets_log)
            val_mae_orig = mean_absolute_error(val_targets_orig, val_preds_orig)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss (Log): {avg_train_loss} | Val MAE (Orig): {val_mae_orig}"
            )

            if val_mae_orig < best_mae:
                best_mae = val_mae_orig
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold+1} Best MAE: {best_mae}")

        # Load Best Model for Inference
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # 1. Generate OOF Predictions
        # We re-run validation loader to ensure order matches val_idx
        oof_preds_fold = []
        with torch.no_grad():
            for batch_spec, _ in val_loader:
                batch_spec = batch_spec.to(device)
                output = model(batch_spec)
                oof_preds_fold.append(output.cpu().numpy())

        oof_preds_fold = np.concatenate(oof_preds_fold, axis=0).flatten()
        # Inverse transform
        oof_preds[val_idx] = np.expm1(oof_preds_fold)

        # 2. Generate Test Predictions
        test_dataset = VolcanoSpectrogramDataset(
            X_test_spec, targets=None, log_target=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds_fold = []
        with torch.no_grad():
            for batch_spec in test_loader:
                batch_spec = batch_spec.to(device)
                output = model(batch_spec)
                test_preds_fold.append(output.cpu().numpy())

        test_preds_fold = np.concatenate(test_preds_fold, axis=0).flatten()
        # Inverse transform and accumulate
        test_preds_accum += np.expm1(test_preds_fold)

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            X_tr,
            X_va,
            train_loader,
            val_loader,
            test_loader,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # Average Test Predictions
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # Overall Score
    total_mae = mean_absolute_error(y_full, oof_preds)
    print(f"\nOverall Vision OOF MAE: {total_mae}")

    # Construct Output DataFrames
    df_oof = pd.DataFrame(
        {
            "segment_id": segment_ids_full,
            "pred_time_to_eruption": oof_preds,
            "true_time_to_eruption": y_full,
        }
    )

    df_test = pd.DataFrame(
        {"segment_id": test_segment_ids, "time_to_eruption": avg_test_preds}
    )

    return df_oof, df_test
