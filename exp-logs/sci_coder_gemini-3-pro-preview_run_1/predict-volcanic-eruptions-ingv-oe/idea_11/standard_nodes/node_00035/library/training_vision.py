import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from library.config import (
    WORK_DIR,
    SEED,
    N_FOLDS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    DEVICE,
    VISION_TRAIN_CACHE_DIR,
    VISION_VAL_CACHE_DIR,
    USE_LOG_TARGET,
    DEBUG,
    MAX_DEBUG_SAMPLES,
)
from library.utils import (
    seed_everything,
    calc_mae,
    log_transform_target,
    exp_transform_target,
)
from library.data_factory import get_vision_dataset
from library.vision_model import VolcanoEfficientNet

# Set seeds
seed_everything(SEED)


class UnifiedVolcanoDataset(Dataset):
    """
    A wrapper dataset that can load spectrograms from multiple directories
    (train/val caches) based on a segment_id mapping.
    """

    def __init__(self, metadata: pd.DataFrame, dir_map: dict, is_test: bool = False):
        self.metadata = metadata
        self.dir_map = dir_map
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        segment_id = int(row["segment_id"])

        # Locate the directory for this segment
        data_dir = self.dir_map.get(segment_id)

        if data_dir is None:
            # Fallback (should not happen if map is complete)
            spectrogram = np.zeros((10, 224, 224), dtype=np.float32)
        else:
            file_path = os.path.join(data_dir, f"{segment_id}.npy")
            if os.path.exists(file_path):
                spectrogram = np.load(file_path)
            else:
                spectrogram = np.zeros((10, 224, 224), dtype=np.float32)

        x = torch.from_numpy(spectrogram)

        if self.is_test:
            y = torch.tensor(0.0, dtype=torch.float32)
        else:
            target_val = float(row["time_to_eruption"])
            if USE_LOG_TARGET:
                target_val = log_transform_target(target_val)
            y = torch.tensor(target_val, dtype=torch.float32)

        return x, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_gpu = y.to(device).unsqueeze(1)

            preds = model(x)
            loss = criterion(preds, y_gpu)
            running_loss += loss.item() * x.size(0)

            # Collect for MAE calculation (inverse transform)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(y.numpy())

    epoch_loss = running_loss / len(loader.dataset)

    preds_arr = np.concatenate(preds_list).flatten()
    targets_arr = np.concatenate(targets_list).flatten()

    # Inverse transform to calculate MAE on original scale
    if USE_LOG_TARGET:
        preds_orig = exp_transform_target(preds_arr)
        targets_orig = exp_transform_target(targets_arr)
    else:
        preds_orig = preds_arr
        targets_orig = targets_arr

    epoch_mae = calc_mae(targets_orig, preds_orig)

    return epoch_loss, epoch_mae, preds_orig


def run_vision_cv(load_cached_data: bool = True):
    """
    Executes 5-Fold Cross-Validation for the Vision Branch.
    """
    print("Initializing Vision Branch Training (EfficientNet)...")

    # 1. Ensure Data Generation
    # Calling get_vision_dataset ensures the .npy files are generated in their respective folders
    ds_train_part = get_vision_dataset("train", load_cached_data=load_cached_data)
    ds_val_part = get_vision_dataset("val", load_cached_data=load_cached_data)
    ds_test = get_vision_dataset("test", load_cached_data=load_cached_data)

    # 2. Prepare Unified Metadata and Directory Map
    meta_train = ds_train_part.metadata.copy()
    meta_val = ds_val_part.metadata.copy()

    # Map segment_ids to their physical directory
    dir_map = {}
    for sid in meta_train["segment_id"]:
        dir_map[sid] = VISION_TRAIN_CACHE_DIR
    for sid in meta_val["segment_id"]:
        dir_map[sid] = VISION_VAL_CACHE_DIR

    # Combine for CV
    df_full = pd.concat([meta_train, meta_val], axis=0).reset_index(drop=True)

    if DEBUG:
        print(f"DEBUG Mode: Subsampling Vision data to {MAX_DEBUG_SAMPLES} samples.")
        df_full = df_full.sample(
            n=min(len(df_full), MAX_DEBUG_SAMPLES), random_state=SEED
        ).reset_index(drop=True)
        # Also subsample test
        ds_test.metadata = ds_test.metadata.sample(
            n=min(len(ds_test.metadata), MAX_DEBUG_SAMPLES), random_state=SEED
        ).reset_index(drop=True)

    print(f"Total Training Samples: {len(df_full)}")
    print(f"Total Test Samples: {len(ds_test)}")

    # 3. Cross Validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(df_full))
    test_preds_accum = np.zeros(len(ds_test))

    train_segment_ids = df_full["segment_id"].values
    train_targets = df_full["time_to_eruption"].values

    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_full)):
        print(f"\n--- Vision Fold {fold + 1}/{N_FOLDS} ---")

        # Split Metadata
        df_train_fold = df_full.iloc[train_idx]
        df_val_fold = df_full.iloc[val_idx]

        # Create Datasets
        train_dataset = UnifiedVolcanoDataset(df_train_fold, dir_map)
        val_dataset = UnifiedVolcanoDataset(df_val_fold, dir_map)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = VolcanoEfficientNet()
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        criterion = nn.L1Loss()  # MAE Loss on Log Targets

        best_mae = float("inf")
        best_model_path = os.path.join(WORK_DIR, f"cnn_fold_{fold}.pth")
        patience = 8
        counter = 0

        # Training Loop
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_mae, _ = validate_one_epoch(
                model, val_loader, criterion, DEVICE
            )

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae}"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), best_model_path)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Fold {fold+1} Best MAE: {best_mae}")
        fold_maes.append(best_mae)

        # Load Best Model for Inference
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.eval()

        # Generate OOF Preds
        _, _, val_preds_orig = validate_one_epoch(model, val_loader, criterion, DEVICE)
        oof_preds[val_idx] = val_preds_orig

        # Generate Test Preds
        test_loader = DataLoader(
            ds_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        fold_test_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(DEVICE)
                out = model(x)
                fold_test_preds.append(out.cpu().numpy())

        fold_test_preds = np.concatenate(fold_test_preds).flatten()

        if USE_LOG_TARGET:
            fold_test_preds = exp_transform_target(fold_test_preds)

        test_preds_accum += fold_test_preds

    # 4. Final Aggregation
    avg_test_preds = test_preds_accum / N_FOLDS
    overall_mae = calc_mae(train_targets, oof_preds)
    avg_fold_mae = np.mean(fold_maes)

    print("\n--- Vision Training Complete ---")
    print(f"Average Fold MAE: {avg_fold_mae}")
    print(f"Overall OOF MAE: {overall_mae}")

    # Save Results
    oof_df = pd.DataFrame(
        {
            "segment_id": train_segment_ids,
            "time_to_eruption": train_targets,
            "cnn_pred": oof_preds,
        }
    )

    test_df = pd.DataFrame(
        {"segment_id": ds_test.metadata["segment_id"], "cnn_pred": avg_test_preds}
    )

    oof_save_path = os.path.join(WORK_DIR, "cnn_oof.csv")
    test_save_path = os.path.join(WORK_DIR, "cnn_test.csv")

    oof_df.to_csv(oof_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print(f"Saved OOF predictions to {oof_save_path}")
    print(f"Saved Test predictions to {test_save_path}")

    return oof_df, test_df
