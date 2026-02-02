import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything, load_npy, save_npy, load_parquet, get_score
from library.dataset import get_data_loaders


class VolcanoEfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for 10-channel seismic spectrogram inputs.
    """

    def __init__(self, model_name=None, pretrained=True):
        super(VolcanoEfficientNet, self).__init__()
        config = Config()
        name = model_name if model_name else config.NN_PARAMS["model_name"]

        # Create model with modified input channels (10 sensors)
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=config.NN_PARAMS["in_channels"],
            num_classes=config.NN_PARAMS["num_classes"],
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data).squeeze()

        # Loss is calculated in the Log-Domain as per strategy
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_epoch(model, loader, criterion, device):
    """
    Validation loop. Calculates Loss (Log-domain) and MAE (Real-domain).
    """
    model.eval()
    running_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze()
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

            # Inverse transform for Real-Scale MAE calculation
            # Target was log1p scaled, so we use expm1
            pred_real = torch.expm1(output)
            target_real = torch.expm1(target)

            preds_all.append(pred_real.cpu().numpy())
            targets_all.append(target_real.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate and calculate Real MAE
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    # Ensure no negative predictions (though expm1 handles this, safety check)
    preds_all = np.maximum(0, preds_all)

    real_mae = get_score(targets_all, preds_all)

    return epoch_loss, real_mae, preds_all


def run_vision_cv(debug=False):
    """
    Orchestrates the 5-Fold Cross-Validation for the Vision Branch.
    Loads data, trains models, generates OOF and Test predictions.
    """
    config = Config()
    seed_everything(config.SEED)
    device = torch.device(config.NN_PARAMS["device"])

    print("Initializing Vision Branch Cross-Validation...")

    # ==========================================
    # 1. Load and Prepare Data
    # ==========================================
    working_dir = config.WORKING_DIR

    # Paths
    train_spec_path = os.path.join(working_dir, "train_spectrograms.npy")
    train_target_path = os.path.join(working_dir, "train_targets.npy")
    train_feat_path = os.path.join(working_dir, "train_features.parquet")

    val_spec_path = os.path.join(working_dir, "val_spectrograms.npy")
    val_target_path = os.path.join(working_dir, "val_targets.npy")
    val_feat_path = os.path.join(working_dir, "val_features.parquet")

    test_spec_path = os.path.join(working_dir, "test_spectrograms.npy")
    test_feat_path = os.path.join(working_dir, "test_features.parquet")

    # Check existence
    if not (
        os.path.exists(train_spec_path)
        and os.path.exists(val_spec_path)
        and os.path.exists(test_spec_path)
    ):
        raise FileNotFoundError(
            "Cached data not found. Run feature_engineering.py first."
        )

    # Load Data
    print("Loading cached data...")
    X_train_part = load_npy(train_spec_path)
    y_train_part = load_npy(train_target_path)
    df_train_part = load_parquet(train_feat_path)

    X_val_part = load_npy(val_spec_path)
    y_val_part = load_npy(val_target_path)
    df_val_part = load_parquet(val_feat_path)

    X_test = load_npy(test_spec_path)
    df_test = load_parquet(test_feat_path)

    # Concatenate Train and Val to form the full development set for CV
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    df_full = pd.concat([df_train_part, df_val_part], axis=0).reset_index(drop=True)

    # Debug Mode: Subsample
    if debug:
        print("DEBUG MODE: Subsampling data...")
        subset_size = 100
        X_full = X_full[:subset_size]
        y_full = y_full[:subset_size]
        df_full = df_full.iloc[:subset_size]
        X_test = X_test[:subset_size]
        df_test = df_test.iloc[:subset_size]
        config.NN_PARAMS["epochs"] = 2  # Reduce epochs for debug

    print(f"Full Train Shape: {X_full.shape}, Test Shape: {X_test.shape}")

    # ==========================================
    # 2. Cross-Validation Loop
    # ==========================================
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)

    # Containers for OOF and Test Predictions
    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros((len(X_test), 5))

    # Criterion (L1 Loss for MAE optimization)
    criterion = nn.L1Loss()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1} / 5 ---")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Get Loaders
        train_loader, val_loader = get_data_loaders(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )

        # Initialize Model, Optimizer, Scheduler
        model = VolcanoEfficientNet().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.NN_PARAMS["learning_rate"],
            weight_decay=config.NN_PARAMS["weight_decay"],
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.NN_PARAMS["scheduler_T_max"]
        )

        # Training Loop
        best_val_mae = float("inf")
        best_model_state = None
        patience = 5
        patience_counter = 0

        for epoch in range(config.NN_PARAMS["epochs"]):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            val_loss, val_mae, _ = validate_epoch(model, val_loader, criterion, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{config.NN_PARAMS['epochs']} | "
                f"Train Loss (Log): {train_loss:.5f} | "
                f"Val Loss (Log): {val_loss:.5f} | "
                f"Val MAE (Real): {val_mae:.5f}"
            )

            # Checkpoint & Early Stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load Best Model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

            # Save Model Checkpoint
            ckpt_path = os.path.join(working_dir, f"cnn_fold_{fold}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Generate OOF Predictions for this fold
        _, _, fold_oof_preds = validate_epoch(model, val_loader, criterion, device)
        oof_preds[val_idx] = fold_oof_preds

        # Generate Test Predictions for this fold
        # Create a dummy loader for test set (reuse get_data_loaders with dummy targets)
        dummy_test_targets = np.zeros(len(X_test))
        _, test_loader = get_data_loaders(
            X_train_fold[:10],
            y_train_fold[:10],  # Dummy train
            X_test,
            dummy_test_targets,  # Real test
        )

        _, _, fold_test_preds = validate_epoch(model, test_loader, criterion, device)
        test_preds_accum[:, fold] = fold_test_preds

    # ==========================================
    # 3. Aggregate and Save Results
    # ==========================================
    print("\nCross-Validation Complete.")

    # Calculate Overall OOF Score
    overall_mae = get_score(y_full, oof_preds)
    print(f"Overall CV MAE (Vision Branch): {overall_mae}")

    # Average Test Predictions
    avg_test_preds = np.mean(test_preds_accum, axis=1)

    # Create DataFrames
    df_oof = pd.DataFrame(
        {"segment_id": df_full["segment_id"].values, "time_to_eruption": oof_preds}
    )

    df_test_pred = pd.DataFrame(
        {"segment_id": df_test["segment_id"].values, "time_to_eruption": avg_test_preds}
    )

    # Save to Working Directory
    oof_path = os.path.join(working_dir, "vision_oof.csv")
    test_pred_path = os.path.join(working_dir, "vision_test_preds.csv")

    df_oof.to_csv(oof_path, index=False)
    df_test_pred.to_csv(test_pred_path, index=False)

    print(f"Saved OOF predictions to {oof_path}")
    print(f"Saved Test predictions to {test_pred_path}")

    return df_oof, df_test_pred
