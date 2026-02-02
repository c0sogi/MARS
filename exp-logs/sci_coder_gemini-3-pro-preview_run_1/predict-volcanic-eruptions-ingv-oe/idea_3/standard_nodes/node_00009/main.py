import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import lightgbm as lgb
import copy
import warnings

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_mae
from library.features import generate_feature_matrix
from library.dataset import get_dataset, SeismicDataset
from library.model_resnet import ResNet1D, train_one_epoch, validate, predict

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    seed_everything(Config.SEED)

    # Configure for Fast Baseline execution
    Config.EPOCHS = 5  # Reduce epochs for speed (Baseline)
    Config.LGBM_PARAMS["n_estimators"] = 2000  # Limit boosting rounds
    Config.LGBM_PARAMS["early_stopping_rounds"] = 50

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading Tabular Features (Branch A)...")
    # Load features for all splits
    df_train_feat = generate_feature_matrix(Config.TRAIN_METADATA, split_name="train")
    df_val_feat = generate_feature_matrix(Config.VAL_METADATA, split_name="val")
    df_test_feat = generate_feature_matrix(Config.TEST_METADATA, split_name="test")

    # Combine Train and Val for robust Cross-Validation
    df_full_feat = pd.concat([df_train_feat, df_val_feat], axis=0, ignore_index=True)

    print("Loading Raw Sensor Data (Branch B)...")
    # Load raw data for all splits
    ds_train_raw = get_dataset(Config.TRAIN_METADATA, "train")
    ds_val_raw = get_dataset(Config.VAL_METADATA, "val")
    ds_test_raw = get_dataset(Config.TEST_METADATA, "test")

    # Combine Train and Val for robust Cross-Validation
    X_raw_full = np.concatenate([ds_train_raw.data, ds_val_raw.data], axis=0)
    y_raw_full = np.concatenate([ds_train_raw.targets, ds_val_raw.targets], axis=0)

    # Verify alignment between tabular and raw data
    if len(df_full_feat) != len(X_raw_full):
        raise ValueError(
            f"Data mismatch: Features {len(df_full_feat)} vs Raw {len(X_raw_full)}"
        )

    # ---------------------------------------------------------
    # 3. Cross-Validation Loop
    # ---------------------------------------------------------
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Prepare Feature Data
    exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df_full_feat.columns if c not in exclude_cols]

    X_feat = df_full_feat[feature_cols].values
    y_feat = df_full_feat["time_to_eruption"].values
    X_test_feat = df_test_feat[feature_cols].values

    # Prepare Raw Test Data Loader
    test_ds = SeismicDataset(ds_test_raw.data, None)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Storage for predictions
    oof_preds_lgbm = np.zeros(len(X_feat))
    test_preds_lgbm = np.zeros(len(X_test_feat))

    oof_preds_resnet = np.zeros(len(X_raw_full))
    test_preds_resnet = np.zeros(len(ds_test_raw.data))

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_feat, y_feat)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # ==========================
        # Branch A: LightGBM
        # ==========================
        print("Training LightGBM...")
        X_tr_l, y_tr_l = X_feat[train_idx], y_feat[train_idx]
        X_val_l, y_val_l = X_feat[val_idx], y_feat[val_idx]

        dtrain = lgb.Dataset(X_tr_l, label=y_tr_l)
        dval = lgb.Dataset(X_val_l, label=y_val_l, reference=dtrain)

        # Prepare params
        params = Config.LGBM_PARAMS.copy()
        es_rounds = params.pop("early_stopping_rounds", 50)
        params["verbose"] = -1

        callbacks = [
            lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model_lgbm = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Inference
        oof_preds_lgbm[val_idx] = model_lgbm.predict(
            X_val_l, num_iteration=model_lgbm.best_iteration
        )
        test_preds_lgbm += (
            model_lgbm.predict(X_test_feat, num_iteration=model_lgbm.best_iteration)
            / Config.N_FOLDS
        )

        lgbm_mae = calculate_mae(y_val_l, oof_preds_lgbm[val_idx])
        print(f"LGBM Fold MAE: {lgbm_mae:.4f}")

        # ==========================
        # Branch B: 1D-ResNet
        # ==========================
        print("Training ResNet...")
        X_tr_r, y_tr_r = X_raw_full[train_idx], y_raw_full[train_idx]
        X_val_r, y_val_r = X_raw_full[val_idx], y_raw_full[val_idx]

        train_ds = SeismicDataset(X_tr_r, y_tr_r)
        val_ds = SeismicDataset(X_val_r, y_val_r)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        model_resnet = ResNet1D().to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(
            model_resnet.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        best_mae = float("inf")
        best_wts = copy.deepcopy(model_resnet.state_dict())

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model_resnet, train_loader, criterion, optimizer, device
            )
            val_loss, val_mae = validate(model_resnet, val_loader, criterion, device)
            scheduler.step()

            if val_mae < best_mae:
                best_mae = val_mae
                best_wts = copy.deepcopy(model_resnet.state_dict())

        # Load best weights
        model_resnet.load_state_dict(best_wts)

        # Inference
        oof_preds_resnet[val_idx] = predict(model_resnet, val_loader, device)
        test_preds_resnet += predict(model_resnet, test_loader, device) / Config.N_FOLDS

        print(f"ResNet Fold MAE: {best_mae:.4f}")

    # ---------------------------------------------------------
    # 4. Ensemble & Evaluation
    # ---------------------------------------------------------
    # Simple Weighted Average (0.5 / 0.5)
    oof_preds_ensemble = (oof_preds_lgbm + oof_preds_resnet) / 2
    final_mae = calculate_mae(y_feat, oof_preds_ensemble)

    print(f"Final Validation Metric: {final_mae}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_feat - oof_preds_ensemble)

    # Use feature matrix for correlation analysis
    analysis_df = pd.DataFrame(X_feat, columns=feature_cols)
    analysis_df["error_magnitude"] = errors

    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .abs()
        .sort_values(ascending=False)
    )
    print("Top 5 Features Correlated with Error Magnitude:")
    print(correlations.head(5))

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 3135965.05
    if final_mae < THRESHOLD:
        print(
            f"\nValidation metric ({final_mae:.2f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        test_preds_ensemble = (test_preds_lgbm + test_preds_resnet) / 2

        submission_df = pd.DataFrame(
            {
                "segment_id": df_test_feat["segment_id"].astype(int),
                "time_to_eruption": test_preds_ensemble,
            }
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_mae:.2f}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
