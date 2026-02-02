import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, expm1_target
from library.feature_engineering import FeatureEngineer
from library.dataset import SeismicDataset
from library.models import ScalarFusedEfficientNet, LightGBMWrapper
from library.train_eval import train_lgbm_fold, train_cnn_fold

# ==========================================
# 1. Configuration Patching for Fast Baseline
# ==========================================
# Modify Config attributes to meet "fast baseline" requirements
Config.EPOCHS = 10  # Reduced from 35 for speed
Config.LGB_PARAMS["n_estimators"] = 1000  # Reduced from 5000 for speed
Config.NUM_WORKERS = 10  # Utilize available vCPUs
Config.DEBUG = False  # Ensure full dataset is used for valid metric calculation


# ==========================================
# 2. Helper Functions
# ==========================================
def predict_cnn_custom(metadata, data_dir, fold_idx, mode="test"):
    """
    Custom inference function for CNN to handle flexible data directories (val vs test).
    """
    device = Config.DEVICE
    model_path = os.path.join(Config.WORKING_DIR, f"cnn_fold_{fold_idx}.pth")

    # Create Dataset and Loader
    dataset = SeismicDataset(metadata, data_dir, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # Load Model
    model = ScalarFusedEfficientNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for images, scalars, _ in loader:
            images = images.to(device)
            scalars = scalars.to(device)

            outputs = model(images, scalars)
            preds_log = outputs.cpu().numpy().flatten()
            preds_raw = expm1_target(preds_log)
            preds.extend(preds_raw)

    return np.array(preds)


def main():
    seed_everything(Config.SEED)
    print("Starting execution...")

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    print("Step 1: Feature Engineering")
    fe = FeatureEngineer()

    # Tabular Features
    print("Processing Tabular Data...")
    df_train_feats = fe.process_tabular("train", load_cached_data=True)
    df_val_feats = fe.process_tabular("val", load_cached_data=True)
    df_test_feats = fe.process_tabular("test", load_cached_data=True)

    # Vision Features
    print("Processing Vision Data...")
    fe.process_vision("train", load_cached_data=True)
    fe.process_vision("val", load_cached_data=True)
    fe.process_vision("test", load_cached_data=True)

    # ==========================================
    # 4. Data Loading & Alignment
    # ==========================================
    print("Step 2: Loading Metadata and Aligning Data")
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Align tabular features with metadata order
    df_train_feats = (
        df_train_feats.set_index("segment_id")
        .reindex(df_train_meta["segment_id"])
        .reset_index()
    )
    df_val_feats = (
        df_val_feats.set_index("segment_id")
        .reindex(df_val_meta["segment_id"])
        .reset_index()
    )
    df_test_feats = (
        df_test_feats.set_index("segment_id")
        .reindex(df_test_meta["segment_id"])
        .reset_index()
    )

    # Prepare Tabular Arrays
    drop_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df_train_feats.columns if c not in drop_cols]

    X_train = df_train_feats[feature_cols]
    y_train = df_train_feats["time_to_eruption"]
    X_val = df_val_feats[feature_cols]
    y_val = df_val_feats["time_to_eruption"]
    X_test = df_test_feats[feature_cols]

    # ==========================================
    # 5. Cross-Validation Training (Train Set)
    # ==========================================
    print("Step 3: Training Models (5-Fold CV)")
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # OOF Arrays for Train
    oof_lgbm = np.zeros(len(df_train_meta))
    oof_cnn = np.zeros(len(df_train_meta))

    # Prediction Arrays for Hold-out Val and Test
    val_preds_lgbm_folds = np.zeros((Config.N_FOLDS, len(df_val_meta)))
    val_preds_cnn_folds = np.zeros((Config.N_FOLDS, len(df_val_meta)))

    test_preds_lgbm_folds = np.zeros((Config.N_FOLDS, len(df_test_meta)))
    test_preds_cnn_folds = np.zeros((Config.N_FOLDS, len(df_test_meta)))

    for fold, (train_idx, cv_val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n--- FOLD {fold} ---")

        # --- Branch A: LightGBM ---
        X_tr_fold, y_tr_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_cv_fold, y_cv_fold = X_train.iloc[cv_val_idx], y_train.iloc[cv_val_idx]

        # Train
        lgbm_model, lgbm_cv_preds = train_lgbm_fold(
            X_tr_fold, y_tr_fold, X_cv_fold, y_cv_fold, fold
        )
        oof_lgbm[cv_val_idx] = lgbm_cv_preds

        # Inference on Hold-out Val & Test
        val_preds_lgbm_folds[fold] = lgbm_model.predict(X_val)
        test_preds_lgbm_folds[fold] = lgbm_model.predict(X_test)

        # --- Branch B: CNN ---
        train_meta_fold = df_train_meta.iloc[train_idx].copy()
        cv_meta_fold = df_train_meta.iloc[cv_val_idx].copy()

        # Train (returns OOF preds for the CV fold)
        cnn_cv_preds = train_cnn_fold(train_meta_fold, cv_meta_fold, fold)
        oof_cnn[cv_val_idx] = cnn_cv_preds

        # Inference on Hold-out Val & Test
        val_preds_cnn_folds[fold] = predict_cnn_custom(
            df_val_meta, Config.SPECTROGRAM_VAL_DIR, fold, mode="val"
        )
        test_preds_cnn_folds[fold] = predict_cnn_custom(
            df_test_meta, Config.SPECTROGRAM_TEST_DIR, fold, mode="test"
        )

    # ==========================================
    # 6. Meta-Learner Training
    # ==========================================
    print("\nStep 4: Training Meta-Learner")
    X_meta_train = np.column_stack([oof_lgbm, oof_cnn])

    meta_model = Ridge(alpha=Config.META_MODEL_ALPHA, random_state=Config.SEED)
    meta_model.fit(X_meta_train, y_train)

    print(
        f"Meta-Learner Coefs: LGBM={meta_model.coef_[0]:.4f}, CNN={meta_model.coef_[1]:.4f}"
    )

    # ==========================================
    # 7. Evaluation on Hold-out Validation Set
    # ==========================================
    print("\nStep 5: Evaluation on Hold-out Validation Set")

    # Average predictions across folds
    avg_val_lgbm = np.mean(val_preds_lgbm_folds, axis=0)
    avg_val_cnn = np.mean(val_preds_cnn_folds, axis=0)

    # Stack
    X_meta_val = np.column_stack([avg_val_lgbm, avg_val_cnn])

    # Predict
    final_val_preds = meta_model.predict(X_meta_val)
    final_val_preds = np.maximum(0, final_val_preds)  # Clip negatives

    # Metric
    val_mae = mean_absolute_error(y_val, final_val_preds)
    print(f"Final Validation Metric: {val_mae}")

    # ==========================================
    # 8. Failure Analysis
    # ==========================================
    print("\nStep 6: Failure Analysis")
    errors = np.abs(y_val - final_val_preds)

    # Calculate correlations between error magnitude and features
    corrs = {}
    for col in X_val.columns:
        # Skip constant columns to avoid warnings
        if X_val[col].std() > 1e-9:
            corrs[col] = np.corrcoef(errors, X_val[col])[0, 1]

    # Sort by absolute correlation
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corrs[:5]:
        print(f"{name}: {val:.4f}")

    # ==========================================
    # 9. Submission Generation
    # ==========================================
    THRESHOLD = 1920624.12
    if val_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({val_mae:.2f}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Average Test Predictions
        avg_test_lgbm = np.mean(test_preds_lgbm_folds, axis=0)
        avg_test_cnn = np.mean(test_preds_cnn_folds, axis=0)

        # Stack
        X_meta_test = np.column_stack([avg_test_lgbm, avg_test_cnn])

        # Predict
        final_test_preds = meta_model.predict(X_meta_test)
        final_test_preds = np.maximum(0, final_test_preds)

        # Save
        submission_df = pd.DataFrame(
            {
                "segment_id": df_test_meta["segment_id"],
                "time_to_eruption": final_test_preds,
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric ({val_mae:.2f}) >= Threshold ({THRESHOLD}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
