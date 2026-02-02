import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import seed_everything, load_parquet, load_npy, get_score


def run_lgbm_cv(debug=False):
    """
    Orchestrates the 5-Fold Cross-Validation for the Tabular Branch (LightGBM).
    Loads features, trains models, generates OOF and Test predictions.

    This function implements the 'Latent-Source Cepstral Stacking' strategy for the
    tabular branch, utilizing the extensive feature set derived from raw sensors
    and the PCA-based virtual source.

    Args:
        debug (bool): If True, runs on a small subset of data for testing purposes.

    Returns:
        tuple: (df_oof, df_test_pred) - DataFrames containing OOF and Test predictions.
    """
    config = Config()
    seed_everything(config.SEED)

    print("Initializing Tabular Branch (LightGBM) Cross-Validation...")

    # ==========================================
    # 1. Load and Prepare Data
    # ==========================================
    working_dir = config.WORKING_DIR

    # Input Paths
    train_feat_path = os.path.join(working_dir, "train_features.parquet")
    train_target_path = os.path.join(working_dir, "train_targets.npy")

    val_feat_path = os.path.join(working_dir, "val_features.parquet")
    val_target_path = os.path.join(working_dir, "val_targets.npy")

    test_feat_path = os.path.join(working_dir, "test_features.parquet")

    # Check existence of cached files
    if not (
        os.path.exists(train_feat_path)
        and os.path.exists(val_feat_path)
        and os.path.exists(test_feat_path)
    ):
        raise FileNotFoundError(
            "Cached tabular features not found. Run feature_engineering.py first."
        )

    # Load Data
    print("Loading cached tabular data...")
    df_train_part = load_parquet(train_feat_path)
    y_train_part = load_npy(train_target_path)

    df_val_part = load_parquet(val_feat_path)
    y_val_part = load_npy(val_target_path)

    df_test = load_parquet(test_feat_path)

    # Concatenate Train and Val to form the full development set for CV
    # We reset index to ensure KFold indexing works correctly
    df_full = pd.concat([df_train_part, df_val_part], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    # Identify Feature Columns (exclude ID and Target placeholders)
    # Note: 'time_to_eruption' is not in features df usually, but checking for safety
    feature_cols = [
        c for c in df_full.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    print(f"Number of features: {len(feature_cols)}")

    X_full = df_full[feature_cols].values
    X_test = df_test[feature_cols].values

    # Debug Mode: Subsample data to verify pipeline speed
    if debug:
        print("DEBUG MODE: Subsampling data...")
        subset_size = 100
        X_full = X_full[:subset_size]
        y_full = y_full[:subset_size]
        df_full = df_full.iloc[:subset_size]
        X_test = X_test[:subset_size]
        df_test = df_test.iloc[:subset_size]
        # Reduce estimators for quick debug
        config.LGBM_PARAMS["n_estimators"] = 50

    print(f"Full Train Shape: {X_full.shape}, Test Shape: {X_test.shape}")

    # ==========================================
    # 2. Cross-Validation Loop
    # ==========================================
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)

    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros((len(X_test), 5))

    # Prepare Hyperparameters
    params = config.LGBM_PARAMS.copy()

    # Extract early stopping rounds to pass to callback
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1} / 5 ---")

        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create LightGBM Datasets
        lgb_train = lgb.Dataset(X_train_fold, y_train_fold, feature_name=feature_cols)
        lgb_val = lgb.Dataset(
            X_val_fold, y_val_fold, feature_name=feature_cols, reference=lgb_train
        )

        # Callbacks for Early Stopping and Logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        # Train Model
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Predict OOF (using best iteration)
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        # Enforce non-negative predictions
        val_pred = np.maximum(0, val_pred)
        oof_preds[val_idx] = val_pred

        # Calculate and Print Fold Score
        fold_score = get_score(y_val_fold, val_pred)
        print(f"Fold {fold + 1} MAE: {fold_score}")

        # Predict Test (accumulate)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_pred = np.maximum(0, test_pred)
        test_preds_accum[:, fold] = test_pred

        # Save Model Artifact (Text dump)
        model_path = os.path.join(working_dir, f"lgb_model_fold_{fold}.txt")
        model.save_model(model_path)

    # ==========================================
    # 3. Aggregate and Save Results
    # ==========================================
    print("\nCross-Validation Complete.")

    # Overall Metric
    overall_mae = get_score(y_full, oof_preds)
    print(f"Overall CV MAE (Tabular Branch): {overall_mae}")

    # Average Test Predictions across folds
    avg_test_preds = np.mean(test_preds_accum, axis=1)

    # Construct Output DataFrames
    df_oof = pd.DataFrame(
        {"segment_id": df_full["segment_id"].values, "time_to_eruption": oof_preds}
    )

    df_test_pred = pd.DataFrame(
        {"segment_id": df_test["segment_id"].values, "time_to_eruption": avg_test_preds}
    )

    # Define Output Paths
    oof_path = os.path.join(working_dir, "tabular_oof.csv")
    test_pred_path = os.path.join(working_dir, "tabular_test_preds.csv")

    # Save CSVs
    df_oof.to_csv(oof_path, index=False)
    df_test_pred.to_csv(test_pred_path, index=False)

    print(f"Saved OOF predictions to {oof_path}")
    print(f"Saved Test predictions to {test_pred_path}")

    return df_oof, df_test_pred
