import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import PathConfig, TrainingConfig, FeatureConfig
from library.utils import seed_everything
from library.data_factory import load_and_preprocess, get_pytorch_dataloaders
from library.training_engine import train_rf, train_mlp, predict_mlp, train_stacker

# --- Configuration Overrides for Fast Baseline ---
TrainingConfig.EPOCHS = 15  # Reduce epochs for speed (default 30)
TrainingConfig.PATIENCE = 4  # Reduce patience


def prepare_rf_data(data_dict):
    """Concatenates features for RF: TFIDF + Community + Meta/Num"""
    return np.hstack(
        [data_dict["tfidf"], data_dict["community"], data_dict["meta_num"]]
    )


def slice_dict(data_dict, indices):
    """Slices all arrays in the dictionary based on indices."""
    sliced = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            sliced[k] = v[indices]
        else:
            sliced[k] = v
    return sliced


def run():
    # 1. Setup
    seed_everything(TrainingConfig.SEED)
    print("Starting execution...")

    # 2. Data Loading
    # load_cached_data=True will use the files in ./working/idea_5/ if they exist
    print("Loading data...")
    stream_a, stream_b = load_and_preprocess(load_cached_data=True)

    # Unpack Data
    # Stream A: RF (Sparse/Dense)
    data_a_train_full, data_a_val_holdout, data_a_test = stream_a
    # Stream B: MLP (Dense/Seq)
    data_b_train_full, data_b_val_holdout, data_b_test = stream_b

    # Targets for CV
    y_full = data_a_train_full["y"]
    N_train = len(y_full)

    # Prepare Holdout Validation Data
    X_rf_val_holdout = prepare_rf_data(data_a_val_holdout)
    y_val_holdout = data_a_val_holdout["y"]

    # Prepare Test Data
    X_rf_test = prepare_rf_data(data_a_test)
    ids_test = data_a_test["ids"]

    # Prepare Loaders for Holdout and Test (MLP)
    # We create these once; they don't change per fold
    _, loader_mlp_val_holdout, loader_mlp_test = get_pytorch_dataloaders(
        data_b_train_full,  # dummy
        data_b_val_holdout,
        data_b_test,
        batch_size=TrainingConfig.BATCH_SIZE,
    )

    # 3. Cross-Validation Initialization
    k_folds = TrainingConfig.NUM_FOLDS
    skf = StratifiedKFold(
        n_splits=k_folds, shuffle=True, random_state=TrainingConfig.SEED
    )

    # Storage for OOF predictions (for Stacker training)
    oof_preds_rf = np.zeros(N_train)
    oof_preds_mlp = np.zeros(N_train)

    # Storage for Averaged Predictions on Holdout (for Validation)
    val_holdout_preds_rf_accum = np.zeros(len(y_val_holdout))
    val_holdout_preds_mlp_accum = np.zeros(len(y_val_holdout))

    # Storage for Averaged Predictions on Test (for Submission)
    test_preds_rf_accum = np.zeros(len(ids_test))
    test_preds_mlp_accum = np.zeros(len(ids_test))

    print(f"Starting {k_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N_train), y_full)):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")

        # --- Prepare Fold Data ---
        # RF Data
        X_rf_full = prepare_rf_data(data_a_train_full)
        X_rf_fold_train = X_rf_full[train_idx]
        y_fold_train = y_full[train_idx]
        X_rf_fold_val = X_rf_full[val_idx]
        y_fold_val = y_full[val_idx]

        # MLP Data
        dict_b_fold_train = slice_dict(data_b_train_full, train_idx)
        dict_b_fold_val = slice_dict(data_b_train_full, val_idx)

        # Create loaders for this fold
        loader_fold_train, loader_fold_val, _ = get_pytorch_dataloaders(
            dict_b_fold_train, dict_b_fold_val, data_b_test
        )

        # --- Train Base Learner A: Random Forest ---
        rf_model = train_rf(X_rf_fold_train, y_fold_train, X_rf_fold_val, y_fold_val)

        # Predict OOF
        oof_preds_rf[val_idx] = rf_model.predict_proba(X_rf_fold_val)[:, 1]
        # Predict Holdout & Test (Accumulate)
        val_holdout_preds_rf_accum += rf_model.predict_proba(X_rf_val_holdout)[:, 1]
        test_preds_rf_accum += rf_model.predict_proba(X_rf_test)[:, 1]

        # --- Train Base Learner B: Triple-Branch MLP ---
        meta_dim = dict_b_fold_train["meta_num"].shape[1]
        mlp_model, _ = train_mlp(loader_fold_train, loader_fold_val, meta_dim)

        # Predict OOF
        oof_preds_mlp[val_idx] = predict_mlp(mlp_model, loader_fold_val)
        # Predict Holdout & Test (Accumulate)
        val_holdout_preds_mlp_accum += predict_mlp(mlp_model, loader_mlp_val_holdout)
        test_preds_mlp_accum += predict_mlp(mlp_model, loader_mlp_test)

    # Average predictions across folds
    val_holdout_preds_rf_avg = val_holdout_preds_rf_accum / k_folds
    val_holdout_preds_mlp_avg = val_holdout_preds_mlp_accum / k_folds

    test_preds_rf_avg = test_preds_rf_accum / k_folds
    test_preds_mlp_avg = test_preds_mlp_accum / k_folds

    # 4. Meta-Learner (Stacking)
    print("\n=== Training Meta-Learner (Stacker) ===")

    # Train Stacker on OOF predictions from the training set
    X_meta_train = np.column_stack([oof_preds_rf, oof_preds_mlp])
    stacker = train_stacker(X_meta_train, y_full)

    print(
        f"Stacker Coefficients: RF={stacker.coef_[0][0]:.4f}, MLP={stacker.coef_[0][1]:.4f}"
    )

    # 5. Final Validation on Holdout Set
    X_meta_val_holdout = np.column_stack(
        [val_holdout_preds_rf_avg, val_holdout_preds_mlp_avg]
    )
    val_final_probs = stacker.predict_proba(X_meta_val_holdout)[:, 1]

    final_val_auc = roc_auc_score(y_val_holdout, val_final_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load original validation dataframe to get features for correlation
    df_val = pd.read_csv(PathConfig.VAL_CSV)

    # Calculate Error Magnitude
    # y_val_holdout is numpy array of 0/1, val_final_probs is float 0..1
    errors = np.abs(y_val_holdout - val_final_probs)
    df_val["prediction_error"] = errors

    # Select numerical columns for correlation
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and error itself from features
    if "requester_received_pizza" in numeric_cols:
        numeric_cols.remove("requester_received_pizza")
    if "prediction_error" in numeric_cols:
        numeric_cols.remove("prediction_error")

    print("Correlation between Prediction Error and Input Features:")
    correlations = {}
    for col in numeric_cols:
        # Handle potential NaNs in raw data
        series = df_val[col].fillna(0)
        corr = series.corr(df_val["prediction_error"])
        correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, score in sorted_corr[:10]:
        print(f"{name:<50}: {score:.4f}")

    # 7. Submission
    threshold = 0.6789999838498684
    if final_val_auc > threshold:
        print("\nValidation metric meets threshold. Generating submission...")

        X_meta_test = np.column_stack([test_preds_rf_avg, test_preds_mlp_avg])
        final_test_probs = stacker.predict_proba(X_meta_test)[:, 1]

        df_sub = pd.DataFrame(
            {"request_id": ids_test, "requester_received_pizza": final_test_probs}
        )

        os.makedirs(PathConfig.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(PathConfig.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {PathConfig.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric {final_val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
