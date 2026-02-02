import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import library.config as config
import library.feature_engineering as fe
import library.model_handler as mh


def run_cross_validation(load_cached_data=True, debug_size=None):
    """
    Orchestrates the Stratified K-Fold Cross-Validation training.
    Loads data, trains models per fold, saves them, and calculates OOF MAE.
    """
    print("Loading training and validation data...")
    # Load separate splits
    df_train_part = fe.get_train_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )
    df_val_part = fe.get_val_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # Concatenate for full CV
    df_full_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Feature selection
    target_col = "time_to_eruption"
    id_col = "segment_id"
    exclude_cols = [id_col, target_col]
    feature_cols = [c for c in df_full_train.columns if c not in exclude_cols]

    X = df_full_train[feature_cols]
    y = df_full_train[target_col].values

    # Stratification bins
    # We bin the continuous target to perform stratified splitting
    num_bins = 10
    if len(y) < num_bins:
        num_bins = 2  # Safety fallback for small debug sizes

    try:
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback if distribution is too singular
        y_bins = np.zeros(len(y))

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    oof_preds = np.zeros(len(X))

    print(f"Starting {config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n{'='*20} Fold {fold + 1} / {config.N_FOLDS} {'='*20}")

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Train
        model = mh.train_fold_model(X_tr, y_tr, X_va, y_va, fold)

        # Save model
        # We save the model using the best iteration found by early stopping.
        # This ensures that when we load it later, we don't need to track best_iteration manually.
        model_path = os.path.join(config.WORKING_DIR, f"lgbm_fold_{fold}.txt")
        model.save_model(model_path, num_iteration=model.best_iteration)
        print(f"Model for Fold {fold + 1} saved to {model_path}")

        # Validation Inference
        # We use predict_batch from model_handler which uses best_iteration
        val_pred = mh.predict_batch(model, X_va)
        oof_preds[val_idx] = val_pred

        # Metric
        fold_mae = np.mean(np.abs(y_va - val_pred))
        print(f"Fold {fold + 1} MAE: {fold_mae}")

    # Overall Metric
    total_mae = np.mean(np.abs(y - oof_preds))
    print("\n" + "=" * 40)
    print(f"Overall CV MAE: {total_mae}")
    print("=" * 40)

    return total_mae


def generate_test_predictions(load_cached_data=True, debug_size=None):
    """
    Generates predictions for the test set using the ensemble of saved models.
    """
    print("Loading test data...")
    df_test = fe.get_test_data(load_cached_data=load_cached_data, debug_size=debug_size)

    id_col = "segment_id"
    # Ensure we use the same features as training.
    # We exclude ID and target (if it existed) to get feature columns.
    exclude_cols = [id_col, "time_to_eruption"]
    feature_cols = [c for c in df_test.columns if c not in exclude_cols]

    X_test = df_test[feature_cols]
    test_ids = df_test[id_col].values

    test_preds_accum = np.zeros((len(X_test), config.N_FOLDS))

    print("Generating predictions with ensemble...")

    for fold in range(config.N_FOLDS):
        model_path = os.path.join(config.WORKING_DIR, f"lgbm_fold_{fold}.txt")
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping fold.")
            continue

        print(f"Loading model from {model_path}...")
        model = lgb.Booster(model_file=model_path)

        # Predict
        # Since we saved the model with `num_iteration=best_iteration`,
        # the file contains only the optimal trees. We can predict using all trees in the file.
        preds = model.predict(X_test)
        test_preds_accum[:, fold] = preds

    # Average predictions (Bagging)
    final_test_preds = np.mean(test_preds_accum, axis=1)

    # Submission
    submission = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": final_test_preds}
    )

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
