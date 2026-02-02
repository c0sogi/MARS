import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import library.config as config
import library.feature_engineering as fe


def train_fold_model(X_train, y_train, X_val, y_val, fold_idx):
    """
    Trains a high-capacity LightGBM model for a single fold.

    Args:
        X_train, y_train: Training features and target.
        X_val, y_val: Validation features and target.
        fold_idx: Index of the current fold (for logging).

    Returns:
        Trained LightGBM Booster.
    """
    # Create LightGBM Datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Load parameters from config
    params = config.LGBM_PARAMS.copy()

    # Setup callbacks for Early Stopping and Logging
    # We monitor 'l1' (MAE) for early stopping as per requirements,
    # even though the objective is 'l2' (MSE).
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=config.EARLY_STOPPING_ROUNDS, first_metric_only=True
        ),
        lgb.log_evaluation(period=100),
    ]

    # Train the model
    # eval_metric='l1' ensures MAE is calculated and used for early stopping
    params["metric"] = "l1"
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def predict_batch(model, X):
    """
    Generates predictions for a batch of data using the best iteration of the model.
    """
    return model.predict(X, num_iteration=model.best_iteration)


def train_and_predict(load_cached_data=True, debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Main driver function to execute the Homogeneous Bagging Ensemble strategy.

    1. Loads and combines Train/Val data.
    2. Performs Stratified K-Fold CV.
    3. Trains 5 High-Capacity LightGBM models.
    4. Aggregates predictions on Test set.
    5. Saves submission.
    """
    print("Loading feature datasets...")
    # Load all available data using the feature engineering library
    df_train_part = fe.get_train_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )
    df_val_part = fe.get_val_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )
    df_test = fe.get_test_data(load_cached_data=load_cached_data, debug_size=debug_size)

    # Combine provided train and val splits to maximize data for Cross-Validation
    df_full_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Define columns
    target_col = "time_to_eruption"
    id_col = "segment_id"
    exclude_cols = [id_col, target_col]

    feature_cols = [c for c in df_full_train.columns if c not in exclude_cols]

    # Prepare X and y
    X = df_full_train[feature_cols]
    y = df_full_train[target_col].values

    # Prepare Test data
    X_test = df_test[feature_cols]
    test_ids = df_test[id_col].values

    # Stratified K-Fold Setup
    # Since target is continuous, we bin it into quantiles to perform stratification
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

    # Arrays to store results
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros((len(X_test), config.N_FOLDS))

    print(f"Starting {config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n{'='*20} Fold {fold + 1} / {config.N_FOLDS} {'='*20}")

        # Split Data
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Train
        model = train_fold_model(X_tr, y_tr, X_va, y_va, fold)

        # Validation Inference
        val_pred = predict_batch(model, X_va)
        oof_preds[val_idx] = val_pred

        # Test Inference (Accumulate for Bagging)
        test_preds_accum[:, fold] = predict_batch(model, X_test)

        # Metric
        fold_mae = np.mean(np.abs(y_va - val_pred))
        print(f"Fold {fold + 1} MAE: {fold_mae}")

    # Final Evaluation
    print("\n" + "=" * 40)
    total_mae = np.mean(np.abs(y - oof_preds))
    print(f"Overall CV MAE: {total_mae}")
    print("=" * 40)

    # Aggregate Test Predictions (Mean)
    final_test_preds = np.mean(test_preds_accum, axis=1)

    # Generate Submission
    submission = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": final_test_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)

    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
