import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings

from library.config import Config
from library.utils import seed_everything, calculate_mae
from library.features import generate_feature_matrix

# Suppress warnings
warnings.filterwarnings("ignore")


def run_lgbm_cv(load_cached_data=True):
    """
    Executes the LightGBM training pipeline with 5-Fold Cross-Validation.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.

    Returns:
        pd.DataFrame: DataFrame containing 'segment_id' and 'time_to_eruption' predictions for the test set.
    """
    seed_everything(Config.SEED)

    print("Initializing LightGBM Branch...")

    # ---------------------------------------------------------
    # 1. Load Features
    # ---------------------------------------------------------
    # We load features for train, val, and test.
    # Note: We combine train and val metadata to perform a full 5-Fold CV
    # as per the strategy to maximize robustness.

    print("Loading/Generating feature matrices...")
    df_train = generate_feature_matrix(
        Config.TRAIN_METADATA, load_cached_data=load_cached_data, split_name="train"
    )

    df_val = generate_feature_matrix(
        Config.VAL_METADATA, load_cached_data=load_cached_data, split_name="val"
    )

    df_test = generate_feature_matrix(
        Config.TEST_METADATA, load_cached_data=load_cached_data, split_name="test"
    )

    # Combine train and val for CV
    df_full_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # ---------------------------------------------------------
    # 2. Prepare Data
    # ---------------------------------------------------------
    # Exclude metadata columns to get feature list
    exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df_full_train.columns if c not in exclude_cols]

    print(
        f"Training with {len(feature_cols)} features on {len(df_full_train)} samples."
    )

    X = df_full_train[feature_cols].values
    y = df_full_train["time_to_eruption"].values

    X_test = df_test[feature_cols].values
    test_ids = df_test["segment_id"].values

    # ---------------------------------------------------------
    # 3. Cross-Validation Loop
    # ---------------------------------------------------------
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))

    # Prepare params
    params = Config.LGBM_PARAMS.copy()
    # Extract callbacks parameters from dict if they exist, to pass to callbacks explicitly
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)
    verbose_eval = params.pop("verbose", -1)

    # Ensure verbosity is controlled in params
    params["verbose"] = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Create LGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # Silence log evaluation
        ]

        # Train
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Predict
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        fold_test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds_accum += fold_test_preds

        # Evaluate
        fold_mae = calculate_mae(y_val, val_preds)
        print(f"Fold {fold + 1} MAE: {fold_mae}")

    # ---------------------------------------------------------
    # 4. Aggregation & Results
    # ---------------------------------------------------------
    # Average test predictions
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # Overall CV Score
    total_mae = calculate_mae(y, oof_preds)
    print(f"\nOverall CV MAE: {total_mae}")

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": avg_test_preds}
    )

    # Ensure segment_id is integer
    submission_df["segment_id"] = submission_df["segment_id"].astype(int)

    return submission_df
