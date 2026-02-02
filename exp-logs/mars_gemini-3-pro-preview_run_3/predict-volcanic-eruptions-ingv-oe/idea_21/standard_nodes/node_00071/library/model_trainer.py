import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import os
import library.config as config


def get_feature_columns(df):
    """
    Identifies feature columns by excluding metadata and target columns.
    """
    excluded_cols = ["segment_id", "time_to_eruption", "target_bin", "file_path"]
    return [c for c in df.columns if c not in excluded_cols]


def train_model_cv(train_df):
    """
    Performs Stratified K-Fold Cross-Validation using LightGBM.

    Implements Loss-Metric Decoupling:
    - Optimizes L2 Loss (defined in config.LGBM_PARAMS['objective'])
    - Evaluates on MAE (defined in config.LGBM_PARAMS['metric'])

    Args:
        train_df (pd.DataFrame): DataFrame containing features and 'time_to_eruption'.

    Returns:
        tuple: (models, oof_preds, scores)
            - models: List of trained LightGBM models.
            - oof_preds: Series containing Out-Of-Fold predictions.
            - scores: List of MAE scores for each fold.
    """
    # Prepare features and target
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols]
    y = train_df["time_to_eruption"]

    # Create bins for StratifiedKFold since target is continuous
    # We use qcut to create bins with equal number of samples
    num_bins = 15
    # Handle potential duplicate edges by dropping them
    target_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    # Initialize Cross-Validation
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    models = []
    oof_preds = np.zeros(len(train_df))
    scores = []

    print(f"Starting training with {config.N_FOLDS} folds...")
    print(f"Features used: {len(feature_cols)}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_bins)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize model with params from config
        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)

        # Callbacks for Early Stopping and Logging
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=config.VERBOSE_EVAL),
        ]

        # Train
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["valid"],
            callbacks=callbacks,
        )

        # Predict
        val_preds = model.predict(X_val)

        # Ensure non-negative predictions (time cannot be negative)
        val_preds = np.maximum(val_preds, 0)

        # Store OOF predictions
        oof_preds[val_idx] = val_preds
        models.append(model)

        # Calculate Metric
        fold_mae = mean_absolute_error(y_val, val_preds)
        scores.append(fold_mae)

        print(f"Fold {fold+1} MAE: {fold_mae}")

    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"Overall OOF MAE: {overall_mae}")

    return models, oof_preds, scores


def generate_submission(models, test_df):
    """
    Generates predictions for the test set using the trained models and saves to CSV.

    Args:
        models (list): List of trained LightGBM models.
        test_df (pd.DataFrame): DataFrame containing test features.
    """
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols]

    # Generate predictions from all models
    all_preds = []
    for model in models:
        preds = model.predict(X_test)
        all_preds.append(preds)

    # Average predictions
    avg_preds = np.mean(all_preds, axis=0)

    # Ensure non-negative
    avg_preds = np.maximum(avg_preds, 0)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": avg_preds}
    )

    # Ensure submission directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission
