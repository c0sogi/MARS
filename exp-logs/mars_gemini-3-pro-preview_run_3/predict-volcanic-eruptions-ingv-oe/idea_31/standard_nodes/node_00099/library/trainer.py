import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import library.config as config
import library.utils as utils


def train_fold_model(X_train, y_train, X_val, y_val, fold_id):
    """
    Trains a single LightGBM model for a specific fold.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        fold_id (int): Identifier for the current fold.

    Returns:
        model: The trained LightGBM Booster.
        float: The Mean Absolute Error on the validation set.
    """
    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Setup callbacks for early stopping and logging
    # Using log_evaluation(0) to suppress verbose output as requested
    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    # Train the model
    model = lgb.train(
        params=config.LGBM_PARAMS,
        train_set=train_set,
        valid_sets=[train_set, val_set],
        callbacks=callbacks,
    )

    # Generate validation predictions using the best iteration
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate metric
    mae = utils.calculate_mae(y_val.values, val_preds)

    # Save model artifact
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    model_path = os.path.join(config.WORKING_DIR, f"lgbm_model_fold_{fold_id}.txt")
    model.save_model(model_path)

    return model, mae


def run_cross_validation(train_df, num_folds=config.NUM_FOLDS, seed=config.SEED):
    """
    Orchestrates Stratified K-Fold Cross-Validation.

    Args:
        train_df (pd.DataFrame): DataFrame containing features and target.
        num_folds (int): Number of folds for CV.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of trained LightGBM models.
    """
    # Ensure reproducibility
    utils.seed_everything(seed)

    # Separate features and target
    X = train_df.drop(columns=["segment_id", "time_to_eruption"])
    y = train_df["time_to_eruption"]

    # Create bins for stratified splitting on continuous target
    # 10 bins is a standard heuristic for regression stratification
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    models = []
    scores = []

    print(f"Starting Stratified K-Fold CV with {num_folds} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model, mae = train_fold_model(X_train, y_train, X_val, y_val, fold)

        models.append(model)
        scores.append(mae)

        print(f"Fold {fold + 1} MAE: {mae}")

    avg_mae = np.mean(scores)
    print(f"Average CV MAE: {avg_mae}")

    return models


def generate_submission(models, test_df):
    """
    Generates predictions for the test set using the ensemble of models.
    Saves the result to submission.csv.

    Args:
        models (list): List of trained LightGBM models.
        test_df (pd.DataFrame): Test features DataFrame.
    """
    # Prepare test data
    X_test = test_df.drop(columns=["segment_id"])
    segment_ids = test_df["segment_id"]

    # Initialize prediction array
    final_preds = np.zeros(len(X_test))

    # Aggregate predictions from all models (Bagging)
    for model in models:
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        final_preds += preds

    # Average the predictions
    final_preds /= len(models)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"segment_id": segment_ids.astype(int), "time_to_eruption": final_preds}
    )

    # Save to file
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
