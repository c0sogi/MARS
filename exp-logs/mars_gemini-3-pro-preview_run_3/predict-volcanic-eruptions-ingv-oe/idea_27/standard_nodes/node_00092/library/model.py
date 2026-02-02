import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

from library import config, utils, features


def train_lgbm_fold(X_train, y_train, X_val, y_val, params):
    """
    Trains a single LightGBM regressor for a specific fold.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.array): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (np.array): Validation targets.
        params (dict): LightGBM hyperparameters.

    Returns:
        model: Trained LightGBM model.
        val_preds (np.array): Predictions on the validation set.
        mae (float): Mean Absolute Error on the validation set.
    """
    # Create LightGBM Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Configure callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 100)),
        lgb.log_evaluation(
            period=0
        ),  # Suppress verbose logging, print final result manually
    ]

    # Train the model
    model = lgb.train(params, train_data, valid_sets=[val_data], callbacks=callbacks)

    # Generate predictions on validation set
    # num_iteration=model.best_iteration is handled automatically by lgb.train object usually,
    # but explicit usage ensures we use the best round.
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate metric
    mae = utils.calculate_mae(y_val, val_preds)

    return model, val_preds, mae


class EnsembleManager:
    def __init__(self):
        self.n_folds = config.N_FOLDS
        self.models = []
        # Copy params to avoid modifying global config
        self.params = config.LGBM_PARAMS.copy()

        # Requirement: Objective = L2, Optimization = Early Stopping based on MAE
        self.params["objective"] = "regression"  # L2 Loss
        self.params["metric"] = "mae"  # Monitor MAE for Early Stopping

    def train_loop(self):
        """
        Executes the Stratified K-Fold Cross-Validation training loop.
        """
        print("Loading metadata...")
        # Load both train and val splits and combine them for full cross-validation
        train_meta = utils.load_metadata("train")
        val_meta = utils.load_metadata("val")
        full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

        print(f"Combined training set size: {len(full_meta)}")

        # Generate Features (Cached)
        # We use a distinct split_name to cache this combined dataset
        print("Generating/Loading features...")
        X_y = features.create_feature_matrix(full_meta, split_name="combined_train")

        # Separate features and target
        target_col = "time_to_eruption"
        drop_cols = ["segment_id", "time_to_eruption"]

        # Ensure target exists
        if target_col not in X_y.columns:
            raise ValueError(
                "Target column 'time_to_eruption' missing from feature matrix."
            )

        y = X_y[target_col].values
        X = X_y.drop(columns=drop_cols)

        # Stratified K-Fold Setup
        # We need to bin the continuous target to perform stratification
        num_bins = 10
        # Adjust bins if dataset is smaller than expected (e.g. debug mode)
        if len(y) < num_bins:
            num_bins = max(2, len(y) // 2)

        target_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=config.SEED
        )

        oof_preds = np.zeros(len(y))

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_bins)):
            print(f"\n--- Fold {fold + 1} / {self.n_folds} ---")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model for this fold
            model, val_preds, val_mae = train_lgbm_fold(
                X_train, y_train, X_val, y_val, self.params
            )

            self.models.append(model)
            oof_preds[val_idx] = val_preds

            print(f"Fold {fold + 1} Best Iteration: {model.best_iteration}")
            print(f"Fold {fold + 1} MAE: {val_mae}")

        # Calculate Overall CV Score
        overall_mae = utils.calculate_mae(y, oof_preds)
        print(f"\nOverall CV MAE: {overall_mae}")

    def predict_average(self):
        """
        Generates predictions for the test set by averaging outputs from all ensemble models.
        """
        if not self.models:
            raise RuntimeError("No models trained. Run train_loop() first.")

        print("\n--- Inference on Test Set ---")

        # Load Test Metadata
        test_meta = utils.load_metadata("test")

        # Generate Test Features
        test_df = features.create_feature_matrix(test_meta, split_name="test")

        # Prepare Features (Drop non-feature columns)
        # Note: time_to_eruption is not in test set
        X_test = test_df.drop(columns=["segment_id"], errors="ignore")

        # Initialize predictions
        avg_preds = np.zeros(len(X_test))

        print(f"Aggregating predictions from {len(self.models)} models...")

        for i, model in enumerate(self.models):
            preds = model.predict(X_test, num_iteration=model.best_iteration)
            avg_preds += preds

        # Average
        avg_preds /= len(self.models)

        # Save Submission
        utils.save_submission(avg_preds, test_meta)
