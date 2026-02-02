import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library import config, data_manager


class ModelTrainer:
    """
    Manages the training and inference of the LightGBM model using Stratified K-Fold CV.
    """

    def __init__(self):
        self.models = []
        self.feature_cols = []
        self.best_score = float("inf")

    def train_cross_validation(self, debug=False):
        """
        Performs Stratified K-Fold Cross-Validation.
        Loads train and val datasets, combines them, and trains the model.

        Args:
            debug (bool): If True, uses a smaller subset of data for debugging.
        """
        print(f"Loading datasets for training (Debug={debug})...")

        # Load training and validation features using the data manager (handles caching)
        train_df = data_manager.generate_feature_matrix("train", debug=debug)
        val_df = data_manager.generate_feature_matrix("val", debug=debug)

        # Combine datasets to maximize data for Cross-Validation
        full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        # Identify feature columns (exclude metadata and target)
        exclude_cols = ["segment_id", "time_to_eruption"]
        self.feature_cols = [c for c in full_df.columns if c not in exclude_cols]

        print(
            f"Training on {len(full_df)} samples with {len(self.feature_cols)} features."
        )

        X = full_df[self.feature_cols]
        y = full_df["time_to_eruption"]

        # Prepare Stratification
        # Bin the continuous target to allow StratifiedKFold
        num_bins = 15
        # Safety check for very small debug datasets
        if len(y) < num_bins:
            num_bins = max(2, len(y) // 5)

        # Create bins based on quantiles
        stratify_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        oof_preds = np.zeros(len(full_df))
        fold_scores = []

        # Prepare LightGBM parameters
        params = config.LGBM_PARAMS.copy()
        # Extract n_estimators to use as num_boost_round in lgb.train
        n_estimators = params.pop("n_estimators", 6000)

        print(f"Starting {config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_bins)):
            print(f"\n--- Fold {fold + 1} / {config.N_FOLDS} ---")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Create LightGBM Datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            # Define Callbacks
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(period=config.VERBOSE_EVAL),
            ]

            # Train Model
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            self.models.append(model)

            # Predict on validation fold
            # best_iteration is automatically used if early_stopping was active
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

            # Calculate and Print Metric
            mae = np.mean(np.abs(y_val - val_preds))
            fold_scores.append(mae)
            print(f"Fold {fold + 1} MAE: {mae}")

        # Overall Metric
        total_mae = np.mean(np.abs(y - oof_preds))
        self.best_score = total_mae
        print(f"\nCross-Validation Completed.")
        print(f"Overall OOF MAE: {total_mae}")

    def predict(self, debug=False):
        """
        Generates predictions for the test set using the trained models.
        Saves the submission file to the configured submission directory.

        Args:
            debug (bool): If True, uses a smaller subset of data for debugging.
        """
        print(f"\nGenerating predictions for Test Set (Debug={debug})...")

        if not self.models:
            raise RuntimeError("No models found. Run train_cross_validation first.")

        # Load test features
        test_df = data_manager.generate_feature_matrix("test", debug=debug)

        # Ensure feature columns match training data
        missing_cols = set(self.feature_cols) - set(test_df.columns)
        if missing_cols:
            print(f"Warning: Missing columns in test set: {missing_cols}")
            for c in missing_cols:
                test_df[c] = 0

        X_test = test_df[self.feature_cols]
        segment_ids = test_df["segment_id"]

        # Ensemble Predictions (Average across folds)
        avg_preds = np.zeros(len(X_test))

        for i, model in enumerate(self.models):
            preds = model.predict(X_test, num_iteration=model.best_iteration)
            avg_preds += preds

        avg_preds /= len(self.models)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": segment_ids, "time_to_eruption": avg_preds}
        )

        # Save Submission
        save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
