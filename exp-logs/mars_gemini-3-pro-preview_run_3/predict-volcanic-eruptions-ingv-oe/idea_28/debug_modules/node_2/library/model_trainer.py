import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from library.config import Config


class ModelTrainer:
    """
    Manages the training of the High-Resolution Hybrid-Transform Bagging Ensemble.
    Implements Stratified K-Fold CV with LightGBM and generates final submissions.
    """

    def __init__(self):
        self.models = []
        self.feature_cols = []
        self.n_folds = Config.N_FOLDS
        self.working_dir = Config.WORKING_DIR
        self.submission_path = Config.SUBMISSION_PATH

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata.
        """
        exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
        return [c for c in df.columns if c not in exclude_cols]

    def train_fold_model(self, X_train, y_train, X_val, y_val, fold_id):
        """
        Trains a single LightGBM model for a specific fold.
        """
        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Prepare callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.TRAIN_PARAMS["early_stopping_rounds"]
            ),
            lgb.log_evaluation(period=Config.TRAIN_PARAMS["verbose_eval"]),
        ]

        # Extract n_estimators to use as num_boost_round
        params = Config.MODEL_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 10000)

        print(f"\n--- Training Fold {fold_id + 1} ---")

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model
        model_path = os.path.join(self.working_dir, f"lgbm_model_fold_{fold_id}.txt")
        model.save_model(model_path)
        print(f"Model for fold {fold_id + 1} saved to {model_path}")

        return model

    def run_stratified_cv(self, train_df):
        """
        Executes Stratified K-Fold Cross-Validation.
        """
        self.feature_cols = self._get_feature_columns(train_df)
        X = train_df[self.feature_cols]
        y = train_df["time_to_eruption"]

        # Binning for stratification
        num_bins = 10
        y_binned = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(train_df))
        fold_scores = []

        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            model = self.train_fold_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_id
            )
            self.models.append(model)

            # Predict on validation fold
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_pred

            # Calculate score
            score = mean_absolute_error(y_val_fold, val_pred)
            fold_scores.append(score)
            print(f"Fold {fold_id + 1} MAE: {score}")

        overall_mae = mean_absolute_error(y, oof_preds)
        print(f"\nOverall CV MAE: {overall_mae}")
        print(f"Average Fold MAE: {np.mean(fold_scores)}")

        return overall_mae

    def generate_submission(self, test_df):
        """
        Generates predictions for the test set using the trained ensemble.
        """
        print("\nGenerating submission...")

        if not self.models:
            # Try to load models if not in memory
            print("Loading models from disk...")
            for fold_id in range(self.n_folds):
                model_path = os.path.join(
                    self.working_dir, f"lgbm_model_fold_{fold_id}.txt"
                )
                if os.path.exists(model_path):
                    model = lgb.Booster(model_file=model_path)
                    self.models.append(model)
                else:
                    raise FileNotFoundError(
                        f"Model file {model_path} not found. Train models first."
                    )

        # Ensure feature columns match
        if not self.feature_cols:
            self.feature_cols = self._get_feature_columns(test_df)

        X_test = test_df[self.feature_cols]

        # Bagging: Average predictions from all folds
        fold_preds = []
        for i, model in enumerate(self.models):
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            fold_preds.append(pred)

        avg_preds = np.mean(fold_preds, axis=0)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": test_df["segment_id"], "time_to_eruption": avg_preds}
        )

        # Save
        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
        print("Sample head:")
        print(submission_df.head())


def train_and_predict(train_df, val_df, test_df):
    """
    Main orchestration function.
    """
    trainer = ModelTrainer()

    # Run CV on the training set
    # Note: We rely on the internal CV of train_df for model creation.
    # val_df can be used as an additional external check if desired,
    # but for this implementation, we focus on the 5-fold CV strategy defined in the Idea.
    trainer.run_stratified_cv(train_df)

    # Generate submission
    trainer.generate_submission(test_df)
