import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.utils import get_logger, compute_qwk

logger = get_logger("Stacking")


class StackingModel:
    """
    Level 2 Stacking Ensemble Model.
    Trains a Gradient Boosting Regressor (LightGBM) on the outputs of the Level 1 model
    combined with explicit meta-features.
    """

    def __init__(self):
        self.features = ["pred_score"] + Config.meta_features
        self.target = "score"
        self.models = []
        # Copy params to avoid modifying global config
        self.params = Config.lgbm_params.copy()

    def train(self, oof_df, train_meta_df):
        """
        Trains the stacking ensemble using OOF predictions from Level 1 and meta-features.

        Args:
            oof_df (pd.DataFrame): DataFrame containing 'essay_id', 'pred_score' (Level 1 OOF), and 'fold'.
            train_meta_df (pd.DataFrame): DataFrame containing 'essay_id', 'score', 'fold', and meta-features.

        Returns:
            float: Average QWK score across folds.
        """
        logger.info("Preparing data for Stacking Level 2...")

        # Select relevant columns from train_meta_df
        # We need 'fold' to ensure we respect the same validation splits
        meta_cols = ["essay_id", "score", "fold"] + Config.meta_features
        train_data = train_meta_df[meta_cols].copy()

        # Merge with OOF predictions on essay_id
        # This adds the 'pred_score' column to our feature set
        train_data = train_data.merge(
            oof_df[["essay_id", "pred_score"]], on="essay_id", how="left"
        )

        # Handle potential missing values in OOF (should not happen in a complete run)
        if train_data["pred_score"].isnull().any():
            missing_count = train_data["pred_score"].isnull().sum()
            logger.warning(
                f"Found {missing_count} missing OOF predictions. Filling with mean."
            )
            train_data["pred_score"] = train_data["pred_score"].fillna(
                train_data["pred_score"].mean()
            )

        scores = []
        self.models = []  # Reset models list

        logger.info(f"Starting Stacking Training with {Config.num_folds} folds...")

        for fold in range(Config.num_folds):
            # Create Train/Val split based on the fold column
            train_idx = train_data["fold"] != fold
            val_idx = train_data["fold"] == fold

            X_train = train_data.loc[train_idx, self.features]
            y_train = train_data.loc[train_idx, self.target]

            X_val = train_data.loc[val_idx, self.features]
            y_val = train_data.loc[val_idx, self.target]

            # Initialize LightGBM Regressor
            model = lgb.LGBMRegressor(**self.params)

            # Define callbacks for Early Stopping and Logging
            # stopping_rounds=50: Stop if validation metric doesn't improve for 50 rounds
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),  # Suppress internal logging
            ]

            # Train the model
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=callbacks,
            )

            # Generate validation predictions
            val_preds = model.predict(X_val)

            # Calculate Metrics
            rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            qwk = compute_qwk(y_val, val_preds)

            logger.info(
                f"Fold {fold} - Stacking RMSE: {rmse:.5f} - Stacking QWK: {qwk:.5f}"
            )

            scores.append(qwk)
            self.models.append(model)

        mean_qwk = np.mean(scores)
        logger.info(f"Average Stacking QWK: {mean_qwk:.5f}")
        return mean_qwk

    def predict(self, test_meta_df, l1_test_preds):
        """
        Generates predictions for the test set by averaging outputs from all fold models.

        Args:
            test_meta_df (pd.DataFrame): Test DataFrame with meta-features.
            l1_test_preds (np.array or pd.Series): Raw float predictions from Level 1 model.

        Returns:
            np.array: Final stacked predictions (floats).
        """
        logger.info("Generating Stacking Predictions...")

        X_test = test_meta_df.copy()

        # Add the Level 1 prediction as a feature
        # We assume l1_test_preds is aligned with test_meta_df
        X_test["pred_score"] = l1_test_preds

        # Select the exact features used during training
        X_test = X_test[self.features]

        # Average predictions from all fold models
        final_preds = np.zeros(len(X_test))
        for model in self.models:
            final_preds += model.predict(X_test)

        final_preds /= len(self.models)

        return final_preds


def run_stacking(oof_df, l1_test_preds, load_cached_data=True):
    """
    Main execution function for the stacking module.

    Args:
        oof_df (pd.DataFrame): OOF predictions from Stage 1.
        l1_test_preds (np.array): Test predictions from Stage 1 (floats).
        load_cached_data (bool): Whether to load processed meta-features from cache.

    Returns:
        pd.DataFrame: The final submission DataFrame.
    """
    # Define paths for cached data
    train_cache_path = os.path.join(Config.cache_dir, "train_processed.parquet")
    test_cache_path = os.path.join(Config.cache_dir, "test_processed.parquet")

    # Load Meta-Features from Cache
    # We rely on data.py having run previously to generate these files
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        logger.info(f"Loading cached meta-features from {Config.cache_dir}")
        train_df = pd.read_parquet(train_cache_path)
        test_df = pd.read_parquet(test_cache_path)
    else:
        logger.error("Cache not found. Stacking requires processed data.")
        return None

    # Initialize and Train Stacker
    stacker = StackingModel()
    stacker.train(oof_df, train_df)

    # Predict on Test Set
    final_preds = stacker.predict(test_df, l1_test_preds)

    # Create Submission DataFrame
    submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": final_preds})

    # Post-processing: Round to nearest integer and clip to [1, 6]
    submission["score"] = np.round(submission["score"]).clip(1, 6).astype(int)

    # Save Submission
    submission.to_csv(Config.submission_path, index=False)
    logger.info(f"Final Stacking Submission saved to {Config.submission_path}")

    return submission
