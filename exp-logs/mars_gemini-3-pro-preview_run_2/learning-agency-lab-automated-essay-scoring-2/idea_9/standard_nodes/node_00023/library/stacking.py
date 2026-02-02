import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import get_logger, seed_everything, compute_qwk
from library.features import extract_linguistic_features


class StackingModel:
    """
    Implements the Meta-Learner using LightGBM.
    Aggregates predictions from Semantic, Lexical, and Morphological branches
    along with Structural features.
    """

    def __init__(self):
        self.logger = get_logger("stacking")
        self.models = []
        self.feature_names = []

    def load_upstream_preds(self, prefix, is_test=False):
        """
        Loads prediction files from disk.
        """
        suffix = "test_preds" if is_test else "oof"
        filename = f"{prefix}_{suffix}.npy"
        path = os.path.join(Config.output_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Upstream prediction file not found: {path}")

        return np.load(path)

    def prepare_meta_features(self, df, is_test=False):
        """
        Constructs the feature matrix for the meta-learner.
        """
        # 1. Load Upstream Predictions
        # Semantic (DeBERTa)
        sem_preds = self.load_upstream_preds("semantic", is_test)
        # Lexical (Word N-gram Ridge)
        lex_preds = self.load_upstream_preds("lexical", is_test)
        # Morphological (Char N-gram Ridge)
        mor_preds = self.load_upstream_preds("morphological", is_test)

        # Verify lengths
        if len(df) != len(sem_preds):
            raise ValueError(
                f"Length mismatch: DF ({len(df)}) vs Semantic Preds ({len(sem_preds)})"
            )

        # 2. Load/Compute Structural Features
        split_name = "test" if is_test else "train_val_merged"
        structural_df = extract_linguistic_features(
            df, split=split_name, load_cached_data=True
        )

        # 3. Combine
        # Convert predictions to column vectors
        X = np.column_stack([sem_preds, lex_preds, mor_preds])

        # Add structural features
        # Ensure we only take numeric columns from structural_df
        struct_feats = structural_df.select_dtypes(include=[np.number]).values
        X = np.hstack([X, struct_feats])

        # Save feature names for importance analysis
        self.feature_names = [
            "semantic",
            "lexical",
            "morphological",
        ] + structural_df.select_dtypes(include=[np.number]).columns.tolist()

        return X

    def train(self):
        """
        Trains the LightGBM meta-learner using Stratified K-Fold.
        """
        seed_everything(Config.seed)
        self.logger.info("Starting Stacking Model Training...")

        # 1. Load and Merge Metadata
        # We must reconstruct the exact order used in upstream CV
        if not os.path.exists(Config.train_path) or not os.path.exists(Config.val_path):
            raise FileNotFoundError("Metadata files missing.")

        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

        # Debug subsetting
        if Config.debug:
            self.logger.info(f"Debug mode: Subsetting to {Config.debug_sample_size}")
            df_full = df_full.head(Config.debug_sample_size)

        y = df_full["score"].values

        # 2. Prepare Features
        X = self.prepare_meta_features(df_full, is_test=False)
        self.logger.info(f"Meta-feature matrix shape: {X.shape}")

        # 3. Cross-Validation
        skf = StratifiedKFold(
            n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
        )

        oof_preds = np.zeros(len(df_full))
        self.models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, y.astype(str))):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Initialize LightGBM
            model = lgb.LGBMRegressor(**Config.lgbm_params)

            # Train
            # Use callbacks for logging if needed, but keeping it simple for script
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
            )

            # Predict
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds

            # Store model
            self.models.append(model)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            qwk = compute_qwk(y_val, val_preds)
            self.logger.info(f"Fold {fold} - RMSE: {rmse} - QWK: {qwk}")

        # Overall Metrics
        overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        overall_qwk = compute_qwk(y, oof_preds)
        self.logger.info(f"Overall Stacking RMSE: {overall_rmse}")
        self.logger.info(f"Overall Stacking QWK (Standard Rounding): {overall_qwk}")

        return oof_preds, y

    def predict(self):
        """
        Generates predictions for the test set.
        """
        self.logger.info("Generating Test Predictions...")

        # 1. Load Test Data
        df_test = pd.read_csv(Config.test_path)
        if Config.debug:
            df_test = df_test.head(Config.debug_sample_size)

        # 2. Prepare Features
        X_test = self.prepare_meta_features(df_test, is_test=True)

        # 3. Predict (Average across folds)
        fold_preds = []
        for i, model in enumerate(self.models):
            pred = model.predict(X_test)
            fold_preds.append(pred)

        avg_preds = np.mean(fold_preds, axis=0)
        return avg_preds, df_test["essay_id"].values


def optimize_thresholds(y_true, y_pred_continuous):
    """
    Optimizes decision boundaries using Nelder-Mead to maximize QWK.
    """
    logger = get_logger("threshold_opt")

    def get_score(y_pred, thresholds):
        # Apply thresholds
        # digitize returns indices 0..len(bins).
        # For thresholds [1.5, 2.5...], bins are:
        # <1.5 -> 0 (map to 1)
        # 1.5-2.5 -> 1 (map to 2)
        # ...
        # >5.5 -> 5 (map to 6)
        # So we add 1 to the result of digitize
        return np.digitize(y_pred, thresholds) + 1

    def objective(thresholds):
        # Constraint: Thresholds must be sorted
        if not np.all(np.diff(thresholds) > 0):
            return 100  # Penalty

        # Apply thresholds
        y_pred_int = get_score(y_pred_continuous, thresholds)

        # Calculate QWK
        # We negate it because we want to minimize the objective
        score = compute_qwk(y_true, y_pred_int)
        return -score

    # Initial thresholds (standard rounding boundaries)
    initial_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    logger.info("Optimizing thresholds using Nelder-Mead...")
    result = minimize(
        objective, initial_thresholds, method="Nelder-Mead", options={"maxiter": 500}
    )

    best_thresholds = result.x
    best_score = -result.fun

    logger.info(f"Optimal Thresholds: {best_thresholds}")
    logger.info(f"Optimized QWK: {best_score}")

    return best_thresholds


def run_stacking():
    """
    Main execution function for the stacking module.
    """
    logger = get_logger("main_stacking")

    # 1. Train Stacking Model
    stacker = StackingModel()
    oof_preds, y_true = stacker.train()

    # 2. Optimize Thresholds
    best_thresholds = optimize_thresholds(y_true, oof_preds)

    # 3. Predict on Test Set
    test_preds_continuous, essay_ids = stacker.predict()

    # 4. Apply Optimized Thresholds
    test_preds_int = np.digitize(test_preds_continuous, best_thresholds) + 1

    # Clip just in case
    test_preds_int = np.clip(test_preds_int, 1, 6)

    # 5. Create Submission
    submission_df = pd.DataFrame({"essay_id": essay_ids, "score": test_preds_int})

    # Ensure submission directory exists
    os.makedirs(Config.submission_dir, exist_ok=True)

    # Save
    submission_df.to_csv(Config.submission_path, index=False)
    logger.info(f"Submission saved to {Config.submission_path}")
    logger.info(f"Submission shape: {submission_df.shape}")

    # Print head
    print(submission_df.head())
