import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.dataset_builder import DatasetBuilder
from library.pipeline_factory import PipelineFactory

logger = setup_logger("trainer")


class Trainer:
    """
    Manages the training workflow for the DRSEF strategy.
    Performs Stratified K-Fold CV with nested Grid Search for hyperparameter tuning.
    Generates the final submission file by averaging predictions from all fold models.
    """

    def __init__(self):
        self.config = Config
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def run_cross_validation(self, load_cached_data=True):
        """
        Executes the full cross-validation loop.

        Args:
            load_cached_data (bool): Whether to load pre-processed data from cache.

        Returns:
            list: Validation AUC scores for each fold.
        """
        set_seed(self.config.RANDOM_SEED)

        # 1. Load Data
        logger.info("Loading datasets...")
        builder = DatasetBuilder()
        X_train_split, y_train_split, X_val_split, y_val_split, X_test, test_ids = (
            builder.build_datasets(load_cached_data)
        )

        # 2. Combine Train and Val for Stratified CV
        # We merge them to maximize the data available for the CV folds
        X_full = pd.concat([X_train_split, X_val_split], axis=0).reset_index(drop=True)
        y_full = np.concatenate([y_train_split, y_val_split], axis=0)

        logger.info(f"Combined Training Data Shape: {X_full.shape}")

        # 3. Setup CV
        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED,
        )

        oof_preds = np.zeros(len(y_full))
        fold_scores = []
        trained_models = []

        # Prepare Parameter Grid for GridSearchCV
        # Map Config param names to Pipeline step names
        # Pipeline structure: 'preprocessor' -> 'classifier' (Bagging) -> 'estimator' (LogReg)
        pipeline_grid = {}
        for key, values in self.config.PARAM_GRID.items():
            pipeline_grid[f"classifier__estimator__{key}"] = values

        # 4. CV Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            fold_num = fold + 1
            logger.info(f"Starting Fold {fold_num}/{self.config.N_FOLDS}")

            X_tr, y_tr = X_full.iloc[train_idx], y_full[train_idx]
            X_va, y_va = X_full.iloc[val_idx], y_full[val_idx]

            # Create fresh pipeline
            pipeline = PipelineFactory.create_pipeline({})

            # Grid Search for Hyperparameter Tuning
            # We use a smaller inner CV (3-fold) for efficiency given the bagging overhead
            gs = GridSearchCV(
                estimator=pipeline,
                param_grid=pipeline_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )

            logger.info(f"Running Grid Search on Fold {fold_num}...")
            gs.fit(X_tr, y_tr)

            best_model = gs.best_estimator_
            logger.info(f"Fold {fold_num} Best Params: {gs.best_params_}")

            # Validation
            y_pred_val = best_model.predict_proba(X_va)[:, 1]
            score = roc_auc_score(y_va, y_pred_val)
            logger.info(f"Fold {fold_num} AUC: {score}")  # Full precision

            oof_preds[val_idx] = y_pred_val
            fold_scores.append(score)
            trained_models.append(best_model)

            # Save Model
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")
            joblib.dump(best_model, model_path)
            logger.info(f"Saved model to {model_path}")

        # 5. Overall Evaluation
        overall_auc = roc_auc_score(y_full, oof_preds)
        avg_auc = np.mean(fold_scores)
        logger.info("Cross-Validation Complete.")
        logger.info(f"OOF AUC: {overall_auc}")
        logger.info(f"Average Fold AUC: {avg_auc}")

        # 6. Generate Submission
        self.generate_submission(trained_models, X_test, test_ids)

        return fold_scores

    def generate_submission(self, models, X_test, test_ids):
        """
        Generates predictions for the test set using the ensemble of trained models.
        Saves the result to the submission file.

        Args:
            models (list): List of trained pipeline objects.
            X_test (pd.DataFrame): Test feature matrix.
            test_ids (np.ndarray): Array of request IDs for the test set.
        """
        logger.info("Generating predictions for test set...")

        test_preds = np.zeros(len(X_test))

        for i, model in enumerate(models):
            # Predict probabilities
            preds = model.predict_proba(X_test)[:, 1]
            test_preds += preds

        # Average predictions (Soft Voting)
        test_preds /= len(models)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": test_preds}
        )

        # Save
        submission_path = self.config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
