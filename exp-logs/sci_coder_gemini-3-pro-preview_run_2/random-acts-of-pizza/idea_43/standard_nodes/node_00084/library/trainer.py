import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.feature_engineering import FeatureEngineer
from library.preprocessing import HAMFPreprocessor
from library.model_factory import ModelFactory


class Trainer:
    """
    Orchestrates the 5-Fold Stratified Cross-Validation training loop,
    hyperparameter tuning, and submission generation for the HAMF-ADBE model.
    """

    def __init__(self):
        self.logger = setup_logger("trainer")
        self.feature_engineer = FeatureEngineer()
        self.model_factory = ModelFactory()

    def run_training(self, load_cached_data: bool = True):
        """
        Executes the full training pipeline:
        1. Loads and merges features.
        2. Runs Stratified K-Fold CV.
        3. Tunes hyperparameters within each fold.
        4. Saves models and preprocessors.
        5. Generates the final submission file.

        Args:
            load_cached_data (bool): Whether to use cached features/embeddings.
        """
        set_seed(Config.SEED)
        self.logger.info("Starting training pipeline...")

        # 1. Load Features
        features = self.feature_engineer.build_feature_set(
            load_cached_data=load_cached_data
        )

        # Merge train and val sets from the feature engineer to perform full CV
        # The feature engineer respects the fixed 80/20 metadata split, but for
        # 5-fold CV we want to use all available labeled data.
        train_feat = features["train"]
        val_feat = features["val"]

        full_data = {}
        # Keys expected: 'y', 'anchor_title', 'anchor_body', 'aux_global', 'aux_hook', 'metadata'
        for key in train_feat.keys():
            if key in val_feat:
                full_data[key] = np.concatenate(
                    [train_feat[key], val_feat[key]], axis=0
                )
            else:
                full_data[key] = train_feat[key]

        y = full_data["y"]
        # Identify feature keys (exclude target)
        feature_keys = [k for k in full_data.keys() if k != "y"]

        # 2. Initialize Cross-Validation
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []

        # 3. CV Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            self.logger.info(f"=== Starting Fold {fold + 1}/{Config.N_FOLDS} ===")

            # Slice data for current fold
            X_train_dict = {k: full_data[k][train_idx] for k in feature_keys}
            y_train_fold = y[train_idx]

            X_val_dict = {k: full_data[k][val_idx] for k in feature_keys}
            y_val_fold = y[val_idx]

            # 4. Preprocessing (Fit on Train, Transform Train & Val)
            # We instantiate a fresh preprocessor to avoid leakage
            preprocessor = HAMFPreprocessor()
            self.logger.info("Fitting preprocessor on training fold data...")
            X_train_processed = preprocessor.fit_transform(X_train_dict)
            X_val_processed = preprocessor.transform(X_val_dict)

            # 5. Model & Hyperparameter Tuning
            self.logger.info("Initializing model and grid search...")
            clf_factory = (
                self.model_factory.create_classifier()
            )  # Returns BaggingClassifier
            param_grid = self.model_factory.get_hyperparameter_grid()

            # Nested CV for hyperparameter tuning (using 3 folds for efficiency)
            grid_search = GridSearchCV(
                clf_factory,
                param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )

            self.logger.info("Tuning hyperparameters...")
            grid_search.fit(X_train_processed, y_train_fold)

            best_model = grid_search.best_estimator_
            self.logger.info(f"Best params for fold {fold}: {grid_search.best_params_}")

            # 6. Evaluation
            y_pred_val = best_model.predict_proba(X_val_processed)[:, 1]
            score = roc_auc_score(y_val_fold, y_pred_val)
            fold_scores.append(score)
            self.logger.info(f"Fold {fold} ROC AUC: {score}")

            # 7. Save Artifacts
            model_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold}.joblib"
            )
            proc_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"processor_fold_{fold}.joblib"
            )

            joblib.dump(best_model, model_path)
            joblib.dump(preprocessor, proc_path)
            self.logger.info(f"Saved model and preprocessor for fold {fold}.")

        avg_auc = np.mean(fold_scores)
        self.logger.info(f"Training Complete. Average CV ROC AUC: {avg_auc}")

        # 8. Generate Submission
        self._generate_submission(features["test"])

    def _generate_submission(self, test_features):
        """
        Generates predictions for the test set using the ensemble of fold models.
        """
        self.logger.info("Generating submission for test set...")
        request_ids = test_features["request_id"]

        # Prepare X_test dict
        # Ensure we only pass the keys expected by the preprocessor
        valid_keys = [
            "anchor_title",
            "anchor_body",
            "aux_global",
            "aux_hook",
            "metadata",
        ]
        X_test_dict = {k: test_features[k] for k in valid_keys if k in test_features}

        fold_preds = []

        # Iterate over all saved fold models
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold}.joblib"
            )
            proc_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"processor_fold_{fold}.joblib"
            )

            if not os.path.exists(model_path) or not os.path.exists(proc_path):
                self.logger.warning(f"Artifacts for fold {fold} not found. Skipping.")
                continue

            # Load model and preprocessor
            model = joblib.load(model_path)
            preprocessor = joblib.load(proc_path)

            # Transform test data using the fold-specific preprocessor
            X_test_processed = preprocessor.transform(X_test_dict)

            # Predict probabilities
            preds = model.predict_proba(X_test_processed)[:, 1]
            fold_preds.append(preds)

        if not fold_preds:
            raise RuntimeError("No predictions generated. Check model training.")

        # Average predictions (CV-Bagging)
        avg_preds = np.mean(fold_preds, axis=0)

        # Create submission DataFrame
        df_sub = pd.DataFrame(
            {
                "request_id": request_ids,
                "requester_received_pizza": avg_preds,
            }
        )

        # Save to disk
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
