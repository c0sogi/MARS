import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import Timer, set_seed
from library.model_definitions import ModelZoo


class TrainingEngine:
    """
    Manages the training lifecycle: Cross-Validation, Meta-Learning,
    Final Retraining, and Submission Generation.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.model_zoo = ModelZoo()
        # Define the active Level 1 models based on the architecture
        self.models_to_train = [
            "unified_rf",
            "lexical_rf",
            "community_rf",
            "semantic_xgb",
            "semantic_rf",
            "metadata_lr",
        ]
        os.makedirs(self.working_dir, exist_ok=True)
        set_seed(Config.RANDOM_STATE)

    def _get_oof_cache_path(self):
        return os.path.join(self.working_dir, "level1_oof_predictions.npz")

    def run_cv_and_generate_oof(self, feature_dict, load_cached_data=True):
        """
        Performs 5-fold Stratified CV on the Training set to generate OOF predictions.
        Implements caching to avoid re-running CV.
        """
        cache_path = self._get_oof_cache_path()

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading OOF predictions from cache: {cache_path}")
            try:
                loaded = np.load(cache_path)
                return {k: loaded[k] for k in loaded.files}
            except Exception as e:
                print(f"Failed to load OOF cache: {e}. Recomputing...")

        # 2. Setup Cross-Validation
        y_train = feature_dict["y_train"]
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
        )

        # Initialize containers for OOF predictions
        oof_preds = {m: np.zeros(len(y_train)) for m in self.models_to_train}

        print(f"Starting Level 1 Cross-Validation ({Config.N_FOLDS} folds)...")

        # 3. Iterate Folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(y_train)), y_train)
        ):
            print(f"  Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            for model_name in self.models_to_train:
                # Get full training data for this model
                X_full, _ = self.model_zoo.format_data(
                    model_name, feature_dict, split="train"
                )

                # Slice data for this fold
                if sparse.issparse(X_full):
                    X_fold_train = X_full[train_idx]
                    X_fold_val = X_full[val_idx]
                else:
                    X_fold_train = X_full[train_idx]
                    X_fold_val = X_full[val_idx]

                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]

                # Instantiate and Train
                model = self.model_zoo.get_model(model_name)

                # For XGBoost, we use the fold validation set for Early Stopping
                # For others, we just train
                if "xgb" in model_name:
                    self.model_zoo.train_model(
                        model,
                        X_fold_train,
                        y_fold_train,
                        X_val=X_fold_val,
                        y_val=y_fold_val,
                        verbose=False,
                    )
                else:
                    self.model_zoo.train_model(
                        model, X_fold_train, y_fold_train, verbose=False
                    )

                # Predict OOF
                preds = self.model_zoo.predict_proba(model, X_fold_val)
                oof_preds[model_name][val_idx] = preds

        # 4. Report Metrics
        print("\nLevel 1 CV Scores (AUC):")
        for model_name in self.models_to_train:
            auc = roc_auc_score(y_train, oof_preds[model_name])
            print(f"  {model_name}: {auc}")

        # 5. Save to Cache
        print(f"Saving OOF predictions to {cache_path}...")
        np.savez(cache_path, **oof_preds)

        return oof_preds

    def train_meta_learner(self, oof_preds, y_train):
        """
        Trains the Level 2 Logistic Regression on the OOF predictions.
        """
        print("\nTraining Level 2 Meta-Learner...")

        # Stack OOF predictions to create meta-features
        # Shape: (n_samples, n_models)
        X_meta = np.column_stack([oof_preds[m] for m in self.models_to_train])

        meta_model = self.model_zoo.get_model("meta_lr")
        meta_model.fit(X_meta, y_train)

        # Sanity check (In-sample AUC)
        preds = meta_model.predict_proba(X_meta)[:, 1]
        auc = roc_auc_score(y_train, preds)
        print(f"Meta-Learner In-Sample AUC: {auc}")

        return meta_model

    def retrain_base_models(self, feature_dict):
        """
        Retrains base models for final inference.
        - XGBoost: Train on Train, use Validation for Early Stopping.
        - Others: Train on Train + Validation combined.
        """
        print("\nRetraining Base Models for Final Inference...")
        final_models = {}

        y_train = feature_dict["y_train"]
        y_val = feature_dict["y_val"]

        for model_name in self.models_to_train:
            print(f"  Retraining {model_name}...")
            model = self.model_zoo.get_model(model_name)

            # Get Data
            X_train, _ = self.model_zoo.format_data(
                model_name, feature_dict, split="train"
            )
            X_val, _ = self.model_zoo.format_data(model_name, feature_dict, split="val")

            if "xgb" in model_name:
                # XGBoost: Explicit Validation Set for Early Stopping
                self.model_zoo.train_model(
                    model, X_train, y_train, X_val=X_val, y_val=y_val, verbose=True
                )
            else:
                # RF / Linear: Concatenate Train + Val
                if sparse.issparse(X_train):
                    X_combined = sparse.vstack([X_train, X_val], format="csr")
                else:
                    X_combined = np.vstack([X_train, X_val])

                y_combined = np.concatenate([y_train, y_val])

                self.model_zoo.train_model(model, X_combined, y_combined, verbose=False)

            final_models[model_name] = model

        return final_models

    def generate_submission(self, final_models, meta_model, feature_dict):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("\nGenerating Final Predictions on Test Set...")

        # 1. Generate Base Model Predictions
        test_preds_base = []
        for model_name in self.models_to_train:
            model = final_models[model_name]
            X_test, _ = self.model_zoo.format_data(
                model_name, feature_dict, split="test"
            )
            preds = self.model_zoo.predict_proba(model, X_test)
            test_preds_base.append(preds)

        # 2. Stack Predictions
        X_meta_test = np.column_stack(test_preds_base)

        # 3. Meta Prediction
        final_probs = meta_model.predict_proba(X_meta_test)[:, 1]

        # 4. Save Submission
        test_ids = feature_dict["test_ids"]
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_probs}
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(f"Submission Shape: {submission_df.shape}")
        print(f"Head:\n{submission_df.head()}")

    def run(self, feature_dict, load_cached_data=True):
        """
        Orchestrates the full training pipeline.
        """
        with Timer("Full Training Pipeline"):
            # 1. Level 1 CV & OOF Generation
            oof_preds = self.run_cv_and_generate_oof(feature_dict, load_cached_data)

            # 2. Level 2 Meta-Learner Training
            meta_model = self.train_meta_learner(oof_preds, feature_dict["y_train"])

            # 3. Final Retraining of Base Models
            final_base_models = self.retrain_base_models(feature_dict)

            # 4. Submission Generation
            self.generate_submission(final_base_models, meta_model, feature_dict)
