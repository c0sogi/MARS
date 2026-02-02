import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import print_header, timer, ensure_dir
from library.model_zoo import get_hept_view_models, get_meta_learner


class HybridEnsembleEngine:
    """
    Implements the Robust Hybrid Inference Protocol for the Hept-View Stacking Ensemble.
    - Volatile Learners (XGB/LGBM): Trained with Early Stopping per fold. Inference via CV-Bagging (Avg of 5 models).
    - Stable Learners (RF/LR): Trained per fold for OOF. Retrained fully on Union Data for Inference.
    """

    def __init__(self, X_train_dict, y_train, X_test_dict, output_dir=None):
        """
        Args:
            X_train_dict: Dict with keys 'lexical', 'behavioral', 'semantic', 'metadata' containing training features.
            y_train: Training targets.
            X_test_dict: Dict with keys 'lexical', 'behavioral', 'semantic', 'metadata' containing test features.
            output_dir: Directory to save models and predictions. Defaults to Config.WORKING_DIR.
        """
        self.X_train_dict = X_train_dict
        self.y_train = (
            y_train.reset_index(drop=True)
            if isinstance(y_train, pd.Series)
            else pd.Series(y_train)
        )
        self.X_test_dict = X_test_dict

        self.output_dir = output_dir if output_dir else Config.WORKING_DIR
        self.models_dir = os.path.join(self.output_dir, "models")
        ensure_dir(self.models_dir)

        # Define Model Groups
        self.volatile_models = [
            "semantic_booster",
            "semantic_gradient",
            "temporal_booster",
        ]
        self.stable_models = [
            "lexical_bagger",
            "community_bagger",
            "semantic_bagger",
            "metadata_anchor",
        ]

        # Early Stopping Config Mapping
        self.early_stopping_config = {
            "semantic_booster": Config.SEMANTIC_BOOSTER_PARAMS.get(
                "early_stopping_rounds", 100
            ),
            "semantic_gradient": Config.SEMANTIC_GRADIENT_PARAMS.get(
                "early_stopping_rounds", 100
            ),
            "temporal_booster": Config.TEMPORAL_BOOSTER_PARAMS.get(
                "early_stopping_rounds", 100
            ),
        }

    def _get_model_features(self, model_name, X_dict):
        """
        Constructs the specific feature set for a given model branch.
        """
        meta = X_dict["metadata"]

        if "lexical" in model_name:
            # Sparse Lexical + Metadata
            return np.hstack([X_dict["lexical"], meta])
        elif "community" in model_name:
            # Sparse Behavioral + Metadata
            return np.hstack([X_dict["behavioral"], meta])
        elif "semantic" in model_name:
            # Dense Semantic + Metadata
            return np.hstack([X_dict["semantic"], meta])
        elif "metadata" in model_name or "temporal" in model_name:
            # Metadata Only
            return meta
        else:
            raise ValueError(f"Unknown model branch: {model_name}")

    def _train_volatile_fold(
        self, model_name, model, X_train, y_train, X_val, y_val, fold_idx
    ):
        """
        Trains a volatile model with Early Stopping and saves it.
        """
        es_rounds = self.early_stopping_config.get(model_name)

        # Fit with early stopping
        # Note: XGBoost and LGBM (sklearn API) handle eval_set differently but compatible enough here
        # We explicitly pass eval_set and early_stopping_rounds to fit()

        if "booster" in model_name or "gradient" in model_name:
            # LGBM and XGBoost
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                # early_stopping_rounds is passed to fit for sklearn API wrappers where it was stripped from init
                # For XGBoost >= 1.6 it might be in init, but passing to fit is generally safe or required for LGBM
                callbacks=None,  # LGBM uses callbacks or params, sklearn wrapper usually takes early_stopping_rounds param in fit
            )
            # For some versions of libraries, early_stopping_rounds is a fit param
            # If the model instance already has it (XGB), it's fine. If not (LGBM wrapper), we might need to pass it.
            # However, standard sklearn API for these often accepts it in fit kwargs.
            # To be robust given the provided model_zoo stripped it for LGBM:
            try:
                # Attempt passing via kwargs
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    early_stopping_rounds=es_rounds,
                )
            except TypeError:
                # Fallback if double fit or param issue (unlikely with recent versions)
                # If init had it, we just fit
                model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc"
                )
        else:
            model.fit(X_train, y_train)

        # Save Fold Model
        save_path = os.path.join(
            self.models_dir, f"{model_name}_fold_{fold_idx}.joblib"
        )
        joblib.dump(model, save_path)

        return model

    def _train_stable_fold(self, model_name, model, X_train, y_train):
        """
        Trains a stable model (no early stopping).
        """
        model.fit(X_train, y_train)
        return model

    def train_cv_and_predict(self):
        """
        Executes the training pipeline:
        1. Level 1 CV (OOF Generation + Volatile Model Saving)
        2. Level 2 Training
        3. Stable Model Full Retraining
        4. Test Prediction & Submission
        """
        print_header("Starting Hybrid Ensemble Training")

        # Initialize OOF DataFrame
        models_map = get_hept_view_models()
        kf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = pd.DataFrame(index=self.y_train.index, columns=models_map.keys())
        oof_preds[:] = np.nan

        # 1. Level 1 Cross-Validation
        print("\n--- Level 1: Cross-Validation & OOF Generation ---")

        for model_name, base_model in models_map.items():
            print(f"\nTraining {model_name}...")

            # Prepare Features
            X_full = self._get_model_features(model_name, self.X_train_dict)
            y_full = self.y_train.values

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
                X_tr, X_val = X_full[train_idx], X_full[val_idx]
                y_tr, y_val = y_full[train_idx], y_full[val_idx]

                model = clone(base_model)

                if model_name in self.volatile_models:
                    # Volatile: Train with ES, Save Model
                    trained_model = self._train_volatile_fold(
                        model_name, model, X_tr, y_tr, X_val, y_val, fold
                    )
                else:
                    # Stable: Train standard
                    trained_model = self._train_stable_fold(
                        model_name, model, X_tr, y_tr
                    )

                # Predict OOF
                if hasattr(trained_model, "predict_proba"):
                    preds = trained_model.predict_proba(X_val)[:, 1]
                else:
                    preds = trained_model.predict(X_val)

                oof_preds.loc[val_idx, model_name] = preds

                # Score
                score = roc_auc_score(y_val, preds)
                fold_scores.append(score)
                # print(f"  Fold {fold} AUC: {score}") # Optional verbosity

            avg_score = np.mean(fold_scores)
            print(f"{model_name} Average CV AUC: {avg_score}")

        # Save OOF predictions
        oof_path = os.path.join(self.output_dir, "oof_predictions.csv")
        oof_preds.to_csv(oof_path)

        # 2. Level 2 Meta-Learner Training
        print("\n--- Level 2: Meta-Learner Training ---")
        meta_learner = get_meta_learner()
        meta_learner.fit(oof_preds, self.y_train)

        meta_score = roc_auc_score(
            self.y_train, meta_learner.predict_proba(oof_preds)[:, 1]
        )
        print(f"Meta-Learner CV AUC (on OOF): {meta_score}")

        joblib.dump(meta_learner, os.path.join(self.models_dir, "meta_learner.joblib"))

        # 3. Retrain Stable Models on Full Data
        print("\n--- Retraining Stable Models on Full Union Dataset ---")
        for model_name in self.stable_models:
            print(f"Retraining {model_name}...")
            base_model = models_map[model_name]
            X_full = self._get_model_features(model_name, self.X_train_dict)

            model = clone(base_model)
            model.fit(X_full, self.y_train)

            joblib.dump(model, os.path.join(self.models_dir, f"{model_name}.joblib"))

        # 4. Generate Test Predictions
        self._predict_test_and_submit(models_map.keys())

    def _predict_test_and_submit(self, model_names):
        """
        Generates predictions for the test set using the Hybrid Protocol.
        """
        print_header("Generating Test Predictions")

        test_level1 = pd.DataFrame(
            index=range(len(next(iter(self.X_test_dict.values())))), columns=model_names
        )

        for model_name in model_names:
            X_test = self._get_model_features(model_name, self.X_test_dict)

            if model_name in self.volatile_models:
                # CV-Bagging: Average predictions from 5 saved fold models
                fold_preds = []
                for fold in range(Config.N_FOLDS):
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    model = joblib.load(model_path)
                    fold_preds.append(model.predict_proba(X_test)[:, 1])

                avg_preds = np.mean(fold_preds, axis=0)
                test_level1[model_name] = avg_preds
                print(f"{model_name}: Inferred via CV-Bagging (5 models)")

            else:
                # Stable: Use single fully retrained model
                model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
                model = joblib.load(model_path)
                preds = model.predict_proba(X_test)[:, 1]
                test_level1[model_name] = preds
                print(f"{model_name}: Inferred via Single Full Model")

        # Meta-Learner Prediction
        meta_learner = joblib.load(os.path.join(self.models_dir, "meta_learner.joblib"))
        final_preds = meta_learner.predict_proba(test_level1)[:, 1]

        # Save Submission
        self._save_submission(final_preds)

    def _save_submission(self, predictions):
        """
        Formats and saves the submission file.
        """
        print("\nSaving submission file...")

        # Load sample submission to get IDs
        # We assume the test set order matches the sample submission as per standard competition format
        # However, to be safe, we load the test metadata to get IDs
        test_meta_path = Config.TEST_PATH
        if os.path.exists(test_meta_path):
            test_df = pd.read_parquet(test_meta_path)
            request_ids = test_df[Config.ID_COL].values
        else:
            # Fallback to sample submission if metadata missing (unlikely)
            sample_sub_path = os.path.join(Config.INPUT_DIR, "sampleSubmission.csv")
            sample_df = pd.read_csv(sample_sub_path)
            request_ids = sample_df[Config.ID_COL].values

        submission = pd.DataFrame(
            {Config.ID_COL: request_ids, Config.TARGET_COL: predictions}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")
