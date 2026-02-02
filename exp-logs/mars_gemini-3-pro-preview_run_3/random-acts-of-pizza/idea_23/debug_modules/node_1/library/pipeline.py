import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import SEED, N_FOLDS, SUBMISSION_PATH, ID_COL, TARGET_COL
from library.utils import set_seed, print_metrics
from library.data_processing import DataProcessor
from library.models import ModelRegistry


class StackingTrainer:
    def __init__(self):
        set_seed(SEED)
        self.processor = DataProcessor()
        self.base_models = ModelRegistry.create_base_models()
        self.meta_learner = ModelRegistry.get_meta_learner()
        self.data = None

        # Mapping model names to feature keys in the data dictionary
        # These keys correspond to the output keys from DataProcessor.process_data
        self.model_feature_map = {
            "lexical_bagger": "lexical",
            "community_bagger": "behavioral",
            "semantic_booster": "semantic",
            "semantic_bagger": "semantic",
            "manifold_neighbor": "manifold",
            "metadata_anchor": "contextual",
        }

    def load_data(self, load_cached_data=True):
        """
        Loads and processes data using the DataProcessor.
        """
        self.data = self.processor.process_data(load_cached_data=load_cached_data)

    def _get_features(self, split, model_name):
        """
        Helper to extract specific features for a model and split.
        split: 'train', 'val', 'test'
        model_name: key in self.base_models
        """
        feature_type = self.model_feature_map[model_name]
        key = f"X_{split}_{feature_type}"
        return self.data[key]

    def run_cv(self):
        """
        Performs 5-Fold Stratified CV to generate OOF predictions and train the Meta-Learner.
        """
        if self.data is None:
            self.load_data()

        print(f"Starting {N_FOLDS}-Fold Cross-Validation...")

        # 1. Prepare Development Data (Train + Val combined)
        y_train = self.data["y_train"]
        y_val = self.data["y_val"]
        y_dev = np.concatenate([y_train, y_val])

        # Prepare Features for all views (concatenated)
        X_dev_dict = {}
        for model_name in self.base_models.keys():
            X_t = self._get_features("train", model_name)
            X_v = self._get_features("val", model_name)
            X_dev_dict[model_name] = np.vstack([X_t, X_v])

        # 2. Initialize OOF Matrix
        model_names = list(self.base_models.keys())
        oof_preds = np.zeros((len(y_dev), len(model_names)))

        # 3. Cross-Validation Loop
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        model_scores = {name: [] for name in model_names}

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(y_dev)), y_dev)
        ):
            y_fold_train, y_fold_val = y_dev[train_idx], y_dev[val_idx]

            for i, name in enumerate(model_names):
                # Instantiate a fresh model for this fold
                model = ModelRegistry.create_base_models()[name]

                X_view = X_dev_dict[name]
                X_fold_train, X_fold_val = X_view[train_idx], X_view[val_idx]

                # Train
                # Note: For CV OOF generation, we fit normally.
                # Strict early stopping inside CV would require a nested split which reduces data further.
                model.fit(X_fold_train, y_fold_train)

                # Predict
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_fold_val)[:, 1]
                else:
                    # Fallback for models that might not support predict_proba
                    p = model.predict(X_fold_val)

                oof_preds[val_idx, i] = p

                # Score
                auc = roc_auc_score(y_fold_val, p)
                model_scores[name].append(auc)

        # 4. Report Level 1 Metrics
        print("\n--- Level 1 Base Learner CV Performance (AUC) ---")
        for name in model_names:
            mean_auc = np.mean(model_scores[name])
            std_auc = np.std(model_scores[name])
            print(f"{name}: {mean_auc:.8f} (+/- {std_auc:.4f})")

        # 5. Train Meta-Learner
        print("\nTraining Level 2 Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y_dev)

        # Evaluate Meta-Learner (OOF Score)
        meta_pred_oof = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(y_dev, meta_pred_oof)
        print_metrics({"Meta-Learner OOF AUC": meta_auc})

    def retrain_final(self):
        """
        Retrains base models on the full dataset.
        Implements the Validation-Guided Retraining Protocol:
        - XGBoost: Trained on Train split, Early Stopping on Val split.
        - Others: Trained on concatenated Train + Val splits.
        """
        print("\nRetraining Base Models for Submission...")

        if self.data is None:
            self.load_data()

        y_train = self.data["y_train"]
        y_val = self.data["y_val"]
        y_full = np.concatenate([y_train, y_val])

        for name, model in self.base_models.items():
            X_t = self._get_features("train", name)
            X_v = self._get_features("val", name)

            if name == "semantic_booster":
                # Validation-Guided Retraining for Boosting
                # Prevents blind overfitting by using the explicit validation set for stopping
                model.fit(
                    X_t,
                    y_train,
                    eval_set=[(X_v, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                # Full Data Retraining for Non-Boosting
                X_full = np.vstack([X_t, X_v])
                model.fit(X_full, y_full)

    def generate_submission(self):
        """
        Generates predictions for the test set using the retrained ensemble
        and saves the result to the submission file.
        """
        print("\nGenerating Submission...")

        if self.data is None:
            self.load_data()

        test_ids = self.data["test_ids"]
        n_samples = len(test_ids)
        model_names = list(self.base_models.keys())

        # 1. Level 1 Predictions
        L1_test_preds = np.zeros((n_samples, len(model_names)))

        for i, name in enumerate(model_names):
            model = self.base_models[name]
            X_test = self._get_features("test", name)

            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_test)[:, 1]
            else:
                p = model.predict(X_test)

            L1_test_preds[:, i] = p

        # 2. Level 2 Prediction
        final_probs = self.meta_learner.predict_proba(L1_test_preds)[:, 1]

        # 3. Save
        submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_probs})

        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Head:\n{submission_df.head()}")
