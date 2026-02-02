import os
import time
import numpy as np
import pandas as pd
import scipy.sparse
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import save_artifact, load_artifact, set_seed
import library.model_definitions as model_factory


class EnsembleTrainer:
    """
    Manages the training lifecycle of the Granular Hept-View Stacking Ensemble.
    Implements the Robust Hybrid Inference Protocol:
    1. 5-Fold CV for Volatile Learners (XGB/LGBM) with Early Stopping.
    2. Full Retraining for Stable Learners (RF/Linear).
    3. Meta-Learner training on OOF predictions.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing processed feature matrices and target.
                              Keys: 'y_train', 'X_meta_train', 'X_lex_train', etc.
        """
        self.data = data_dict
        self.y_train = data_dict["y_train"]
        self.n_samples = len(self.y_train)
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Define the 7 base learners and their types
        # Type: 'volatile' (CV-Bagging) or 'stable' (Full-Retrain)
        # Branch: Helper to identify which feature set to use
        self.learner_config = {
            "lexical_bagger": {
                "type": "stable",
                "branch": "lexical",
                "factory": model_factory.get_lexical_bagger,
            },
            "community_bagger": {
                "type": "stable",
                "branch": "behavioral",
                "factory": model_factory.get_community_bagger,
            },
            "semantic_booster": {
                "type": "volatile",
                "branch": "semantic",
                "factory": model_factory.get_semantic_booster,
            },
            "semantic_gradient": {
                "type": "volatile",
                "branch": "semantic",
                "factory": model_factory.get_semantic_gradient,
            },
            "semantic_bagger": {
                "type": "stable",
                "branch": "semantic",
                "factory": model_factory.get_semantic_bagger,
            },
            "metadata_anchor": {
                "type": "stable",
                "branch": "metadata",
                "factory": model_factory.get_metadata_anchor,
            },
            "temporal_booster": {
                "type": "volatile",
                "branch": "metadata",
                "factory": model_factory.get_temporal_booster,
            },
        }

    def _get_features(self, branch, indices=None, is_test=False):
        """
        Constructs the feature matrix for a specific branch.
        Concatenates the specific view with the Global Metadata.
        """
        suffix = "test" if is_test else "train"

        # Get Global Metadata
        X_meta = self.data[f"X_meta_{suffix}"]
        if indices is not None:
            X_meta = X_meta[indices]

        # Get Specific View
        if branch == "metadata":
            return X_meta

        view_key = ""
        if branch == "lexical":
            view_key = f"X_lex_{suffix}"
        elif branch == "behavioral":
            view_key = f"X_beh_{suffix}"
        elif branch == "semantic":
            view_key = f"X_sem_{suffix}"

        X_view = self.data[view_key]
        if indices is not None:
            X_view = X_view[indices]

        # Concatenate (Stacking Sparse + Dense if necessary)
        if scipy.sparse.issparse(X_view):
            return scipy.sparse.hstack([X_view, X_meta], format="csr")
        else:
            return np.hstack([X_view, X_meta])

    def train_ensemble(self):
        """
        Executes the full training pipeline.
        """
        print(f"Starting Ensemble Training on {self.n_samples} samples...")
        set_seed(Config.RANDOM_SEED)

        # 1. Initialize OOF Matrix
        oof_preds = pd.DataFrame(
            index=range(self.n_samples), columns=self.learner_config.keys()
        )
        oof_preds[:] = 0.0

        # 2. Cross-Validation Loop
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(self.n_samples), self.y_train)
        ):
            print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]

            for name, config in self.learner_config.items():
                print(f"Training {name}...", end=" ")
                start_time = time.time()

                # Prepare Data
                X_tr = self._get_features(config["branch"], train_idx)
                X_val = self._get_features(config["branch"], val_idx)

                # Instantiate
                model = config["factory"]()

                # Fit
                if config["type"] == "volatile":
                    # Early Stopping for Volatile Learners
                    # Note: We assume the factory returns an sklearn-compatible wrapper (XGBClassifier/LGBMClassifier)
                    # that accepts eval_set in fit().
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    # Standard Fit for Stable Learners
                    model.fit(X_tr, y_tr)

                # Predict (OOF)
                val_probs = model.predict_proba(X_val)[:, 1]
                oof_preds.loc[val_idx, name] = val_probs

                # Evaluate
                fold_auc = roc_auc_score(y_val, val_probs)
                print(f"AUC: {fold_auc:.16f} ({time.time() - start_time:.2f}s)")

                # Save Fold Model (Required for Volatile Hybrid Inference, and compliance for Stable)
                self._save_model(model, f"{name}_fold_{fold}")

        # 3. Retrain Stable Learners on Full Data
        print("\n--- Retraining Stable Learners on Full Union Dataset ---")
        for name, config in self.learner_config.items():
            if config["type"] == "stable":
                print(f"Retraining {name} (Full)...", end=" ")
                X_full = self._get_features(config["branch"], is_test=False)
                model = config["factory"]()
                model.fit(X_full, self.y_train)
                self._save_model(model, f"{name}_full")
                print("Done.")

        # 4. Train Meta-Learner
        print("\n--- Training Level 2 Meta-Learner ---")
        X_level2 = oof_preds.values
        meta_learner = model_factory.get_meta_learner()
        meta_learner.fit(X_level2, self.y_train)

        meta_auc = roc_auc_score(
            self.y_train, meta_learner.predict_proba(X_level2)[:, 1]
        )
        print(f"Meta-Learner OOF AUC: {meta_auc:.16f}")

        self._save_model(meta_learner, "meta_learner")

        # Save OOF predictions for analysis
        oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
        oof_preds["target"] = self.y_train
        oof_preds.to_csv(oof_path, index=False)
        print(f"OOF predictions saved to {oof_path}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the Hybrid Inference Protocol.
        """
        print("\n--- Generating Submission ---")

        # 1. Generate Level 1 Predictions
        level1_test_preds = pd.DataFrame(
            index=range(self.data["X_meta_test"].shape[0]),
            columns=self.learner_config.keys(),
        )

        for name, config in self.learner_config.items():
            print(f"Predicting {name}...", end=" ")
            X_test = self._get_features(config["branch"], is_test=True)

            if config["type"] == "volatile":
                # Hybrid Inference: Average of 5 fold models
                fold_preds = []
                for fold in range(Config.N_FOLDS):
                    model = self._load_model(f"{name}_fold_{fold}")
                    fold_preds.append(model.predict_proba(X_test)[:, 1])
                level1_test_preds[name] = np.mean(fold_preds, axis=0)
                print(f"Averaged {Config.N_FOLDS} folds.")

            else:  # stable
                # Hybrid Inference: Use single full model
                model = self._load_model(f"{name}_full")
                level1_test_preds[name] = model.predict_proba(X_test)[:, 1]
                print("Used full model.")

        # 2. Generate Level 2 Predictions
        print("Predicting with Meta-Learner...")
        meta_learner = self._load_model("meta_learner")
        final_probs = meta_learner.predict_proba(level1_test_preds.values)[:, 1]

        # 3. Create Submission File
        # Load test IDs from original metadata to ensure alignment
        test_meta_df = load_artifact(Config.TEST_PATH)
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_meta_df[Config.ID_COL], Config.TARGET_COL: final_probs}
        )

        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Head:\n{submission_df.head()}")

    def _save_model(self, model, name):
        path = os.path.join(self.models_dir, f"{name}.joblib")
        joblib.dump(model, path)

    def _load_model(self, name):
        path = os.path.join(self.models_dir, f"{name}.joblib")
        return joblib.load(path)
