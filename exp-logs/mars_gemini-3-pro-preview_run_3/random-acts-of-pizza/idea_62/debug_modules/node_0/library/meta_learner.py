import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import setup_logger, set_seed
from library.model_factory import ModelFactory

logger = setup_logger("meta_learner")


class Level2MetaLearner:
    """
    Manages the Level-2 Stacking Ensemble logic for the Conservative Granular Hept-View architecture.
    Handles:
    1. Training the Meta-Learner (Logistic Regression) on OOF predictions.
    2. Generating final predictions for the test set by aggregating Level-1 models (Hybrid Inference).
    """

    def __init__(self):
        self.model_dir = Config.MODEL_DIR
        self.submission_path = Config.SUBMISSION_PATH
        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_STATE

        # Define the branch configuration for inference.
        # This must match the topology defined in TrainingPipeline.
        # 'stable': Uses a single model retrained on the full Union Dataset.
        # 'volatile': Uses CV-Bagging (average of N fold models).
        self.branch_config = {
            "lexical_bagger": "stable",
            "community_bagger": "stable",
            "semantic_booster": "volatile",
            "semantic_gradient": "volatile",
            "semantic_bagger": "stable",
            "metadata_anchor": "stable",
            "temporal_booster": "volatile",
        }

    def _prepare_features(self, features_dict, branch_name, split="test"):
        """
        Reconstructs the specific feature set for a given branch by concatenating
        the primary view with the augmented global metadata.

        Args:
            features_dict (dict): Dictionary containing feature matrices.
            branch_name (str): Name of the branch.
            split (str): 'train' or 'test'.

        Returns:
            scipy.sparse.csr_matrix or np.ndarray: The combined feature matrix.
        """
        prefix = "train" if split == "train" else "test"
        X_meta = features_dict[f"{prefix}_meta"]

        if branch_name == "lexical_bagger":
            # Sparse Lexical + Dense Meta
            X_lex = features_dict[f"{prefix}_lexical"]
            return sp.hstack([X_lex, X_meta], format="csr")

        elif branch_name == "community_bagger":
            # Sparse Community + Dense Meta
            X_comm = features_dict[f"{prefix}_community"]
            return sp.hstack([X_comm, X_meta], format="csr")

        elif branch_name in [
            "semantic_booster",
            "semantic_gradient",
            "semantic_bagger",
        ]:
            # Dense Semantic + Dense Meta
            X_sem = features_dict[f"{prefix}_semantic"]
            return np.hstack([X_sem, X_meta])

        elif branch_name in ["metadata_anchor", "temporal_booster"]:
            # Pure Meta
            return X_meta

        else:
            raise ValueError(f"Unknown branch: {branch_name}")

    def train(self, oof_df):
        """
        Trains the Level 2 Meta Learner on the OOF predictions from Level 1.

        Args:
            oof_df (pd.DataFrame): DataFrame containing OOF probabilities for each branch and the target.
        """
        logger.info(f"\n{'='*20} Level 2: Meta Learner Training {'='*20}")

        # Ensure consistent column ordering based on config
        feature_cols = list(self.branch_config.keys())

        # Verify all required columns are present
        missing_cols = [c for c in feature_cols if c not in oof_df.columns]
        if missing_cols:
            raise ValueError(f"OOF DataFrame is missing columns: {missing_cols}")

        X = oof_df[feature_cols].values
        y = oof_df["target"].values

        # Initialize Meta Learner
        meta_learner = ModelFactory.get_meta_learner()

        # Cross-Validation for Performance Estimation
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        fold_scores = []

        logger.info("Starting Cross-Validation for Meta-Learner...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = clone(meta_learner)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]

            score = roc_auc_score(y_val, probs)
            fold_scores.append(score)
            logger.info(f"  Meta-Learner Fold {fold} AUC: {score:.16f}")

        avg_score = np.mean(fold_scores)
        logger.info(f"Meta-Learner CV Average AUC: {avg_score:.16f}")

        # Final Training on Full OOF Data
        logger.info("Retraining Meta-Learner on full OOF dataset...")
        meta_learner.fit(X, y)

        # Save Model
        save_path = os.path.join(self.model_dir, "meta_learner.joblib")
        joblib.dump(meta_learner, save_path)
        logger.info(f"Saved Meta-Learner to {save_path}")

        return meta_learner

    def generate_submission(self, features_dict, test_ids):
        """
        Generates the final submission file by running inference on the test set.
        Implements Consistent Hybrid Inference:
        - Loads single models for 'stable' branches.
        - Loads and averages fold models for 'volatile' branches.
        - Stacks predictions using the Meta-Learner.

        Args:
            features_dict (dict): Dictionary containing test feature matrices.
            test_ids (array-like): List/Array of request IDs for the test set.
        """
        logger.info(f"\n{'='*20} Generating Submission {'='*20}")

        level1_preds = pd.DataFrame()

        # 1. Level 1 Inference
        for name, mode in self.branch_config.items():
            logger.info(f"Running inference for branch: {name} (Mode: {mode})")

            # Reconstruct test features for this branch
            X_test = self._prepare_features(features_dict, name, split="test")

            if mode == "stable":
                # Load single full-data model
                model_path = os.path.join(self.model_dir, f"{name}.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Stable model not found at {model_path}")

                model = joblib.load(model_path)
                preds = model.predict_proba(X_test)[:, 1]

            elif mode == "volatile":
                # CV-Bagging: Average predictions from all fold models
                fold_preds = []
                for fold in range(self.n_folds):
                    model_path = os.path.join(
                        self.model_dir, f"{name}_fold_{fold}.joblib"
                    )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Fold model not found at {model_path}")

                    model = joblib.load(model_path)
                    p = model.predict_proba(X_test)[:, 1]
                    fold_preds.append(p)

                # Average probabilities
                preds = np.mean(fold_preds, axis=0)

            else:
                raise ValueError(f"Unknown inference mode: {mode}")

            level1_preds[name] = preds

        # 2. Level 2 Inference
        logger.info("Running inference for Level 2 Meta-Learner...")
        meta_model_path = os.path.join(self.model_dir, "meta_learner.joblib")
        if not os.path.exists(meta_model_path):
            raise FileNotFoundError(
                f"Meta-Learner model not found at {meta_model_path}"
            )

        meta_learner = joblib.load(meta_model_path)

        # Ensure column order matches training
        X_meta_input = level1_preds[list(self.branch_config.keys())].values

        final_probs = meta_learner.predict_proba(X_meta_input)[:, 1]

        # 3. Create and Save Submission
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_probs}
        )

        submission.to_csv(self.submission_path, index=False)
        logger.info(f"Submission saved to {self.submission_path}")
        logger.info(f"Submission shape: {submission.shape}")

        return submission
