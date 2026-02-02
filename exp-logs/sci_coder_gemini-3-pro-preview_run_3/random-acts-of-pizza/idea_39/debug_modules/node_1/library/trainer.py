import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import timer, print_metrics
from library.model_registry import get_base_models, get_meta_model


class StackingTrainer:
    """
    Manages the training, cross-validation, and retraining of the Hex-View Stacking Ensemble.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_models = get_base_models()
        self.meta_model = get_meta_model()
        self.n_folds = 5
        self.random_state = Config.RANDOM_SEED

    def _prepare_input(self, X_dict: dict, model_name: str, idx=None):
        """
        Prepares the specific feature combination for a given model.

        Args:
            X_dict: Dictionary of feature views.
            model_name: Name of the model to prepare data for.
            idx: Optional indices to slice the data (for CV splits).

        Returns:
            Combined feature matrix (sparse or dense).
        """

        # Helper to slice if indices are provided
        def slice_data(data, indices):
            if indices is None:
                return data
            return data[indices]

        # Extract views
        view_lexical = slice_data(X_dict["view_lexical"], idx)
        view_behavioral = slice_data(X_dict["view_behavioral"], idx)
        view_semantic = slice_data(X_dict["view_semantic"], idx)
        view_interaction = slice_data(X_dict["view_interaction"], idx)
        view_meta = slice_data(X_dict["view_meta"], idx)

        # 1. Lexical Bagger: Sparse Text + Dense Meta
        if model_name == "lexical_bagger":
            return sp.hstack([view_lexical, view_meta], format="csr")

        # 2. Community Bagger: Sparse History + Dense Meta
        elif model_name == "community_bagger":
            return sp.hstack([view_behavioral, view_meta], format="csr")

        # 3. Semantic Booster: Dense Embeddings + Dense Meta
        elif model_name == "semantic_booster":
            return np.hstack([view_semantic, view_meta])

        # 4. Semantic Bagger: Dense Embeddings + Dense Meta
        elif model_name == "semantic_bagger":
            return np.hstack([view_semantic, view_meta])

        # 5. Interaction Booster: SVD_Text + SVD_History + Meta (Already combined)
        elif model_name == "interaction_booster":
            return view_interaction

        # 6. Metadata Anchor: Meta only
        elif model_name == "metadata_anchor":
            return view_meta

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def generate_oof(self, X_dict: dict, y: np.ndarray):
        """
        Performs 5-Fold Stratified CV to generate Out-Of-Fold predictions.

        Args:
            X_dict: Dictionary of training feature views.
            y: Target array.

        Returns:
            oof_preds: DataFrame of OOF predictions (N_samples x N_models).
        """
        self.logger.info(
            f"Starting {self.n_folds}-Fold Stratified CV for OOF generation..."
        )

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        model_names = list(self.base_models.keys())
        n_samples = len(y)
        oof_matrix = np.zeros((n_samples, len(model_names)))

        # Store fold scores for reporting
        fold_scores = {name: [] for name in model_names}

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
            self.logger.info(f"Processing Fold {fold + 1}/{self.n_folds}")

            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            for i, name in enumerate(model_names):
                model = clone(self.base_models[name])

                # Prepare data for this model and fold
                X_train_fold = self._prepare_input(X_dict, name, train_idx)
                X_val_fold = self._prepare_input(X_dict, name, val_idx)

                # Train
                # Check if model supports early stopping (XGBoost)
                if "booster" in name:
                    # XGBoost requires eval_set for early stopping
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False,
                    )
                else:
                    # Standard Sklearn fit
                    model.fit(X_train_fold, y_train_fold)

                # Predict
                probs = model.predict_proba(X_val_fold)[:, 1]
                oof_matrix[val_idx, i] = probs

                # Score
                score = roc_auc_score(y_val_fold, probs)
                fold_scores[name].append(score)

        # Report average scores
        self.logger.info("OOF CV Results (AUC):")
        mean_scores = {}
        for name, scores in fold_scores.items():
            mean_auc = np.mean(scores)
            mean_scores[name] = mean_auc
            print_metrics({f"{name}_mean_auc": mean_auc})

        oof_df = pd.DataFrame(oof_matrix, columns=model_names)
        return oof_df

    def train_meta(self, oof_df: pd.DataFrame, y: np.ndarray):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        self.logger.info("Training Level 2 Meta-Learner...")
        with timer("Meta-Learner Training"):
            self.meta_model.fit(oof_df, y)

            # Check coefficients to see contribution of each base model
            coefs = self.meta_model.coef_[0]
            coef_dict = dict(zip(oof_df.columns, coefs))
            self.logger.info(f"Meta-Learner Coefficients: {coef_dict}")

    def retrain_final_models(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Retrains base models using the Validation-Guided Retraining Protocol.

        RF/Linear: Train on Train + Val.
        XGBoost: Train on Train, use Val for Early Stopping.
        """
        self.logger.info("Retraining Level 1 Base Learners for Final Submission...")

        self.final_models = {}

        for name, base_model in self.base_models.items():
            self.logger.info(f"Retraining {name}...")
            model = clone(base_model)

            # Prepare Data
            X_train_feat = self._prepare_input(X_train_dict, name)
            X_val_feat = self._prepare_input(X_val_dict, name)

            if "booster" in name:
                # XGBoost: Train on Train, Early Stopping on Val
                # This prevents "blind overfitting"
                model.fit(
                    X_train_feat, y_train, eval_set=[(X_val_feat, y_val)], verbose=False
                )
            else:
                # RF / Linear: Train on Combined (Train + Val)
                # Maximize data usage
                if sp.issparse(X_train_feat):
                    X_combined = sp.vstack([X_train_feat, X_val_feat], format="csr")
                else:
                    X_combined = np.vstack([X_train_feat, X_val_feat])

                y_combined = np.concatenate([y_train, y_val])

                model.fit(X_combined, y_combined)

            self.final_models[name] = model

    def predict(self, X_test_dict: dict) -> np.ndarray:
        """
        Generates final predictions for the test set.

        1. Generate Level 1 predictions using retrained base models.
        2. Feed into Level 2 Meta-Learner.
        """
        self.logger.info("Generating Final Predictions...")

        model_names = list(self.base_models.keys())
        n_samples = X_test_dict["view_meta"].shape[0]
        l1_preds = np.zeros((n_samples, len(model_names)))

        # Level 1 Predictions
        for i, name in enumerate(model_names):
            model = self.final_models[name]
            X_test_feat = self._prepare_input(X_test_dict, name)
            l1_preds[:, i] = model.predict_proba(X_test_feat)[:, 1]

        l1_df = pd.DataFrame(l1_preds, columns=model_names)

        # Level 2 Prediction
        final_probs = self.meta_model.predict_proba(l1_df)[:, 1]

        return final_probs
