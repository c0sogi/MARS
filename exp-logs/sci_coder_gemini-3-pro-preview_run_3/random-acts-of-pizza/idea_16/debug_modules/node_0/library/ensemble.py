import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library.config import (
    SEED,
    N_FOLDS,
    RF_LEXICAL_PARAMS,
    RF_BEHAVIORAL_PARAMS,
    XGB_SEMANTIC_PARAMS,
    XGB_EARLY_STOPPING_ROUNDS,
    RF_SEMANTIC_PARAMS,
    LOGREG_ANCHOR_PARAMS,
    META_LEARNER_PARAMS,
)
from library.utils import set_seed, evaluate_auc


class PentViewStackingEnsemble:
    def __init__(self):
        """
        Initializes the Regularized Pent-View Stacking Ensemble.
        Configures the 5 base learners and the meta-learner using global config.
        """
        set_seed(SEED)

        # Level 1 Base Learners
        self.base_models = {
            "lexical_rf": RandomForestClassifier(**RF_LEXICAL_PARAMS),
            "behavioral_rf": RandomForestClassifier(**RF_BEHAVIORAL_PARAMS),
            "semantic_xgb": XGBClassifier(**XGB_SEMANTIC_PARAMS),
            "semantic_rf": RandomForestClassifier(**RF_SEMANTIC_PARAMS),
            "anchor_lr": LogisticRegression(**LOGREG_ANCHOR_PARAMS),
        }

        # Level 2 Meta Learner
        self.meta_learner = LogisticRegression(**META_LEARNER_PARAMS)

        # Track optimal iterations for XGBoost
        self.xgb_best_iterations = []

    def _get_feature_view(self, model_name, X_dict):
        """
        Constructs the specific feature view (topology) for a given model.

        Args:
            model_name (str): Name of the model key.
            X_dict (dict): Dictionary of feature arrays.

        Returns:
            array-like: The concatenated feature matrix (sparse or dense).
        """
        metadata = X_dict["metadata"]

        if model_name == "lexical_rf":
            # Sparse Topology: TF-IDF Text + Dense Metadata
            # Use scipy.sparse.hstack to handle mixed types efficiently
            return sp.hstack([X_dict["lexical"], metadata], format="csr")

        elif model_name == "behavioral_rf":
            # Sparse Topology: TF-IDF Subreddits + Dense Metadata
            return sp.hstack([X_dict["behavioral"], metadata], format="csr")

        elif model_name in ["semantic_xgb", "semantic_rf"]:
            # Dense Topology: SBERT Embeddings + Dense Metadata
            return np.hstack([X_dict["semantic"], metadata])

        elif model_name == "anchor_lr":
            # Linear Topology: Metadata only
            return metadata

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def fit_oof(self, X_dict, y):
        """
        Performs 5-Fold Cross-Validation to:
        1. Generate Out-Of-Fold (OOF) predictions for the Meta-Learner.
        2. Train the Meta-Learner on these OOF predictions.

        Args:
            X_dict (dict): Feature dictionary.
            y (array-like): Target labels.

        Returns:
            np.ndarray: The OOF predictions matrix (N_samples, 5).
        """
        set_seed(SEED)
        n_samples = len(y)
        model_names = list(self.base_models.keys())
        oof_preds = np.zeros((n_samples, len(model_names)))

        # Prepare CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        print(f"Starting Level 1 Training (OOF Generation) with {N_FOLDS} folds...")

        # Reset XGB iteration tracker
        self.xgb_best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
            print(f"  Processing Fold {fold + 1}/{N_FOLDS}...")

            y_train, y_val = y[train_idx], y[val_idx]

            for i, name in enumerate(model_names):
                # Construct view for this fold
                X_view = self._get_feature_view(name, X_dict)
                X_train = X_view[train_idx]
                X_val = X_view[val_idx]

                # Clone/Reset model for this fold
                if name == "lexical_rf":
                    model = RandomForestClassifier(**RF_LEXICAL_PARAMS)
                elif name == "behavioral_rf":
                    model = RandomForestClassifier(**RF_BEHAVIORAL_PARAMS)
                elif name == "semantic_xgb":
                    model = XGBClassifier(**XGB_SEMANTIC_PARAMS)
                elif name == "semantic_rf":
                    model = RandomForestClassifier(**RF_SEMANTIC_PARAMS)
                elif name == "anchor_lr":
                    model = LogisticRegression(**LOGREG_ANCHOR_PARAMS)

                # Train
                if name == "semantic_xgb":
                    # XGBoost with Early Stopping
                    model.fit(
                        X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
                    )
                    # Track best iteration
                    if hasattr(model, "best_iteration"):
                        self.xgb_best_iterations.append(model.best_iteration)
                else:
                    # Standard Sklearn fit
                    model.fit(X_train, y_train)

                # Predict
                # Use predict_proba for probabilities (class 1)
                preds = model.predict_proba(X_val)[:, 1]
                oof_preds[val_idx, i] = preds

        # Evaluate Level 1 Performance
        print("Level 1 OOF AUC Scores:")
        for i, name in enumerate(model_names):
            auc = evaluate_auc(y, oof_preds[:, i])
            print(f"  {name}: {auc}")

        # Train Level 2 Meta-Learner
        print("Training Level 2 Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y)

        # Evaluate Ensemble on OOF
        meta_oof_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        ensemble_auc = evaluate_auc(y, meta_oof_preds)
        print(f"Level 2 Ensemble OOF AUC: {ensemble_auc}")

        return oof_preds

    def fit_final(self, X_dict, y):
        """
        Retrains all Level 1 Base Learners on the full dataset.
        Uses the average optimal trees from CV for XGBoost.

        Args:
            X_dict (dict): Feature dictionary (full train).
            y (array-like): Target labels (full train).
        """
        print("Retraining Level 1 Base Learners on full dataset...")
        set_seed(SEED)

        for name, model in self.base_models.items():
            X_full = self._get_feature_view(name, X_dict)

            if name == "semantic_xgb":
                # Use average best iteration from CV if available
                if self.xgb_best_iterations:
                    avg_trees = int(np.mean(self.xgb_best_iterations))
                    print(
                        f"  Retraining {name} with n_estimators={avg_trees} (derived from CV)..."
                    )
                    # Update params for final fit
                    model.set_params(n_estimators=avg_trees)
                    # No early stopping on full fit (no val set)
                    model.fit(X_full, y, verbose=False)
                else:
                    # Fallback if fit_oof wasn't run
                    print(f"  Retraining {name} with default config...")
                    model.fit(X_full, y, verbose=False)
            else:
                print(f"  Retraining {name}...")
                model.fit(X_full, y)

        print("Final retraining complete.")

    def predict(self, X_dict):
        """
        Generates final predictions using the stacked ensemble.

        Args:
            X_dict (dict): Feature dictionary (test set).

        Returns:
            np.ndarray: Probability of success (class 1).
        """
        n_samples = X_dict["metadata"].shape[0]
        model_names = list(self.base_models.keys())
        level1_preds = np.zeros((n_samples, len(model_names)))

        # Generate Level 1 predictions
        for i, name in enumerate(model_names):
            model = self.base_models[name]
            X_view = self._get_feature_view(name, X_dict)
            level1_preds[:, i] = model.predict_proba(X_view)[:, 1]

        # Generate Level 2 prediction
        final_preds = self.meta_learner.predict_proba(level1_preds)[:, 1]

        return final_preds
