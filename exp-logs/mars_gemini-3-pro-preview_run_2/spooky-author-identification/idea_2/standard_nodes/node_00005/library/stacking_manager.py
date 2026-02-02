import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.model_factory import ModelFactory
from library.utils import calculate_log_loss, seed_everything


class StackingManager:
    """
    Manages the training and inference of the Two-Layer Stacking Ensemble.
    Handles Cross-Validation for OOF generation, Meta-Learner training,
    and final Ensemble prediction.
    """

    def __init__(self):
        """
        Initialize the manager. Sets random seeds for reproducibility.
        """
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED
        seed_everything(self.seed)

    def get_oof_predictions(
        self, X_sparse, X_dense, y, load_cached_data=True, debug=False
    ):
        """
        Generates Out-Of-Fold (OOF) predictions for the base models.
        Uses Stratified K-Fold CV.
        Implements caching to avoid re-computation.

        Args:
            X_sparse: Sparse feature matrix (for LR, MNB).
            X_dense: Dense feature matrix (for XGB).
            y: Target labels.
            load_cached_data (bool): Whether to load from cache.
            debug (bool): Debug mode flag for cache naming.

        Returns:
            dict: Dictionary containing OOF probability arrays for each base model.
        """
        # Determine cache path based on config hash and debug state
        suffix = "_debug" if debug else ""
        cache_filename = f"oof_predictions{suffix}"
        cache_path = Config.get_cache_path(cache_filename) + ".npz"

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading OOF predictions from cache: {cache_path}")
            try:
                with np.load(cache_path) as data:
                    # Convert NpzFile object back to dict
                    return {key: data[key] for key in data.files}
            except Exception as e:
                print(f"Error loading OOF cache: {e}. Recomputing...")

        print("Computing OOF predictions...")

        # Initialize containers
        n_samples = y.shape[0]
        # Infer number of classes from target
        n_classes = len(np.unique(y))

        oof_preds = {
            "lr": np.zeros((n_samples, n_classes)),
            "mnb": np.zeros((n_samples, n_classes)),
            "xgb": np.zeros((n_samples, n_classes)),
        }

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
            print(f"  - Processing Fold {fold + 1}/{self.n_folds}")

            # Prepare Fold Data
            X_sparse_train = X_sparse[train_idx]
            X_sparse_val = X_sparse[val_idx]

            X_dense_train = X_dense[train_idx]
            X_dense_val = X_dense[val_idx]

            y_train = y[train_idx]

            # Instantiate Base Models (Fresh instance per fold)
            models = ModelFactory.get_base_models()

            # 1. Logistic Regression (Sparse)
            models["lr"].fit(X_sparse_train, y_train)
            oof_preds["lr"][val_idx] = models["lr"].predict_proba(X_sparse_val)

            # 2. Multinomial Naive Bayes (Sparse)
            models["mnb"].fit(X_sparse_train, y_train)
            oof_preds["mnb"][val_idx] = models["mnb"].predict_proba(X_sparse_val)

            # 3. XGBoost (Dense)
            # Operates on SVD-reduced dense features
            models["xgb"].fit(X_dense_train, y_train)
            oof_preds["xgb"][val_idx] = models["xgb"].predict_proba(X_dense_val)

        # Save to cache
        print(f"Saving OOF predictions to cache: {cache_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, **oof_preds)

        return oof_preds

    def train_meta_learner(self, oof_preds, y_true):
        """
        Trains the Layer 2 Meta-Learner on the OOF predictions.

        Args:
            oof_preds (dict): Dictionary of OOF probability arrays.
            y_true: True target labels.

        Returns:
            The trained meta-learner model.
        """
        print("Training Meta-Learner...")

        # Construct Meta-Features: Horizontal concatenation of probas
        # Order: LR, MNB, XGB
        meta_features = np.hstack([oof_preds["lr"], oof_preds["mnb"], oof_preds["xgb"]])

        # Get and Train Meta-Learner
        meta_learner = ModelFactory.get_meta_learner()
        meta_learner.fit(meta_features, y_true)

        # Evaluate on Training Data (OOF Score)
        preds = meta_learner.predict_proba(meta_features)
        loss = calculate_log_loss(y_true, preds)
        print(f"Meta-Learner CV Log Loss: {loss}")

        return meta_learner

    def refit_base_models(self, X_sparse, X_dense, y):
        """
        Retrains all base models on the full training dataset for final inference.

        Args:
            X_sparse: Full sparse training features.
            X_dense: Full dense training features.
            y: Full training labels.

        Returns:
            dict: Dictionary of trained base models.
        """
        print("Refitting Base Models on full dataset...")
        models = ModelFactory.get_base_models()

        # 1. Logistic Regression
        models["lr"].fit(X_sparse, y)

        # 2. Multinomial Naive Bayes
        models["mnb"].fit(X_sparse, y)

        # 3. XGBoost
        models["xgb"].fit(X_dense, y)

        return models

    def predict_ensemble(self, base_models, meta_learner, X_sparse_test, X_dense_test):
        """
        Generates final predictions using the trained ensemble.

        Args:
            base_models (dict): Dictionary of trained base models.
            meta_learner: Trained meta-learner.
            X_sparse_test: Sparse test features.
            X_dense_test: Dense test features.

        Returns:
            np.ndarray: Final predicted probabilities.
        """
        print("Generating Ensemble Predictions...")

        # 1. Base Model Predictions
        p_lr = base_models["lr"].predict_proba(X_sparse_test)
        p_mnb = base_models["mnb"].predict_proba(X_sparse_test)
        p_xgb = base_models["xgb"].predict_proba(X_dense_test)

        # 2. Construct Meta-Features (Must match training order)
        meta_features = np.hstack([p_lr, p_mnb, p_xgb])

        # 3. Meta-Learner Prediction
        final_probs = meta_learner.predict_proba(meta_features)

        return final_probs

    def save_submission(self, test_ids, probabilities, class_names):
        """
        Saves the final predictions to the submission file.

        Args:
            test_ids (array-like): IDs for the test samples.
            probabilities (np.ndarray): Predicted probabilities (N_samples, N_classes).
            class_names (array-like): Names of the classes (columns).
        """
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")

        # Create DataFrame
        df_sub = pd.DataFrame(probabilities, columns=class_names)
        df_sub.insert(0, "id", test_ids)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
