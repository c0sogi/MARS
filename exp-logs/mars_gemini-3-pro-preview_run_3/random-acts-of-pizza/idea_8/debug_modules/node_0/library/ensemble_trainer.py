import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library import config, model_definitions, utils


class TriViewStackingEnsemble:
    """
    Implements a Tri-View Stacking Ensemble architecture.

    Level 1: Three Random Forest classifiers trained on distinct views:
             1. Lexical (Text TF-IDF + Meta)
             2. Semantic (SBERT Embeddings + Meta)
             3. Behavioral (Subreddit History TF-IDF + Meta)

    Level 2: A Logistic Regression meta-learner trained on the probabilities
             output by the Level 1 models.
    """

    def __init__(self):
        # Initialize Level 1 Base Learners
        self.l1_models = {
            "lexical": model_definitions.get_level1_model(),
            "semantic": model_definitions.get_level1_model(),
            "behavioral": model_definitions.get_level1_model(),
        }

        # Initialize Level 2 Meta Learner
        self.meta_model = model_definitions.get_meta_model()

        # Placeholders for fitted state
        self.is_fitted = False

    def _prepare_view_data(self, features_dict, view_name):
        """
        Concatenates the specific view features with the shared meta features.
        Handles the combination of sparse and dense matrices.

        Args:
            features_dict (dict): Dictionary containing 'lexical', 'semantic',
                                  'behavioral', and 'meta' feature arrays.
            view_name (str): The key for the specific view ('lexical', 'semantic', 'behavioral').

        Returns:
            scipy.sparse.csr_matrix: The concatenated feature matrix.
        """
        view_features = features_dict[view_name]
        meta_features = features_dict["meta"]

        # Ensure meta features are 2D
        if meta_features.ndim == 1:
            meta_features = meta_features.reshape(-1, 1)

        # Use sparse hstack to combine. If view_features is dense (like SBERT),
        # it will be converted to sparse, which is acceptable for Random Forest.
        combined_X = sp.hstack([view_features, meta_features], format="csr")
        return combined_X

    def fit(self, features_dict, y):
        """
        Performs 5-Fold Cross-Validation to generate OOF predictions, trains the
        meta-learner, and then retrains all base learners on the full dataset.

        Args:
            features_dict (dict): Dictionary of feature arrays/matrices.
            y (array-like): Target labels.
        """
        utils.set_seed(config.SEED)

        n_samples = len(y)
        n_folds = config.N_FOLDS

        # Initialize OOF prediction matrix: (n_samples, 3 views)
        oof_preds = np.zeros((n_samples, 3))
        view_indices = {"lexical": 0, "semantic": 1, "behavioral": 2}

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

        print(f"[Ensemble] Starting {n_folds}-Fold Cross-Validation Stacking...")

        # Iterate over folds
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y)
        ):
            print(f"  - Processing Fold {fold_idx + 1}/{n_folds}")

            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Train each view's model
            for view_name, model in self.l1_models.items():
                # Prepare data for this view
                X_full_view = self._prepare_view_data(features_dict, view_name)
                X_train_fold = X_full_view[train_idx]
                X_val_fold = X_full_view[val_idx]

                # Clone model parameters (create a fresh instance for the fold) is not strictly
                # necessary if we just call fit, but good practice. Here we just call fit
                # on a fresh clone or the same object. Since we need to retrain on full data
                # later, we can use temporary models for CV.
                fold_model = model_definitions.get_level1_model()

                # Fit
                fold_model.fit(X_train_fold, y_train_fold)

                # Predict (Probability of class 1)
                val_probs = fold_model.predict_proba(X_val_fold)[:, 1]

                # Store OOF
                col_idx = view_indices[view_name]
                oof_preds[val_idx, col_idx] = val_probs

        # Calculate and print OOF scores for each view
        print("\n[Ensemble] Level 1 OOF Performance:")
        for view_name, col_idx in view_indices.items():
            auc = roc_auc_score(y, oof_preds[:, col_idx])
            print(f"  - {view_name.capitalize()} View AUC: {auc}")

        # Train Level 2 Meta Learner on OOF predictions
        print("\n[Ensemble] Training Level 2 Meta-Learner on OOF predictions...")
        self.meta_model.fit(oof_preds, y)

        # Calculate Stacking Score
        stacking_probs = self.meta_model.predict_proba(oof_preds)[:, 1]
        stacking_auc = roc_auc_score(y, stacking_probs)
        print(f"  - Stacking Ensemble OOF AUC: {stacking_auc}")

        # Retrain Level 1 Models on Full Data
        print("\n[Ensemble] Retraining Level 1 Base Learners on Full Dataset...")
        for view_name, model in self.l1_models.items():
            print(f"  - Retraining {view_name} model...")
            X_full = self._prepare_view_data(features_dict, view_name)
            model.fit(X_full, y)

        self.is_fitted = True
        print("[Ensemble] Training Complete.")

    def predict(self, features_dict):
        """
        Generates predictions for the test set.

        Args:
            features_dict (dict): Dictionary of feature arrays/matrices for the test set.

        Returns:
            np.ndarray: Probability of success (class 1) for each sample.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        n_samples = features_dict["meta"].shape[0]
        l1_preds = np.zeros((n_samples, 3))
        view_indices = {"lexical": 0, "semantic": 1, "behavioral": 2}

        # Get predictions from each base learner
        for view_name, model in self.l1_models.items():
            X_test = self._prepare_view_data(features_dict, view_name)
            probs = model.predict_proba(X_test)[:, 1]
            l1_preds[:, view_indices[view_name]] = probs

        # Get final prediction from meta learner
        final_probs = self.meta_model.predict_proba(l1_preds)[:, 1]

        return final_probs
