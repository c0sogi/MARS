import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    GLOBAL_LDA_SOLVER,
    GLOBAL_LDA_SHRINKAGE,
    SUBSPACE_LDA_SOLVER,
    SUBSPACE_LDA_SHRINKAGE,
    N_ESTIMATORS,
    FEATURE_FRACTION,
    WEIGHT_GLOBAL,
    WEIGHT_SUBSPACE,
    SEED,
    SUBMISSION_FILE_PATH,
)
from library.preprocessing import get_preprocessed_data
from library.utils import calculate_log_loss, save_submission


class HybridLDAEnsemble:
    """
    Implements the Hybrid Global-Subspace LDA Ensemble.

    Components:
    1. Global Expert: LDA trained on all features.
    2. Subspace Expert: Ensemble of LDA models trained on random feature subsets.
    """

    def __init__(self):
        self.global_model = None
        self.subspace_models = []
        self.subspace_masks = []
        self.classes_ = None
        self.rng = np.random.RandomState(SEED)

    def fit(self, X, y):
        """
        Trains the ensemble components.

        Args:
            X (array-like): Training features (n_samples, n_features).
            y (array-like): Training labels (n_samples,).
        """
        n_features = X.shape[1]
        n_subspace_features = int(n_features * FEATURE_FRACTION)

        # 1. Train Global Expert
        print("Training Global Expert...")
        self.global_model = LinearDiscriminantAnalysis(
            solver=GLOBAL_LDA_SOLVER, shrinkage=GLOBAL_LDA_SHRINKAGE
        )
        self.global_model.fit(X, y)
        self.classes_ = self.global_model.classes_

        # 2. Train Subspace Expert
        print(f"Training Subspace Expert ({N_ESTIMATORS} estimators)...")
        self.subspace_models = []
        self.subspace_masks = []

        for i in range(N_ESTIMATORS):
            # Generate random feature mask
            # We select indices without replacement
            feature_indices = self.rng.choice(
                n_features, size=n_subspace_features, replace=False
            )

            # Create boolean mask for easier slicing later if needed,
            # but storing indices is efficient
            mask = feature_indices

            # Subset features
            X_subset = X[:, mask]

            # Initialize and fit LDA
            model = LinearDiscriminantAnalysis(
                solver=SUBSPACE_LDA_SOLVER, shrinkage=SUBSPACE_LDA_SHRINKAGE
            )
            model.fit(X_subset, y)

            self.subspace_models.append(model)
            self.subspace_masks.append(mask)

        print("Training complete.")

    def predict_proba(self, X):
        """
        Predicts class probabilities using the weighted ensemble.

        Args:
            X (array-like): Features (n_samples, n_features).

        Returns:
            array-like: Predicted probabilities (n_samples, n_classes).
        """
        if self.global_model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        # 1. Global Expert Predictions
        P_global = self.global_model.predict_proba(X)

        # 2. Subspace Expert Predictions
        P_subspace_accum = np.zeros_like(P_global)

        for model, mask in zip(self.subspace_models, self.subspace_masks):
            X_subset = X[:, mask]
            P_subspace_accum += model.predict_proba(X_subset)

        P_subspace = P_subspace_accum / len(self.subspace_models)

        # 3. Weighted Aggregation
        P_final = (WEIGHT_GLOBAL * P_global) + (WEIGHT_SUBSPACE * P_subspace)

        return P_final


def run_modeling(load_cached_data=True):
    """
    Orchestrates the modeling pipeline: data loading, training, validation, and submission.
    """
    # 1. Load Preprocessed Data
    # This function handles caching internally as per requirements
    X_train, y_train, X_val, y_val, X_test, test_ids = get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    print(f"Data Loaded: Train shape {X_train.shape}, Val shape {X_val.shape}")

    # 2. Initialize and Train Model
    model = HybridLDAEnsemble()
    model.fit(X_train, y_train)

    # 3. Validation
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss using the provided utility
    # We pass model.classes_ to ensure correct column mapping
    val_loss = calculate_log_loss(y_val, val_probs, model.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 4. Generate Submission
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)

    save_submission(
        ids=test_ids,
        class_labels=model.classes_,
        probabilities=test_probs,
        output_path=SUBMISSION_FILE_PATH,
    )
