import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    GLOBAL_LDA_SOLVER,
    GLOBAL_LDA_SHRINKAGE,
    SUBMISSION_FILE_PATH,
)
from library.preprocessing import get_preprocessed_data
from library.utils import calculate_log_loss, save_submission


class GlobalLDAModel:
    """
    Implements a single Global Linear Discriminant Analysis model.

    Cite solution_lesson_node_00024: Ensembling on linearly separable data can
    degrade performance. A single regularized LDA model is sufficient and optimal
    for this high-dimensional, small-sample dataset.
    """

    def __init__(self):
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Trains the global LDA model.

        Args:
            X (array-like): Training features (n_samples, n_features).
            y (array-like): Training labels (n_samples,).
        """
        print("Training Global LDA Model...")
        self.model = LinearDiscriminantAnalysis(
            solver=GLOBAL_LDA_SOLVER, shrinkage=GLOBAL_LDA_SHRINKAGE
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        print("Training complete.")

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (array-like): Features (n_samples, n_features).

        Returns:
            array-like: Predicted probabilities (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        return self.model.predict_proba(X)


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
    model = GlobalLDAModel()
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
