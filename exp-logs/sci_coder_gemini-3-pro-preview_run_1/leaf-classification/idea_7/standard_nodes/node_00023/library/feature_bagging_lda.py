import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from library.config import (
    SEED,
    N_ESTIMATORS,
    FEATURE_SUBSAMPLE_RATE,
    LDA_SOLVER,
    LDA_SHRINKAGE,
)
from library.utils import set_seed, calculate_log_loss


class FeatureBaggingLDAEnsemble:
    """
    A Homogeneous Ensemble of Linear Discriminant Analysis (LDA) estimators
    using Random Subspace Sampling (Feature Bagging).

    Attributes:
        n_estimators (int): Number of LDA models in the ensemble.
        subsample_rate (float): Fraction of features to use for each estimator.
        random_state (int): Seed for reproducibility.
        estimators_ (list): List of tuples (trained_model, feature_indices).
        classes_ (np.ndarray): Unique class labels.
    """

    def __init__(
        self,
        n_estimators=N_ESTIMATORS,
        subsample_rate=FEATURE_SUBSAMPLE_RATE,
        random_state=SEED,
    ):
        self.n_estimators = n_estimators
        self.subsample_rate = subsample_rate
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the ensemble of LDA models.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples,).

        Returns:
            self: The fitted estimator.
        """
        # Ensure reproducibility
        rng = np.random.RandomState(self.random_state)
        set_seed(self.random_state)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_features_subset = int(n_features * self.subsample_rate)

        # Ensure at least one feature is selected
        n_features_subset = max(1, n_features_subset)

        self.estimators_ = []

        print(
            f"Training FeatureBaggingLDAEnsemble with {self.n_estimators} estimators..."
        )
        print(
            f"Feature Subsampling: {n_features_subset}/{n_features} features per estimator."
        )

        for i in range(self.n_estimators):
            # 1. Generate Random Feature Mask
            # We use choice without replacement to select indices
            feature_indices = rng.choice(
                n_features, size=n_features_subset, replace=False
            )

            # Sort indices for consistency (though not strictly necessary for LDA)
            feature_indices.sort()

            # 2. Subset Data
            X_subset = X[:, feature_indices]

            # 3. Initialize and Fit Base Estimator
            # Using 'lsqr' solver and 'auto' shrinkage (Ledoit-Wolf)
            clf = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)
            clf.fit(X_subset, y)

            # 4. Store Model and Mask
            self.estimators_.append((clf, feature_indices))

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the ensemble.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet.")

        # Initialize sum of probabilities
        # Assuming all estimators predict same classes in same order
        # (guaranteed if trained on same y vector)
        sum_proba = None

        for clf, feature_indices in self.estimators_:
            # Subset features using the stored mask
            X_subset = X[:, feature_indices]

            # Get probabilities
            proba = clf.predict_proba(X_subset)

            if sum_proba is None:
                sum_proba = proba
            else:
                sum_proba += proba

        # Average probabilities
        avg_proba = sum_proba / len(self.estimators_)

        return avg_proba

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def train_and_evaluate(X_train, y_train, X_val, y_val, classes):
    """
    Helper function to train the ensemble and evaluate on validation data.
    Prints metrics with full precision.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        classes (np.ndarray): List of all possible classes (for log loss calculation).

    Returns:
        model: The trained FeatureBaggingLDAEnsemble instance.
    """
    print("\n--- Starting Ensemble Training ---")

    # Initialize model
    model = FeatureBaggingLDAEnsemble(
        n_estimators=N_ESTIMATORS,
        subsample_rate=FEATURE_SUBSAMPLE_RATE,
        random_state=SEED,
    )

    # Fit model
    model.fit(X_train, y_train)

    # Evaluate on Training Data
    print("\n--- Evaluating on Training Set ---")
    train_proba = model.predict_proba(X_train)
    train_pred = model.classes_[np.argmax(train_proba, axis=1)]

    train_loss = calculate_log_loss(y_train, train_proba, labels=classes)
    train_acc = accuracy_score(y_train, train_pred)

    print(f"Train Log Loss: {train_loss}")
    print(f"Train Accuracy: {train_acc}")

    # Evaluate on Validation Data
    print("\n--- Evaluating on Validation Set ---")
    val_proba = model.predict_proba(X_val)
    val_pred = model.classes_[np.argmax(val_proba, axis=1)]

    val_loss = calculate_log_loss(y_val, val_proba, labels=classes)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Validation Log Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    return model
