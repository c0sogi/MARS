import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, accuracy_score

from library.config import Config
from library.utils import seed_everything, save_submission
from library.preprocessing import FeaturePipeline
from library.data_loader import _get_class_mapping


class RandomSubspaceLDA:
    """
    A Random Subspace Ensemble of Linear Discriminant Analysis (LDA) models.

    This ensemble trains multiple LDA classifiers, each on a random subset of
    the input features but using the full training sample set. This technique
    (Feature Bagging) helps stabilize covariance estimation in high-dimensional
    spaces and improves generalization.
    """

    def __init__(
        self,
        n_estimators=Config.N_ESTIMATORS,
        subspace_fraction=Config.SUBSPACE_FRACTION,
        solver=Config.LDA_SOLVER,
        shrinkage=Config.LDA_SHRINKAGE,
        random_state=Config.SEED,
    ):
        """
        Args:
            n_estimators (int): Number of LDA models in the ensemble.
            subspace_fraction (float): Fraction of features to use for each estimator (0.0 < f <= 1.0).
            solver (str): Solver to use for LDA ('lsqr' or 'eigen' required for shrinkage).
            shrinkage (str or float): Shrinkage parameter for LDA ('auto' recommended).
            random_state (int): Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.subspace_fraction = subspace_fraction
        self.solver = solver
        self.shrinkage = shrinkage
        self.random_state = random_state

        self.estimators_ = []
        self.feature_indices_ = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the ensemble of LDA models.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples,).

        Returns:
            self
        """
        # Ensure reproducibility
        seed_everything(self.random_state)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_subset = int(n_features * self.subspace_fraction)

        # Ensure at least one feature is selected
        n_subset = max(1, n_subset)

        self.estimators_ = []
        self.feature_indices_ = []

        print(f"Training RandomSubspaceLDA with {self.n_estimators} estimators...")
        print(f"  Input features: {n_features}")
        print(f"  Features per estimator: {n_subset} ({self.subspace_fraction:.2%})")

        for i in range(self.n_estimators):
            # Use a deterministic seed for each estimator derived from the global seed
            # This ensures that the i-th estimator always sees the same feature subset
            rng = np.random.RandomState(self.random_state + i)

            # Select random features without replacement
            indices = rng.choice(n_features, n_subset, replace=False)
            self.feature_indices_.append(indices)

            # Extract feature subset
            X_subset = X[:, indices]

            # Initialize and train LDA
            clf = LinearDiscriminantAnalysis(
                solver=self.solver, shrinkage=self.shrinkage
            )
            clf.fit(X_subset, y)
            self.estimators_.append(clf)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if not self.estimators_:
            raise RuntimeError("Model not fitted yet.")

        # Accumulate probabilities from all estimators
        sum_proba = None

        for clf, indices in zip(self.estimators_, self.feature_indices_):
            # Extract the same feature subset used during training
            X_subset = X[:, indices]

            # Predict
            proba = clf.predict_proba(X_subset)

            if sum_proba is None:
                sum_proba = proba
            else:
                sum_proba += proba

        # Compute arithmetic mean
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
        # Map indices back to original classes (though usually indices are 0..K-1)
        return self.classes_[np.argmax(proba, axis=1)]


def train_evaluate_predict(load_cached_data=True):
    """
    Main execution function to run the training pipeline.

    1. Loads processed data (Train, Val, Test) via FeaturePipeline.
    2. Trains the RandomSubspaceLDA ensemble.
    3. Evaluates performance on the Validation set.
    4. Generates predictions for the Test set.
    5. Saves the submission file.

    Args:
        load_cached_data (bool): Whether to attempt loading features from cache.
    """
    # Set global seed
    seed_everything(Config.SEED)

    print("========================================")
    print("Starting Random Subspace LDA Pipeline")
    print("========================================")

    # 1. Initialize Pipeline and Load Data
    pipeline = FeaturePipeline()

    print("\n[1/5] Loading Training Data...")
    X_train, y_train, _ = pipeline.get_processed_data(
        "train", load_cached_data=load_cached_data
    )

    print("\n[2/5] Loading Validation Data...")
    X_val, y_val, _ = pipeline.get_processed_data(
        "val", load_cached_data=load_cached_data
    )

    # 2. Train Model
    print("\n[3/5] Training Ensemble Model...")
    model = RandomSubspaceLDA(
        n_estimators=Config.N_ESTIMATORS,
        subspace_fraction=Config.SUBSPACE_FRACTION,
        solver=Config.LDA_SOLVER,
        shrinkage=Config.LDA_SHRINKAGE,
        random_state=Config.SEED,
    )

    model.fit(X_train, y_train)

    # 3. Evaluate
    print("\n[4/5] Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)

    # Calculate metrics
    # y_val contains class indices (0..98).
    # model.classes_ contains the sorted unique class indices found in training.
    val_loss = log_loss(y_val, val_probs, labels=model.classes_)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Validation Log Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    # 4. Predict on Test and Save
    print("\n[5/5] Generating Test Predictions...")
    X_test, _, ids_test = pipeline.get_processed_data(
        "test", load_cached_data=load_cached_data
    )

    test_probs = model.predict_proba(X_test)

    # Retrieve class names for column headers
    # _get_class_mapping returns (dict, list_of_names)
    _, class_names = _get_class_mapping(load_cached_data=load_cached_data)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(ids_test, test_probs, class_names, filename=Config.SUBMISSION_PATH)

    print("Pipeline completed successfully.")
