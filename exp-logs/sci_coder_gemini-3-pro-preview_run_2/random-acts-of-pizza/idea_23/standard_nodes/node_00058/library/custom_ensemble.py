import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from library.utils import setup_logger


class StratifiedRandomSubspaceEnsemble:
    """
    A custom ensemble classifier that applies Stratified Random Subspace Bagging.

    Strategy:
    1. Bagging: Each base learner is trained on a bootstrap sample of the data.
    2. Feature Stratification:
       - Tabular Metadata: 100% retention (always included).
       - Text Embeddings: Random subspace sampling (controlled by subspace_fraction).

    This ensures critical metadata signals are preserved while regularizing high-dimensional text.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        subspace_fraction: float = 0.5,
        C: float = 1.0,
        class_weight=None,
        random_state: int = 42,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        """
        Args:
            n_estimators (int): Number of base Logistic Regression models.
            subspace_fraction (float): Fraction of text embedding features to sample for each learner.
            C (float): Inverse regularization strength for the base Logistic Regression.
            class_weight (dict or 'balanced'): Weights associated with classes.
            random_state (int): Seed for reproducibility.
            n_jobs (int): Number of CPU cores to use for base learners (passed to LogisticRegression).
            verbose (int): Verbosity level.
        """
        self.n_estimators = n_estimators
        self.subspace_fraction = subspace_fraction
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.logger = setup_logger("SRSLEnsemble")
        self.estimators_ = []
        self.feature_masks_ = (
            []
        )  # Stores indices of text features used by each estimator
        self.rng = np.random.RandomState(random_state)

    def fit(self, X_text: np.ndarray, X_tabular: np.ndarray, y: np.ndarray):
        """
        Fits the ensemble of Logistic Regression models.

        Args:
            X_text (np.ndarray): Text embeddings (N_samples, N_text_features).
            X_tabular (np.ndarray): Tabular metadata (N_samples, N_tabular_features).
            y (np.ndarray): Target labels (N_samples,).

        Returns:
            self
        """
        n_samples = X_text.shape[0]
        n_text_features = X_text.shape[1]
        n_subspace_text = int(n_text_features * self.subspace_fraction)

        # Ensure consistency
        if X_tabular.shape[0] != n_samples:
            raise ValueError(
                "X_text and X_tabular must have the same number of samples."
            )

        self.estimators_ = []
        self.feature_masks_ = []

        if self.verbose > 0:
            self.logger.info(f"Training SRSLE with {self.n_estimators} estimators...")

        for i in range(self.n_estimators):
            # 1. Bootstrap Sampling (Instance Bagging)
            # We use a distinct seed for each resampling to ensure diversity
            seed = self.rng.randint(0, 100000)
            indices = resample(
                np.arange(n_samples),
                replace=True,
                n_samples=n_samples,
                random_state=seed,
            )

            # 2. Feature Subspace Sampling (Feature Stratification)
            # Select random subset of text features
            text_feature_indices = self.rng.choice(
                n_text_features, size=n_subspace_text, replace=False
            )

            # Store the mask (indices) for inference
            self.feature_masks_.append(text_feature_indices)

            # 3. Construct Training Data for this Base Learner
            # Concatenate: [Selected Text Features, All Tabular Features]
            X_text_subset = X_text[indices][:, text_feature_indices]
            X_tabular_subset = X_tabular[indices]

            # Horizontal stack
            X_combined = np.hstack([X_text_subset, X_tabular_subset])
            y_subset = y[indices]

            # 4. Train Base Learner
            # We use L2 penalty (Ridge) as per design
            clf = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
                n_jobs=1,  # We run the loop sequentially, so keep base learner single-threaded or low overhead
            )

            clf.fit(X_combined, y_subset)
            self.estimators_.append(clf)

            if self.verbose > 1 and (i + 1) % 10 == 0:
                self.logger.info(f"Trained estimator {i + 1}/{self.n_estimators}")

        return self

    def predict_proba(self, X_text: np.ndarray, X_tabular: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X_text (np.ndarray): Text embeddings.
            X_tabular (np.ndarray): Tabular metadata.

        Returns:
            np.ndarray: Predicted probabilities of shape (N_samples, 2).
        """
        # Check if fitted
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet.")

        n_samples = X_text.shape[0]

        # Accumulator for probabilities (summing prob of class 1)
        # We'll store full proba array to be safe (N, 2)
        proba_sum = np.zeros((n_samples, 2))

        for clf, text_indices in zip(self.estimators_, self.feature_masks_):
            # 1. Reconstruct Feature Space
            # Select the specific text features used by this estimator
            X_text_subset = X_text[:, text_indices]

            # Concatenate with full tabular data
            X_combined = np.hstack([X_text_subset, X_tabular])

            # 2. Predict
            proba = clf.predict_proba(X_combined)

            # 3. Accumulate
            proba_sum += proba

        # Average
        avg_proba = proba_sum / self.n_estimators

        return avg_proba

    def predict(self, X_text: np.ndarray, X_tabular: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.

        Args:
            X_text (np.ndarray): Text embeddings.
            X_tabular (np.ndarray): Tabular metadata.

        Returns:
            np.ndarray: Predicted labels (0 or 1).
        """
        proba = self.predict_proba(X_text, X_tabular)
        return np.argmax(proba, axis=1)
