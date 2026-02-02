import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from library.config import (
    LR_PARAMS,
    LDA_PARAMS,
    GPC_PARAMS,
    PCA_EXPLAINED_VARIANCE,
    RANDOM_SEED,
)


def make_logistic_cv():
    """
    Creates a LogisticRegressionCV instance using configuration from library.config.
    """
    params = LR_PARAMS.copy()
    # Ensure random_state is set
    if "random_state" not in params:
        params["random_state"] = RANDOM_SEED

    return LogisticRegressionCV(**params)


def make_lda():
    """
    Creates a LinearDiscriminantAnalysis instance using configuration from library.config.
    """
    return LinearDiscriminantAnalysis(**LDA_PARAMS)


def make_gpc_pipeline():
    """
    Creates a Pipeline consisting of PCA and GaussianProcessClassifier.
    The PCA component preserves the variance specified in PCA_EXPLAINED_VARIANCE.
    The GPC uses an RBF kernel and parameters from GPC_PARAMS.
    """
    # Initialize RBF kernel.
    # 1.0 * RBF(1.0) allows the optimizer to tune both the variance (constant kernel)
    # and the length scale.
    kernel = 1.0 * RBF(length_scale=1.0)

    gpc_params = GPC_PARAMS.copy()
    gpc_params["kernel"] = kernel
    if "random_state" not in gpc_params:
        gpc_params["random_state"] = RANDOM_SEED

    # PCA for dimensionality reduction on the GPC branch
    pca = PCA(n_components=PCA_EXPLAINED_VARIANCE, random_state=RANDOM_SEED)

    gpc = GaussianProcessClassifier(**gpc_params)

    pipeline = Pipeline([("pca", pca), ("gpc", gpc)])

    return pipeline


class HybridEnsemble(BaseEstimator, ClassifierMixin):
    """
    A Soft-Voting Ensemble combining:
    1. Discriminative Linear Model (Logistic Regression)
    2. Generative Linear Model (LDA)
    3. Probabilistic Non-Linear Model (GPC on PCA features)
    """

    def __init__(self):
        self.lr = make_logistic_cv()
        self.lda = make_lda()
        self.gpc = make_gpc_pipeline()
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits all constituent models on the provided dataset.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.

        Returns:
            self
        """
        print("Starting HybridEnsemble training...")

        # 1. Logistic Regression
        print("Fitting Logistic Regression (Discriminative Linear)...")
        self.lr.fit(X, y)
        print("Logistic Regression training complete.")

        # 2. LDA
        print("Fitting LDA (Generative Linear)...")
        self.lda.fit(X, y)
        print("LDA training complete.")

        # 3. GPC
        print("Fitting GPC Pipeline (Probabilistic Non-Linear)...")
        self.gpc.fit(X, y)
        n_components = self.gpc.named_steps["pca"].n_components_
        print(f"GPC training complete. PCA retained {n_components} components.")

        # Store classes from one of the estimators
        self.classes_ = self.lr.classes_

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using soft voting (averaging).

        Args:
            X (np.ndarray): Features to predict.

        Returns:
            np.ndarray: Averaged probability estimates of shape (n_samples, n_classes).
        """
        # Get probabilities from each component
        # Each returns (n_samples, n_classes)
        probs_lr = self.lr.predict_proba(X)
        probs_lda = self.lda.predict_proba(X)
        probs_gpc = self.gpc.predict_proba(X)

        # Average the probabilities
        avg_probs = (probs_lr + probs_lda + probs_gpc) / 3.0

        return avg_probs

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Args:
            X (np.ndarray): Features to predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
