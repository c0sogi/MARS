import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from library.config import Config


class PLSSupervisedProjector(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies Supervised Partial Least Squares (PLS)
    dimensionality reduction to specific columns (embeddings) while passing
    others (metadata) through.

    This allows us to learn a latent space that maximizes covariance with the
    target variable, effectively extracting 'success' signals from the text.
    """

    def __init__(self, n_components=10, embedding_prefix="emb_"):
        self.n_components = n_components
        self.embedding_prefix = embedding_prefix
        self.pls = None
        self.embedding_cols_ = None
        self.metadata_cols_ = None

    def fit(self, X, y=None):
        """
        Fits the PLS model on the embedding columns using the target y.
        """
        if y is None:
            raise ValueError(
                "PLSSupervisedProjector requires target 'y' for supervised dimensionality reduction."
            )

        # Identify embedding columns and metadata columns
        if isinstance(X, pd.DataFrame):
            self.embedding_cols_ = [
                c for c in X.columns if str(c).startswith(self.embedding_prefix)
            ]
            self.metadata_cols_ = [
                c for c in X.columns if c not in self.embedding_cols_
            ]

            X_emb = X[self.embedding_cols_].values
            # X_meta is not needed for fitting PLS, but we track columns
        else:
            # Fallback for numpy arrays if column names are lost (assuming embeddings are first N columns)
            # This is risky, so we primarily support DataFrame input as per pipeline design
            raise TypeError(
                "PLSSupervisedProjector expects pandas DataFrame input to identify columns."
            )

        # Fit PLS
        self.pls = PLSRegression(
            n_components=self.n_components, scale=False
        )  # Data is already scaled/normalized
        self.pls.fit(X_emb, y)

        return self

    def transform(self, X):
        """
        Projects embedding columns using fitted PLS and concatenates with metadata.
        """
        if self.pls is None:
            raise RuntimeError("Transformer has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_emb = X[self.embedding_cols_].values
            X_meta = X[self.metadata_cols_].values
        else:
            raise TypeError("PLSSupervisedProjector expects pandas DataFrame input.")

        # Transform embeddings
        # PLSRegression transform returns X_scores
        X_pls = self.pls.transform(X_emb)

        # Concatenate: [PLS_Components, Metadata]
        X_transformed = np.hstack([X_pls, X_meta])

        return X_transformed


def build_linear_branch(
    C=1.0, penalty="l2", solver="liblinear", max_iter=2000, random_state=Config.SEED
):
    """
    Constructs the Linear Anchor branch (Logistic Regression).

    Args:
        C (float): Inverse of regularization strength.
        penalty (str): Regularization norm.
        solver (str): Optimization algorithm.
        max_iter (int): Maximum iterations.
        random_state (int): Seed.

    Returns:
        sklearn.pipeline.Pipeline: The linear classification pipeline.
    """
    # Note: Input X is already scaled (metadata) and normalized (embeddings).
    # We pass it directly to Logistic Regression.

    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipeline = Pipeline([("classifier", clf)])

    return pipeline


def build_kernel_branch(
    pls_n_components=10, svm_C=1.0, svm_gamma="scale", random_state=Config.SEED
):
    """
    Constructs the Non-Linear Expert branch (PLS + SVM).

    Args:
        pls_n_components (int): Number of latent components to extract from text.
        svm_C (float): SVM regularization parameter.
        svm_gamma (str/float): Kernel coefficient for RBF.
        random_state (int): Seed.

    Returns:
        sklearn.pipeline.Pipeline: The non-linear classification pipeline wrapped in calibration.
    """

    # 1. Feature Projection Step
    projector = PLSSupervisedProjector(
        n_components=pls_n_components, embedding_prefix="emb_"
    )

    # 2. SVM Classifier
    # We use probability=False here for speed, and wrap in CalibratedClassifierCV
    # to get probabilities via Sigmoid calibration (Platt scaling).
    svc = SVC(
        kernel="rbf",
        C=svm_C,
        gamma=svm_gamma,
        probability=False,
        class_weight="balanced",
        random_state=random_state,
    )

    # 3. Calibration Wrapper
    # Note: CalibratedClassifierCV with cv='prefit' expects the base estimator to be fitted.
    # Here we use cv=3 (or default) internal CV for calibration during fit,
    # effectively making this a self-contained probabilistic classifier.
    calibrated_svc = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=3)

    pipeline = Pipeline([("projector", projector), ("classifier", calibrated_svc)])

    return pipeline
