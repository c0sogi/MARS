import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("model_components")


class DifferentialScaler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that applies differential scaling to tabular features
    while preserving text embeddings.

    It splits the input into text and tabular components based on `tabular_start_idx`.
    The tabular component undergoes QuantileTransformation (RankGauss) and is then
    multiplied by a scalar `alpha`. The text component is passed through unchanged.

    This allows a downstream linear model to apply different effective regularization
    strengths to the two modalities.
    """

    def __init__(self, alpha=1.0, tabular_start_idx=Config.EMBEDDING_DIM):
        """
        Args:
            alpha (float): Scalar multiplier for tabular features.
            tabular_start_idx (int): Column index where tabular features begin.
        """
        self.alpha = alpha
        self.tabular_start_idx = tabular_start_idx
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )

    def fit(self, X, y=None):
        """
        Fits the QuantileTransformer on the tabular portion of X.
        """
        # Check if we have tabular features
        if X.shape[1] > self.tabular_start_idx:
            X_tab = X[:, self.tabular_start_idx :]
            self.qt.fit(X_tab)
        return self

    def transform(self, X):
        """
        Transforms the tabular portion of X and concatenates with text features.
        """
        # Split features
        X_text = X[:, : self.tabular_start_idx]

        if X.shape[1] > self.tabular_start_idx:
            X_tab = X[:, self.tabular_start_idx :]

            # Apply RankGauss
            X_tab_trans = self.qt.transform(X_tab)

            # Apply Differential Scaling
            # Scaling up features -> Larger values -> Smaller weights in Linear Model -> Less L2 Penalty
            X_tab_scaled = X_tab_trans * self.alpha

            # Recombine
            return np.hstack([X_text, X_tab_scaled])

        return X_text


def build_pipeline(
    C=1.0,
    alpha=1.0,
    class_weight=None,
    max_iter=Config.LOGREG_MAX_ITER,
):
    """
    Constructs the training pipeline consisting of the DifferentialScaler
    and a Bagged Logistic Regression classifier.

    Args:
        C (float): Inverse regularization strength for Logistic Regression.
        alpha (float): Scaling factor for tabular features.
        class_weight (str or dict): Class weights for Logistic Regression.
        max_iter (int): Maximum iterations for the solver.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # 1. Initialize Differential Scaler
    scaler = DifferentialScaler(alpha=alpha, tabular_start_idx=Config.EMBEDDING_DIM)

    # 2. Initialize Base Estimator (Logistic Regression)
    logreg = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=Config.LOGREG_SOLVER,
        max_iter=max_iter,
        random_state=Config.RANDOM_SEED,
        penalty=Config.LOGREG_PENALTY,
    )

    # 3. Initialize Bagging Ensemble
    # Note: 'estimator' is the parameter name for scikit-learn >= 1.2
    bagging = BaggingClassifier(
        estimator=logreg,
        n_estimators=Config.BAGGING_N_ESTIMATORS,
        max_samples=Config.BAGGING_MAX_SAMPLES,
        random_state=Config.RANDOM_SEED,
        n_jobs=1,  # Set to 1 to avoid nested parallelism conflicts with outer CV loops
    )

    # 4. Construct Pipeline
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("clf", bagging),
        ]
    )

    return pipeline
