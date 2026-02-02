import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from library.config import EXPERT_LIBRARY, FLOAT_PRECISION
from library.preprocessing import get_preprocessor


class OASLinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis with Oracle Approximating Shrinkage (OAS)
    for covariance estimation.

    This implementation fits an OAS covariance estimator on the centered data
    (pooled covariance assumption) and uses it to derive the LDA decision boundary.
    It operates strictly in the configured float precision.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the OAS-LDA model.

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Encode labels to 0..K-1
        self.le_ = LabelEncoder()
        y_idx = self.le_.fit_transform(y)

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # Center data for pooled covariance estimation
        # We compute X_centered = X - mu_y (where mu_y is the class mean)
        X_centered = np.empty_like(X, dtype=FLOAT_PRECISION)

        for i in range(n_classes):
            mask = y_idx == i
            X_k = X[mask]

            # Calculate class mean and prior
            self.means_[i] = X_k.mean(axis=0)
            self.priors_[i] = float(len(X_k)) / len(X)

            # Center the data
            X_centered[mask] = X_k - self.means_[i]

        # Estimate Covariance Matrix using OAS
        # We pass the centered data. OAS will center it again (subtracting ~0),
        # effectively computing the pooled covariance matrix of the residuals.
        oas = OAS()
        oas.fit(X_centered)
        self.covariance_ = oas.covariance_.astype(FLOAT_PRECISION)

        # Compute Precision Matrix (Inverse Covariance)
        # Using scipy.linalg.inv is safe for OAS shrunk matrices as they are well-conditioned.
        self.precision_ = scipy.linalg.inv(self.covariance_)

        # Compute Linear Coefficients (Weights) and Intercept (Bias) for the decision function
        # Formula:
        #   W_k = Sigma^-1 * mu_k
        #   b_k = -0.5 * mu_k^T * Sigma^-1 * mu_k + log(pi_k)

        # coef_: (n_classes, n_features)
        self.coef_ = np.dot(self.means_, self.precision_)

        # intercept_: (n_classes,)
        # term1 calculation: -0.5 * diag(means @ precision @ means.T)
        # Efficiently computed as: -0.5 * sum(means * coef, axis=1)
        term1 = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        term2 = np.log(self.priors_)
        self.intercept_ = term1 + term2

        return self

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        """
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(FLOAT_PRECISION)

        # Linear score: X @ W.T + b
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return scores

    def predict_proba(self, X):
        """
        Estimate probability.
        """
        scores = self.decision_function(X)

        # Softmax with numerical stability
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probas

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def build_expert_library():
    """
    Constructs the dictionary of expert pipelines based on the configuration.

    Returns:
        dict: A dictionary where keys are expert IDs and values are sklearn Pipelines.
    """
    experts = {}

    for config in EXPERT_LIBRARY:
        expert_id = config["id"]
        model_type = config["model_type"]
        shrinkage = config["shrinkage"]
        prep_name = config["preprocessing"]

        # 1. Get Preprocessor
        scaler = get_preprocessor(prep_name)

        # 2. Get Classifier
        if model_type == "lda_fixed":
            # LDA with fixed shrinkage
            # solver='lsqr' supports shrinkage
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        elif model_type == "lda_lw":
            # LDA with Ledoit-Wolf shrinkage (auto)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif model_type == "lda_oas":
            # LDA with OAS covariance estimation (Custom Implementation)
            clf = OASLinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 3. Build Pipeline
        # Steps: Preprocessing -> Classifier
        pipeline = Pipeline([("scaler", scaler), ("clf", clf)])

        experts[expert_id] = pipeline

    return experts
