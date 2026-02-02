import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS, LedoitWolf
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from library import config


class CovarianceLDA(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis classifier using robust covariance estimators
    (OAS or Ledoit-Wolf) for the pooled covariance matrix.

    This implementation assumes homoscedasticity (shared covariance matrix across classes),
    which is the defining characteristic of LDA vs QDA. It calculates the pooled
    covariance matrix using the specified robust estimator and then applies the
    standard LDA decision rule.
    """

    def __init__(self, method="oas"):
        self.method = method
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_ = None
        self.precision_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data and parameters.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # Center data for pooled covariance estimation
        X_centered = np.zeros_like(X, dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]

            # Estimate class mean and prior
            self.means_[idx] = np.mean(X_cls, axis=0)
            self.priors_[idx] = float(len(X_cls)) / len(X)

            # Center the class data
            X_centered[mask] = X_cls - self.means_[idx]

        # Estimate Pooled Covariance Matrix
        if self.method == "oas":
            cov_estimator = OAS()
        elif self.method == "lw":
            cov_estimator = LedoitWolf()
        else:
            raise ValueError(f"Unknown covariance estimation method: {self.method}")

        cov_estimator.fit(X_centered)
        self.covariance_ = cov_estimator.covariance_
        self.precision_ = cov_estimator.precision_

        # Calculate LDA Parameters
        # The linear discriminant function for class k is:
        # delta_k(x) = x.T * Sigma^-1 * mu_k - 0.5 * mu_k.T * Sigma^-1 * mu_k + log(pi_k)
        # We can rewrite this as a linear score: x * coef_.T + intercept_

        # coef_[k] = Sigma^-1 * mu_k
        # Note: self.precision_ is symmetric
        self.coef_ = np.dot(
            self.means_, self.precision_
        )  # Shape: (n_classes, n_features)

        # intercept_[k] = -0.5 * mu_k.T * Sigma^-1 * mu_k + log(pi_k)
        # The quadratic term is the diagonal of (means_ * precision_ * means_.T)
        # Which is equivalent to row-wise dot product of means_ and coef_
        quad_term = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        self.intercept_ = quad_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Estimate probability.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Compute linear scores
        scores = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax with numerical stability
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def build_expert_library():
    """
    Constructs the library of probabilistic experts based on the Dual-Basis architecture.

    Returns:
        list: A list of dictionaries, each containing:
            - 'id': Unique identifier string.
            - 'model': The scikit-learn pipeline.
            - 'view': The feature view to use ('global' or 'macro').
    """
    experts = []

    # =========================================================================
    # Basis A: Parametric Gaussian Anchors (Global View)
    # Preprocessing: PowerTransformer (Yeo-Johnson)
    # Algorithm: LDA with OAS, Ledoit-Wolf, and Fixed Shrinkage
    # =========================================================================

    # 1. LDA with Ledoit-Wolf (Auto)
    experts.append(
        {
            "id": "BasisA_LDA_LW",
            "view": "global",
            "model": Pipeline(
                [
                    ("pt", PowerTransformer(method="yeo-johnson")),
                    (
                        "lda",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    ),
                ]
            ),
        }
    )

    # 2. LDA with OAS (Custom)
    experts.append(
        {
            "id": "BasisA_LDA_OAS",
            "view": "global",
            "model": Pipeline(
                [
                    ("pt", PowerTransformer(method="yeo-johnson")),
                    ("lda", CovarianceLDA(method="oas")),
                ]
            ),
        }
    )

    # 3. LDA with Fixed Shrinkage
    for shrink in config.FIXED_SHRINKAGE_VALUES:
        experts.append(
            {
                "id": f"BasisA_LDA_Fixed_{shrink}",
                "view": "global",
                "model": Pipeline(
                    [
                        ("pt", PowerTransformer(method="yeo-johnson")),
                        (
                            "lda",
                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink),
                        ),
                    ]
                ),
            }
        )

    # =========================================================================
    # Basis B: Constrained Non-Parametric Experts (Global View)
    # Preprocessing: QuantileTransformer (n_quantiles=30, output=normal)
    # Algorithm: LDA with OAS, Ledoit-Wolf, and Fixed Shrinkage
    # =========================================================================

    # 1. LDA with Ledoit-Wolf
    experts.append(
        {
            "id": "BasisB_LDA_LW",
            "view": "global",
            "model": Pipeline(
                [
                    (
                        "qt",
                        QuantileTransformer(
                            n_quantiles=config.N_QUANTILES,
                            output_distribution="normal",
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                    (
                        "lda",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    ),
                ]
            ),
        }
    )

    # 2. LDA with OAS
    experts.append(
        {
            "id": "BasisB_LDA_OAS",
            "view": "global",
            "model": Pipeline(
                [
                    (
                        "qt",
                        QuantileTransformer(
                            n_quantiles=config.N_QUANTILES,
                            output_distribution="normal",
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                    ("lda", CovarianceLDA(method="oas")),
                ]
            ),
        }
    )

    # 3. LDA with Fixed Shrinkage
    for shrink in config.FIXED_SHRINKAGE_VALUES:
        experts.append(
            {
                "id": f"BasisB_LDA_Fixed_{shrink}",
                "view": "global",
                "model": Pipeline(
                    [
                        (
                            "qt",
                            QuantileTransformer(
                                n_quantiles=config.N_QUANTILES,
                                output_distribution="normal",
                                random_state=config.RANDOM_SEED,
                            ),
                        ),
                        (
                            "lda",
                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink),
                        ),
                    ]
                ),
            }
        )

    # =========================================================================
    # Basis C: Orthogonal Morphometric Experts (Macro View)
    # Preprocessing: PowerTransformer
    # Algorithm: LDA (Ledoit-Wolf)
    # =========================================================================

    experts.append(
        {
            "id": "BasisC_LDA_LW",
            "view": "macro",
            "model": Pipeline(
                [
                    ("pt", PowerTransformer(method="yeo-johnson")),
                    (
                        "lda",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    ),
                ]
            ),
        }
    )

    return experts
