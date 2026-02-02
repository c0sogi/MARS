import os
import numpy as np
from sklearn.covariance import OAS
from library import config, utils


class OASTeacher:
    """
    Generative Model (Teacher) using Oracle Approximating Shrinkage (OAS) for LDA.
    Responsible for manifold estimation and synthetic data generation.
    """

    def __init__(self):
        self.means_ = None  # Class means (n_classes, n_features)
        self.covariance_ = None  # Shared covariance (n_features, n_features)
        self.precision_ = None  # Inverse covariance (n_features, n_features)
        self.priors_ = None  # Class priors (n_classes,)
        self.classes_ = None  # Unique class labels
        self.n_features_ = None

    def fit(self, X, y):
        """
        Fits the OAS-LDA model to the training data.

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training labels (n_samples,).
        """
        # Ensure high precision
        X = utils.enforce_float64(X)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize storage
        self.means_ = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)

        # 1. Compute Class Means and Priors
        # We also compute residuals for covariance estimation
        residuals = np.zeros_like(X, dtype=config.FLOAT_PRECISION)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]

            # Empirical mean
            mean_cls = np.mean(X_cls, axis=0)
            self.means_[idx] = mean_cls

            # Prior
            self.priors_[idx] = len(X_cls) / n_samples

            # Center data for this class
            residuals[mask] = X_cls - mean_cls

        # 2. Estimate Shared Covariance using OAS
        # assume_centered=True because we manually centered the data using class means
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.covariance_ = utils.enforce_float64(oas.covariance_)
        self.precision_ = utils.enforce_float64(oas.precision_)

        return self

    def generate_synthetic_data(self, n_samples_per_class=config.N_SYNTHETIC):
        """
        Generates synthetic data by sampling from the learned Multivariate Gaussian distributions.

        Args:
            n_samples_per_class (int): Number of samples to generate per class.

        Returns:
            tuple: (X_syn, y_syn)
        """
        if self.means_ is None or self.covariance_ is None:
            raise RuntimeError("Teacher must be fitted before generating data.")

        # Use Cholesky decomposition for faster sampling: X = Z * L.T + mu
        # Sigma = L * L.T
        # Z ~ N(0, I)
        rng = np.random.default_rng(config.RANDOM_SEED)
        L = np.linalg.cholesky(self.covariance_)

        n_classes = len(self.classes_)
        total_samples = n_classes * n_samples_per_class

        # Pre-allocate arrays
        X_syn = np.zeros(
            (total_samples, self.n_features_), dtype=config.FLOAT_PRECISION
        )
        y_syn = np.zeros(total_samples, dtype=int)  # Assuming integer encoded labels

        # Generate in batches per class to keep memory usage reasonable and logic simple
        start_idx = 0
        for idx, cls in enumerate(self.classes_):
            end_idx = start_idx + n_samples_per_class

            # Generate standard normal noise
            Z = rng.standard_normal((n_samples_per_class, self.n_features_))
            Z = utils.enforce_float64(Z)

            # Transform: X = Z @ L.T + mean
            # L is (n_features, n_features), Z is (n_samples, n_features)
            # Result is (n_samples, n_features)
            X_cls = Z @ L.T + self.means_[idx]

            X_syn[start_idx:end_idx] = X_cls
            y_syn[start_idx:end_idx] = cls

            start_idx = end_idx

        return X_syn, y_syn

    def get_analytic_weights(self):
        """
        Computes the analytic LDA weights and biases to initialize the Student model.

        For class k:
        w_k = Sigma^-1 * mu_k
        b_k = log(prior_k) - 0.5 * mu_k.T * Sigma^-1 * mu_k

        Returns:
            tuple: (coef, intercept)
            coef: (n_classes, n_features)
            intercept: (n_classes,)
        """
        if self.means_ is None or self.precision_ is None:
            raise RuntimeError("Teacher must be fitted before computing weights.")

        # coef = means @ precision.T (since precision is symmetric, same as means @ precision)
        # means: (n_classes, n_features)
        # precision: (n_features, n_features)
        # result: (n_classes, n_features)
        coef = self.means_ @ self.precision_

        # intercept calculation
        # term1: log(priors)
        term1 = np.log(self.priors_)

        # term2: -0.5 * diag(means @ precision @ means.T)
        # We can reuse coef: -0.5 * diag(coef @ means.T)
        # Or more efficiently: -0.5 * sum(coef * means, axis=1)
        term2 = -0.5 * np.sum(coef * self.means_, axis=1)

        intercept = term1 + term2

        return coef, intercept

    def predict_proba(self, X):
        """
        Predicts class probabilities using the Linear Formulation of LDA.
        Cite solution_lesson_node_00055: Linear Formulation avoids numerical saturation.
        Cite solution_lesson_node_00057: Float64 precision prevents performance ceiling.
        """
        if self.means_ is None:
            raise RuntimeError("Model not fitted.")

        X = utils.enforce_float64(X)

        # Get analytic weights (Linear Formulation)
        coef, intercept = self.get_analytic_weights()

        # Compute Linear Scores: X @ W.T + b
        scores = X @ coef.T + intercept

        # Softmax with numerical stability shift
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)

        # Normalize
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Clip to avoid extremes
        probs = np.clip(probs, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX)

        return probs

    def evaluate(self, X, y):
        """
        Evaluates the model on a dataset.
        """
        probs = self.predict_proba(X)
        loss = utils.calculate_log_loss(y, probs, self.classes_)
        return loss


def get_synthetic_data(X_train, y_train, load_cached_data=True):
    """
    Fits the teacher on real data and retrieves synthetic data (from cache or generation).

    Args:
        X_train (np.ndarray): Real training features.
        y_train (np.ndarray): Real training labels.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_syn, y_syn, teacher_instance)
    """
    # 1. Always fit the teacher first (computationally cheap)
    # This ensures we have the teacher instance ready for weight initialization
    # regardless of whether we load data from cache or generate it.
    teacher = OASTeacher()
    teacher.fit(X_train, y_train)

    # 2. Define Cache Paths
    cache_dir = config.WORKING_DIR
    x_syn_path = os.path.join(cache_dir, "X_synthetic.npy")
    y_syn_path = os.path.join(cache_dir, "y_synthetic.npy")

    # 3. Check Cache
    if load_cached_data and os.path.exists(x_syn_path) and os.path.exists(y_syn_path):
        print("Loading synthetic data from cache...")
        X_syn = np.load(x_syn_path)
        y_syn = np.load(y_syn_path)
        return utils.enforce_float64(X_syn), y_syn, teacher

    # 4. Generate Data if cache miss
    print(f"Generating synthetic data ({config.N_SYNTHETIC} samples per class)...")
    X_syn, y_syn = teacher.generate_synthetic_data(
        n_samples_per_class=config.N_SYNTHETIC
    )

    # 5. Save to Cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(x_syn_path, X_syn)
    np.save(y_syn_path, y_syn)
    print(f"Synthetic data saved to {cache_dir}")

    return X_syn, y_syn, teacher
