import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from library.preprocessor import process_and_cache_data
from library.utils import calculate_log_loss, save_submission


class OASLinearDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis using Oracle Approximating Shrinkage (OAS).

    This implementation focuses on maximum numerical precision and stability
    for high-dimensional, small-sample regimes. It avoids empirical covariance
    estimation and uses the linear formulation of the discriminant function.
    """

    def __init__(self):
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.means_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation.
        """
        # Enforce float64 precision (Cite solution_lesson_node_00073)
        X = X.astype(np.float64)

        # Validate inputs
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # 1. Compute Common Statistics
        # Map labels to 0..K-1
        le = LabelEncoder()
        y_idx = le.fit_transform(y)

        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # Compute means and priors
        for k in range(n_classes):
            mask = y_idx == k
            X_k = X[mask]
            self.means_[k] = np.mean(X_k, axis=0)
            self.priors_[k] = X_k.shape[0] / n_samples

        # Compute Centered Residuals R = X - mu_y
        # This centers the data around the class means, effectively pooling covariance
        R = X - self.means_[y_idx]

        # 2. Covariance Estimation via OAS
        # Use assume_centered=True to respect the geometric centroids (Cite solution_lesson_node_00061)
        # Use OAS over Ledoit-Wolf for Gaussianized data (Cite solution_lesson_node_00047)
        oas = OAS(assume_centered=True)
        oas.fit(R)

        # Use precision_ directly instead of solving linear systems (Cite solution_lesson_node_00062)
        prec_oas = oas.precision_  # Shape (n_features, n_features)

        # 3. Compute Linear Parameters (Linear Formulation)
        # Cite solution_lesson_node_00055
        # W = Sigma^-1 * mu  -> Shape (n_classes, n_features)
        self.coef_ = np.dot(self.means_, prec_oas)

        # b = -0.5 * diag(mu.T * Sigma^-1 * mu) + log(prior)
        #   = -0.5 * diag(mu * W.T) + log(prior)
        # Element-wise mult and sum across features gives the diagonal of the product
        term1 = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        self.intercept_ = term1 + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities using the linear decision boundary.
        """
        check_is_fitted(self)
        # Enforce float64 precision for inference (Cite solution_lesson_node_00057)
        X = check_array(X).astype(np.float64)

        # Compute Logits: Z = X @ W.T + b
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax with stability shift
        # Shift logits by max per row to avoid overflow in exp
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits_shifted = logits - logits_max

        exp_logits = np.exp(logits_shifted)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)

        probas = exp_logits / sum_exp

        return probas

    def predict(self, X):
        """
        Predicts class labels.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def run_training_pipeline(load_cached_data=True):
    """
    Executes the full training, evaluation, and submission pipeline.
    """
    print(
        "Initializing Dual-Covariance Precision-Interpolated Discriminant Pipeline..."
    )

    # 1. Load and Preprocess Data
    # This handles loading raw data, applying Yeo-Johnson + StandardScaler, and caching
    data = process_and_cache_data(load_cached_data=load_cached_data)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    ids_test = data["ids_test"]
    classes = data["classes"]

    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Initialize Model
    # Using default jitter of 1e-6 as specified in the strategy
    model = DualExpertLDA(empirical_jitter=1e-6)

    # 3. Train Model
    print("Fitting DualExpertLDA model...")
    model.fit(X_train, y_train)

    # 4. Evaluate on Validation Set
    print("Evaluating on Validation set...")
    val_probas = model.predict_proba(X_val)

    # Calculate Log Loss
    # Note: y_val contains indices, calculate_log_loss expects indices or labels
    # Our utils function handles the clipping and rescaling required by the metric
    val_loss = calculate_log_loss(y_val, val_probas)

    print("-" * 30)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")
    print("-" * 30)

    # 5. Generate Test Predictions
    print("Generating predictions for Test set...")
    test_probas = model.predict_proba(X_test)

    # 6. Save Submission
    save_submission(ids_test, test_probas, classes)

    return val_loss
