import numpy as np
import scipy.sparse
from library.utils import save_artifact, load_artifact


class VectorizedMNB:
    """
    A Vectorized One-vs-Rest Multinomial Naive Bayes Classifier.

    This implementation is optimized for multi-label classification with a large number
    of labels and features. It computes parameters for all classes simultaneously
    using sparse matrix operations, avoiding the overhead of training K separate models.
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter.
        """
        self.alpha = alpha
        self.coef_ = None  # Weights matrix of shape (n_classes, n_features)
        self.intercept_ = None  # Bias vector of shape (n_classes,)
        self.is_fitted = False

    def fit(self, X, Y):
        """
        Fits the model on the training data.

        Args:
            X (scipy.sparse.spmatrix): Document-term matrix of shape (n_samples, n_features).
            Y (scipy.sparse.spmatrix): Document-label matrix of shape (n_samples, n_classes).

        Returns:
            self
        """
        print(
            f"Fitting VectorizedMNB on {X.shape[0]} samples with {Y.shape[1]} classes..."
        )

        # Ensure inputs are in optimal sparse format
        # X as CSR is good for row slicing, but for Y.T @ X, efficient formats vary.
        # Scipy handles this well, but converting Y to CSC makes Y.T CSR, which is good.
        if not scipy.sparse.isspmatrix_csr(X):
            X = X.tocsr()
        if not scipy.sparse.isspmatrix_csc(Y):
            Y = Y.tocsc()

        n_samples, n_features = X.shape
        _, n_classes = Y.shape

        # ---------------------------------------------------------
        # 1. Compute Feature Counts
        # ---------------------------------------------------------
        print("Computing feature counts per class...")

        # Positive Counts: Sum of feature vectors for documents belonging to each class
        # Operation: (K, N) @ (N, V) -> (K, V)
        feature_counts_pos = Y.T @ X

        # Global Counts: Sum of feature vectors across all documents
        # Operation: Sum(X, axis=0) -> (1, V)
        global_feature_counts = np.array(X.sum(axis=0)).flatten()

        # Negative Counts: Global - Positive
        # We perform this calculation in dense format as the result is likely dense
        # and we need to add alpha smoothing anyway.
        # Note: (K, V) dense matrix for K=5000, V=50000 is ~1GB RAM (float32), which is safe.
        feature_counts_pos = feature_counts_pos.toarray().astype(np.float32)
        feature_counts_neg = global_feature_counts - feature_counts_pos

        # ---------------------------------------------------------
        # 2. Compute Class Priors (Document Counts)
        # ---------------------------------------------------------
        # Count of documents for each class
        class_doc_counts = np.array(Y.sum(axis=0)).flatten()

        # ---------------------------------------------------------
        # 3. Compute Log Probabilities (Weights)
        # ---------------------------------------------------------
        print("Computing log-likelihoods...")

        # --- Positive Class Statistics ---
        # Total tokens per class (sum over features)
        total_tokens_pos = feature_counts_pos.sum(axis=1)[:, np.newaxis]  # (K, 1)

        # Smoothed Log Probabilities: log( (N_wc + alpha) / (N_c + alpha*V) )
        numerator_pos = np.log(feature_counts_pos + self.alpha)
        denominator_pos = np.log(total_tokens_pos + (self.alpha * n_features))
        log_prob_pos = numerator_pos - denominator_pos

        # --- Negative Class Statistics ---
        total_tokens_neg = feature_counts_neg.sum(axis=1)[:, np.newaxis]  # (K, 1)

        numerator_neg = np.log(feature_counts_neg + self.alpha)
        denominator_neg = np.log(total_tokens_neg + (self.alpha * n_features))
        log_prob_neg = numerator_neg - denominator_neg

        # ---------------------------------------------------------
        # 4. Compute Model Parameters
        # ---------------------------------------------------------
        print("Computing final weights and bias...")

        # Weight = log(P(w|C=1)) - log(P(w|C=0))
        self.coef_ = log_prob_pos - log_prob_neg

        # Bias = log(P(C=1)) - log(P(C=0))
        # Using smoothed class priors
        prob_class_pos = (class_doc_counts + self.alpha) / (n_samples + 2 * self.alpha)
        prob_class_neg = 1.0 - prob_class_pos

        self.intercept_ = np.log(prob_class_pos) - np.log(prob_class_neg)

        self.is_fitted = True

        # Clean up large intermediate matrices
        del feature_counts_pos, feature_counts_neg, log_prob_pos, log_prob_neg
        import gc

        gc.collect()

        print("Model fitted successfully.")
        return self

    def predict_scores(self, X):
        """
        Computes the raw decision scores (log-odds ratios) for each class.
        Score > 0 implies Positive Class probability > Negative Class probability (if threshold is 0).

        Args:
            X (scipy.sparse.spmatrix): Input features.

        Returns:
            np.ndarray: Dense matrix of scores (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        print(f"Predicting scores for {X.shape[0]} samples...")

        # Score = X @ W.T + b
        # X: (N, V) sparse
        # W.T: (V, K) dense
        # Result: (N, K) dense
        scores = X @ self.coef_.T
        scores += self.intercept_

        return scores

    def predict(self, X, threshold=0.0):
        """
        Predicts binary labels based on a threshold.

        Args:
            X (scipy.sparse.spmatrix): Input features.
            threshold (float): Decision threshold for log-odds score.
                               0.0 corresponds to probability 0.5.

        Returns:
            scipy.sparse.csr_matrix: Binary prediction matrix (n_samples, n_classes).
        """
        scores = self.predict_scores(X)

        print(f"Applying threshold ({threshold}) to generate binary predictions...")
        # Convert boolean result to sparse matrix immediately to save memory
        # if the matrix is large (though dense boolean is usually manageable).
        predictions = scipy.sparse.csr_matrix(scores > threshold, dtype=bool)

        return predictions

    def save(self, filepath):
        """
        Saves the model parameters to a file.
        """
        model_data = {
            "coef": self.coef_,
            "intercept": self.intercept_,
            "alpha": self.alpha,
        }
        save_artifact(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Loads the model parameters from a file.
        """
        print(f"Loading model from {filepath}...")
        model_data = load_artifact(filepath)

        instance = cls(alpha=model_data["alpha"])
        instance.coef_ = model_data["coef"]
        instance.intercept_ = model_data["intercept"]
        instance.is_fitted = True

        return instance
