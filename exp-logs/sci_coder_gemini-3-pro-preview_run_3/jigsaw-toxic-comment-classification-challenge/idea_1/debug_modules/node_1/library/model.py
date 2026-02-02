import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.utils import get_logger

logger = get_logger("model")


class NBSVM(BaseEstimator, ClassifierMixin):
    """
    NBSVM (Naive Bayes - Support Vector Machine) / NB-Logistic Regression.

    This model computes the log-count ratios (Naive Bayes features) for each feature
    based on the binary target, scales the feature matrix by these ratios, and then
    trains a linear classifier (Logistic Regression) on the scaled features.
    """

    def __init__(
        self,
        C=1.0,
        dual=False,
        n_jobs=1,
        max_iter=100,
        solver="liblinear",
        random_state=42,
    ):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.r = None
        self.model = None

    def fit(self, X, y):
        """
        Fit the NBSVM model.

        Args:
            X: Sparse matrix of shape (n_samples, n_features).
            y: Array-like of shape (n_samples,).
        """
        # Ensure X is sparse CSR for efficient slicing and multiplication
        if not sparse.isspmatrix(X):
            X = sparse.csr_matrix(X)

        y = np.array(y)

        # 1. Compute Log-Count Ratios (Naive Bayes Step)
        # p: Sum of features for the positive class (with smoothing +1)
        # q: Sum of features for the negative class (with smoothing +1)
        # We cast to np.array to ensure we don't get np.matrix objects from scipy sum
        p = np.array(X[y == 1].sum(axis=0)) + 1.0
        q = np.array(X[y == 0].sum(axis=0)) + 1.0

        # Normalize counts to probabilities
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate the log-ratio vector r
        self.r = np.log(p / q)

        # 2. Scale Features
        # Element-wise multiplication of the feature matrix X by the ratio vector r.
        # scipy.sparse.csr_matrix.multiply broadcasts the row vector r across all rows of X.
        X_nb = X.multiply(self.r)

        # 3. Train Linear Classifier
        self.model = LogisticRegression(
            C=self.C,
            dual=self.dual,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.model.fit(X_nb, y)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if self.model is None or self.r is None:
            raise RuntimeError("Model is not fitted.")

        if not sparse.isspmatrix(X):
            X = sparse.csr_matrix(X)

        # Scale features using the learned log-ratios
        X_nb = X.multiply(self.r)

        # Return probability estimates
        return self.model.predict_proba(X_nb)

    def predict(self, X):
        """
        Predict class labels.
        """
        if self.model is None or self.r is None:
            raise RuntimeError("Model is not fitted.")

        if not sparse.isspmatrix(X):
            X = sparse.csr_matrix(X)

        X_nb = X.multiply(self.r)
        return self.model.predict(X_nb)


class MultiLabelNBSVM(BaseEstimator):
    """
    Wrapper for training multiple NBSVM models for multi-label classification.
    """

    def __init__(
        self,
        C=1.0,
        dual=False,
        n_jobs=1,
        max_iter=100,
        solver="liblinear",
        random_state=42,
    ):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.models = {}
        self.labels = None

    def fit(self, X, Y):
        """
        Fit one NBSVM model per column in Y.

        Args:
            X: Feature matrix.
            Y: pandas DataFrame or numpy array of binary labels (n_samples, n_labels).
        """
        if hasattr(Y, "columns"):
            self.labels = Y.columns.tolist()
            Y_arr = Y.values
        else:
            self.labels = [f"label_{i}" for i in range(Y.shape[1])]
            Y_arr = Y

        logger.info(f"Training MultiLabelNBSVM on {len(self.labels)} labels...")

        for i, label in enumerate(self.labels):
            y_col = Y_arr[:, i]

            # Instantiate a new binary NBSVM for this label
            model = NBSVM(
                C=self.C,
                dual=self.dual,
                n_jobs=self.n_jobs,
                max_iter=self.max_iter,
                solver=self.solver,
                random_state=self.random_state,
            )

            model.fit(X, y_col)
            self.models[label] = model

        return self

    def predict_proba(self, X):
        """
        Predict probabilities for all labels.

        Returns:
            np.ndarray: Matrix of shape (n_samples, n_labels) with probabilities for class 1.
        """
        if not self.models:
            raise RuntimeError("Models not fitted.")

        n_samples = X.shape[0]
        n_labels = len(self.labels)
        preds = np.zeros((n_samples, n_labels))

        for i, label in enumerate(self.labels):
            # predict_proba returns [prob_0, prob_1], we take prob_1
            preds[:, i] = self.models[label].predict_proba(X)[:, 1]

        return preds

    def score(self, X, Y):
        """
        Calculates and prints ROC AUC for each column and the mean.

        Args:
            X: Feature matrix.
            Y: True labels (DataFrame or array).

        Returns:
            float: Mean column-wise ROC AUC.
        """
        preds = self.predict_proba(X)

        if hasattr(Y, "columns"):
            Y_arr = Y.values
            labels = Y.columns.tolist()
        else:
            Y_arr = Y
            labels = self.labels

        aucs = []
        logger.info("Validation Metrics:")

        for i, label in enumerate(labels):
            y_true = Y_arr[:, i]
            y_pred = preds[:, i]

            # Calculate AUC
            try:
                score = roc_auc_score(y_true, y_pred)
            except ValueError:
                score = 0.5  # Handle case with single class in validation

            aucs.append(score)
            # Print full precision without formatting
            print(f"Column {label} ROC AUC: {score}")

        mean_auc = np.mean(aucs)
        print(f"Mean Column-wise ROC AUC: {mean_auc}")

        return mean_auc
