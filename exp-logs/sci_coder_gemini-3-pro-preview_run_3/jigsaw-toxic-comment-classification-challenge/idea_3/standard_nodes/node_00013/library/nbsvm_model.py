import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from library.config import Config
from library.utils import calculate_roc_auc


class NBSVM(BaseEstimator, ClassifierMixin):
    """
    NBSVM (Naive Bayes - Support Vector Machine) variant using Logistic Regression.
    Scales features by the Naive Bayes log-count ratio before fitting a linear model.
    """

    def __init__(self, C=1.0, dual=False, n_jobs=1, random_state=None):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.classifiers = []
        self.ratios = []

    def _compute_ratio(self, X, y):
        """
        Computes the Naive Bayes log-count ratio r = log(p/q).
        """
        # Smoothing factor
        alpha = 1.0

        # Select rows for positive and negative classes
        # X is sparse, so we sum along axis 0 (columns) to get feature counts
        # p: Probability of feature given class 1
        p = X[y == 1].sum(axis=0) + alpha

        # q: Probability of feature given class 0
        q = X[y == 0].sum(axis=0) + alpha

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Log ratio
        r = np.log(p / q)
        return r

    def fit(self, X, y):
        """
        Fits one Logistic Regression classifier per label.
        """
        # Reset internal state
        self.classifiers = []
        self.ratios = []

        # Ensure y is a numpy array
        y = np.array(y)

        # Handle multi-label input (N, L)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_labels = y.shape[1]

        for i in range(n_labels):
            y_i = y[:, i]

            # Compute NB ratio for this label
            r = self._compute_ratio(X, y_i)
            # Flatten to 1D array (n_features,)
            r = np.asarray(r).flatten()
            self.ratios.append(r)

            # Scale features: element-wise multiplication
            # X is (N, V), r is (V,). X.multiply(r) broadcasts r across rows.
            X_nb = X.multiply(r)

            # Train Logistic Regression
            # Use 'liblinear' if dual=True (good for wide sparse data), else 'lbfgs'
            solver = "liblinear" if self.dual else "lbfgs"

            clf = LogisticRegression(
                C=self.C,
                dual=self.dual,
                n_jobs=self.n_jobs,
                solver=solver,
                random_state=self.random_state,
                max_iter=2000,  # Increased to ensure convergence
            )
            clf.fit(X_nb, y_i)
            self.classifiers.append(clf)

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities for all labels.
        Returns: np.ndarray of shape (n_samples, n_labels)
        """
        preds = []
        for i, clf in enumerate(self.classifiers):
            r = self.ratios[i]
            # Scale features using the ratio learned during training
            X_nb = X.multiply(r)
            # Predict probability of the positive class (index 1)
            prob = clf.predict_proba(X_nb)[:, 1]
            preds.append(prob)

        # Stack predictions column-wise
        return np.column_stack(preds)


def train_and_predict_nbsvm(X_train, y_train, X_val, y_val, X_test):
    """
    Orchestrates the training and prediction process for the NBSVM model.

    Args:
        X_train, X_val, X_test: Sparse feature matrices (TF-IDF).
        y_train, y_val: Target labels (numpy arrays or DataFrames).

    Returns:
        tuple: (val_preds, test_preds) - Predicted probabilities as numpy arrays.
    """
    print(f"Initializing NBSVM (C={Config.NBSVM_C})...")

    # Instantiate model
    # Using n_jobs=-1 to utilize available CPU cores for the solver
    model = NBSVM(
        C=Config.NBSVM_C,
        dual=False,  # lbfgs is robust and supports parallel execution
        n_jobs=-1,
        random_state=Config.SEED,
    )

    print("Fitting NBSVM model...")
    model.fit(X_train, y_train)

    print("Predicting on Validation set...")
    val_preds = model.predict_proba(X_val)

    # Calculate and print metric
    score = calculate_roc_auc(y_val, val_preds)
    print(f"NBSVM Validation Mean Column-wise ROC AUC: {score}")

    print("Predicting on Test set...")
    test_preds = model.predict_proba(X_test)

    return val_preds, test_preds
