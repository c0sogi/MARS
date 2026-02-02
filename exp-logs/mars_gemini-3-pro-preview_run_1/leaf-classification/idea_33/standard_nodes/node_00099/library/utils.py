import os
import random
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.special
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.covariance import OAS
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(ids, classes, probs, output_path):
    """
    Saves the prediction probabilities to a CSV file in the required format.

    Args:
        ids: Array of image ids.
        classes: List of class names (columns).
        probs: Matrix of probabilities (n_samples, n_classes).
        output_path: Path to save the CSV.
    """
    # Create DataFrame
    df = pd.DataFrame(probs, columns=classes)
    df.insert(0, "id", ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


class CholeskyOASClassifier:
    """
    Custom Linear Discriminant Classifier using OAS Covariance and Cholesky Solver.
    Implements the 'Alphanumeric Cholesky-Solved Exact-Precision OAS Discriminant'.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.covariance_ = None
        self.weights_ = None
        self.bias_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and Cholesky decomposition.
        Operations performed in float64.
        """
        # Ensure float64
        X = X.astype(np.float64)

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Parameter Estimation
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        for k in range(n_classes):
            X_k = X[y_enc == k]
            self.means_[k] = np.mean(X_k, axis=0)
            self.priors_[k] = len(X_k) / len(X)

        # Compute residuals (centered data)
        # We assume a shared covariance matrix across classes (LDA assumption)
        # Residuals = X - mean_of_class_y
        residuals = X - self.means_[y_enc]

        # Estimate Covariance using OAS
        # assume_centered=True because we manually centered using class means
        oas = OAS(assume_centered=True)
        oas.fit(residuals)
        self.covariance_ = oas.covariance_.astype(np.float64)

        # 2. Exact Weight Derivation via Cholesky
        # Solve Sigma * W.T = Means.T  => W = (Sigma^-1 * Means.T).T
        # System: Sigma * x = mu  -> solve for x (which is a column of W.T)

        # Factorize Sigma = L * L.T
        # scipy.linalg.cho_factor returns (c, lower)
        L = scipy.linalg.cho_factor(self.covariance_, lower=True)

        # Solve for Weights: W_k = Sigma^-1 * mu_k
        # We solve Sigma * W_T = Means_T
        self.weights_ = scipy.linalg.cho_solve(L, self.means_.T).T

        # 3. Compute Bias
        # b_k = -0.5 * (mu_k.T * Sigma^-1 * mu_k) + log(pi_k)
        # Note: W_k = Sigma^-1 * mu_k
        # So b_k = -0.5 * (mu_k . W_k) + log(pi_k)

        # Dot product per class
        term1 = -0.5 * np.sum(self.means_ * self.weights_, axis=1)
        term2 = np.log(self.priors_)
        self.bias_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities using linearized inference.
        """
        X = X.astype(np.float64)

        # Linear Scoring: Z = X * W.T + b
        logits = X @ self.weights_.T + self.bias_

        # Softmax
        probs = scipy.special.softmax(logits, axis=1)

        # Clip to avoid log extremes as per metric spec
        # range [1e-15, 1 - 1e-15]
        probs = np.clip(probs, 1e-15, 1 - 1e-15)

        return probs


def load_data(cache_dir="./working/idea_33/", load_cached_data=True):
    """
    Loads, preprocesses, and caches the dataset.
    Implements Alphanumeric Feature Ordering and Inductive Preprocessing.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.parquet"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.parquet"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.parquet"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = pd.read_parquet(cache_files["X_train"]).values
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = pd.read_parquet(cache_files["X_val"]).values
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = pd.read_parquet(cache_files["X_test"]).values
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Identify Feature Columns
    # Filter columns starting with margin, shape, texture
    feature_cols = [
        c for c in train_df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Alphanumeric Sorting (Lexicographical)
    # e.g. margin_1, margin_10, margin_11 ...
    feature_cols = sorted(feature_cols)

    # Extract Data (High Precision)
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["species"].values

    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = val_df["species"].values

    X_test = test_df[feature_cols].values.astype(np.float64)
    test_ids = test_df["id"].values

    # Inductive Preprocessing Pipeline
    # 1. Yeo-Johnson Power Transformation (standardize=False)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)

    # 2. Standard Scaling
    sc = StandardScaler()

    # Fit only on Train
    X_train = pt.fit_transform(X_train)
    X_train = sc.fit_transform(X_train)

    # Transform Val and Test
    X_val = sc.transform(pt.transform(X_val))
    X_test = sc.transform(pt.transform(X_test))

    # Get Classes
    classes = np.unique(y_train)

    # Save to Cache
    pd.DataFrame(X_train, columns=feature_cols).to_parquet(cache_files["X_train"])
    np.save(cache_files["y_train"], y_train)
    pd.DataFrame(X_val, columns=feature_cols).to_parquet(cache_files["X_val"])
    np.save(cache_files["y_val"], y_val)
    pd.DataFrame(X_test, columns=feature_cols).to_parquet(cache_files["X_test"])
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def run_training(X_train, y_train, X_val, y_val):
    """
    Trains the CholeskyOASClassifier and evaluates on validation set.
    """
    print(
        "Initializing Alphanumeric Cholesky-Solved Exact-Precision OAS Discriminant..."
    )
    model = CholeskyOASClassifier()

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # Ensure y_val is encoded or passed correctly. log_loss handles string labels if classes provided.
    loss = log_loss(y_val, val_probs, labels=model.classes_)

    print(f"Validation Multi-class Log Loss: {loss}")

    return model
