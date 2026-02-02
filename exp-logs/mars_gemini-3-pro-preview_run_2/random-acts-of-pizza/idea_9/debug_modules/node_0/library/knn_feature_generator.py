import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class KNNFeatureAugmenter:
    """
    Implements Instance-Based Feature Engineering using k-Nearest Neighbors.
    Generates 'Local Success Probability' features based on the labels of neighbors.
    """

    def __init__(self, k=50, metric="cosine", n_jobs=-1):
        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs
        self.knn = None

    def generate_oof_features(self, X, y):
        """
        Generates Out-Of-Fold (OOF) probability features for the training set.
        Uses Stratified K-Fold Cross-Validation to prevent data leakage (i.e., preventing
        a sample from being its own neighbor).

        Args:
            X (np.ndarray): Training features (embeddings).
            y (np.ndarray): Training labels.

        Returns:
            np.ndarray: Column vector of shape (N, 1) containing the probability of success.
        """
        # Initialize a fresh KNN for CV
        knn = KNeighborsClassifier(
            n_neighbors=self.k, metric=self.metric, n_jobs=self.n_jobs
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Generate probabilities for class 1
        # cross_val_predict with method='predict_proba' returns shape (n_samples, n_classes)
        y_probs = cross_val_predict(
            knn, X, y, cv=skf, method="predict_proba", n_jobs=self.n_jobs
        )

        # Return probability of the positive class (index 1)
        return y_probs[:, 1].reshape(-1, 1)

    def fit(self, X, y):
        """
        Fits the k-NN model on the full training set.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.knn = KNeighborsClassifier(
            n_neighbors=self.k, metric=self.metric, n_jobs=self.n_jobs
        )
        self.knn.fit(X, y)
        return self

    def transform(self, X):
        """
        Generates probability features for a query set using the fitted model.

        Args:
            X (np.ndarray): Query features (Validation or Test).

        Returns:
            np.ndarray: Column vector of shape (N, 1) containing the probability of success.
        """
        if self.knn is None:
            raise RuntimeError("Model must be fitted before calling transform.")

        y_probs = self.knn.predict_proba(X)
        return y_probs[:, 1].reshape(-1, 1)


def run_knn_feature_generation(
    X_text_train, y_train, X_text_val, X_text_test, k=50, load_cached_data=True
):
    """
    Orchestrates the generation of k-NN features with caching.

    Args:
        X_text_train (np.ndarray): Training text embeddings.
        y_train (np.ndarray): Training labels.
        X_text_val (np.ndarray): Validation text embeddings.
        X_text_test (np.ndarray): Test text embeddings.
        k (int): Number of neighbors.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_knn_train, X_knn_val, X_knn_test)
    """
    cache_dir = "./working/idea_9/"
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_knn_train": os.path.join(cache_dir, "X_knn_train.npy"),
        "X_knn_val": os.path.join(cache_dir, "X_knn_val.npy"),
        "X_knn_test": os.path.join(cache_dir, "X_knn_test.npy"),
    }

    # Check if files exist
    all_exist = all(os.path.exists(f) for f in files.values())

    if load_cached_data and all_exist:
        print("Loading k-NN features from cache...")
        try:
            X_knn_train = np.load(files["X_knn_train"])
            X_knn_val = np.load(files["X_knn_val"])
            X_knn_test = np.load(files["X_knn_test"])
            return X_knn_train, X_knn_val, X_knn_test
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
    else:
        print("Computing k-NN features from scratch...")

    augmenter = KNNFeatureAugmenter(k=k)

    # 1. Generate OOF features for Training Data (Leakage Prevention)
    print("Generating OOF features for training set...")
    X_knn_train = augmenter.generate_oof_features(X_text_train, y_train)

    # 2. Fit on full Training Data
    print("Fitting k-NN on full training set...")
    augmenter.fit(X_text_train, y_train)

    # 3. Generate features for Validation and Test Data
    print("Generating features for validation and test sets...")
    X_knn_val = augmenter.transform(X_text_val)
    X_knn_test = augmenter.transform(X_text_test)

    # Save to cache
    print("Saving k-NN features to cache...")
    np.save(files["X_knn_train"], X_knn_train)
    np.save(files["X_knn_val"], X_knn_val)
    np.save(files["X_knn_test"], X_knn_test)

    return X_knn_train, X_knn_val, X_knn_test
