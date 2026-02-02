import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from library.config import WORK_DIR, FAISS_INDEX_PATH, K_NEIGHBORS, METRIC, SEED


class KNNClassifier:
    """
    Retrieval-based classifier using k-Nearest Neighbors.

    This class wraps sklearn's KNeighborsClassifier to implement a retrieval system
    that predicts semiotic classes based on the distance-weighted majority vote
    of the k-nearest neighbors in the embedding space.
    """

    def __init__(self, k=K_NEIGHBORS, metric=METRIC, n_jobs=-1):
        """
        Initialize the KNN Classifier.

        Args:
            k (int): Number of neighbors to retrieve.
            metric (str): Distance metric to use (e.g., 'l2', 'euclidean').
            n_jobs (int): Number of parallel jobs for neighbor search.
        """
        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs

        # 'weights="distance"' implements the distance-weighted majority voting requirement.
        # 'algorithm="auto"' allows sklearn to choose the best indexing structure (KDTree, BallTree, or Brute).
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric=metric,
            n_jobs=n_jobs,
            algorithm="auto",
        )
        self.is_fitted = False

    def build_index(self, train_vectors, train_labels):
        """
        Builds the retrieval index (fits the k-NN model) using training data.

        Args:
            train_vectors (np.ndarray): Dense embedding vectors for training tokens.
            train_labels (np.ndarray): Target semiotic class labels.
        """
        print(f"Building k-NN index with {len(train_vectors)} samples...")
        self.model.fit(train_vectors, train_labels)
        self.is_fitted = True
        print("Index built successfully.")

    def predict(self, test_vectors):
        """
        Retrieves neighbors and predicts the class for test vectors.

        Args:
            test_vectors (np.ndarray): Dense embedding vectors for test tokens.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Index not built. Call build_index first.")

        print(f"Querying index for {len(test_vectors)} samples...")
        # predict() in KNeighborsClassifier with weights='distance' performs the weighted vote.
        return self.model.predict(test_vectors)

    def evaluate(self, val_vectors, val_labels):
        """
        Evaluates the model on validation data and prints accuracy with full precision.

        Args:
            val_vectors (np.ndarray): Validation features.
            val_labels (np.ndarray): Validation ground truth labels.

        Returns:
            float: Accuracy score.
        """
        print(f"Evaluating on {len(val_vectors)} validation samples...")
        preds = self.predict(val_vectors)
        acc = accuracy_score(val_labels, preds)
        # Print full precision as requested
        print(f"Validation Accuracy: {acc}")
        return acc

    def save(self, path):
        """
        Saves the fitted model to disk.

        Args:
            path (str): File path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Loads a fitted model from disk.

        Args:
            path (str): File path to load the model from.
        """
        if os.path.exists(path):
            print(f"Loading model from {path}...")
            self.model = joblib.load(path)
            self.is_fitted = True

            # Update internal params to match loaded model
            self.k = self.model.n_neighbors
            self.metric = self.model.metric
        else:
            raise FileNotFoundError(f"Model file {path} not found.")


def load_or_train_index(
    train_vectors, train_labels, load_cached_model=True, save_path=FAISS_INDEX_PATH
):
    """
    Orchestrates the loading or training of the KNNClassifier.

    1. If load_cached_model is True and file exists, loads the model.
    2. Otherwise, initializes a new model, trains (builds index) on provided data, and saves it.

    Args:
        train_vectors (np.ndarray): Training features.
        train_labels (np.ndarray): Training labels.
        load_cached_model (bool): Flag to enable loading from cache.
        save_path (str): Path for persistence.

    Returns:
        KNNClassifier: The ready-to-use classifier.
    """
    classifier = KNNClassifier(k=K_NEIGHBORS, metric=METRIC)

    if load_cached_model and os.path.exists(save_path):
        try:
            classifier.load(save_path)
            print("Loaded k-NN index from cache.")
            return classifier
        except Exception as e:
            print(f"Failed to load cache ({e}). Rebuilding index from scratch...")

    # Train/Build
    classifier.build_index(train_vectors, train_labels)

    # Save
    classifier.save(save_path)

    return classifier
