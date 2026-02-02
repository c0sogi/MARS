import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
import joblib
import gc

# Import from provided library files
from library.data_processing import get_processed_data
from library.feature_extraction import get_tfidf_features, TfidfEmbedder
from library.label_manager import get_target_matrix, TagEncoder
from library.utils import save_submission


class SparseNCC:
    """
    Nearest Centroid Classifier for sparse data.
    Represents each class as the centroid (mean) of its training samples.
    Predictions are based on cosine similarity.
    """

    def __init__(self):
        self.centroids_ = None

    def fit(self, X, y):
        """
        Computes class centroids.

        Args:
            X (sparse matrix): Feature matrix (n_samples, n_features).
            y (sparse matrix): Label matrix (n_samples, n_classes).
        """
        print("Fitting SparseNCC...")
        # Compute sum of vectors for each class
        # y.T is (n_classes, n_samples), X is (n_samples, n_features)
        # Result is (n_classes, n_features)
        # We convert y to float to ensure float arithmetic
        y_float = y.astype(np.float32)

        # Calculate raw sum of features for each tag
        print("Computing centroid sums...")
        centroid_sums = y_float.T @ X

        # Normalize to unit length (L2 norm) to facilitate cosine similarity
        # If a class has no samples, it remains 0
        print("Normalizing centroids...")
        self.centroids_ = normalize(centroid_sums, norm="l2", axis=1)

        return self

    def predict_proba(self, X):
        """
        Computes cosine similarity between X and centroids.

        Args:
            X (sparse matrix): Feature matrix (n_samples, n_features).

        Returns:
            sparse matrix or dense array: Similarity scores (n_samples, n_classes).
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted.")

        # X is (n_samples, n_features), centroids is (n_classes, n_features)
        # X @ centroids.T -> (n_samples, n_classes)
        # Assuming X is also L2 normalized (standard for TF-IDF)
        print("Computing similarity scores...")
        return X @ self.centroids_.T

    def save(self, path):
        """Saves the centroids matrix."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save as sparse matrix if it is sparse, else dense
        # Usually centroids become dense-ish, but let's use joblib
        joblib.dump(self.centroids_, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Loads the centroids matrix."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.centroids_ = joblib.load(path)
        print(f"Model loaded from {path}")
        return self


def evaluate_thresholds(y_true, y_scores, thresholds):
    """
    Evaluates F1 score at different thresholds.

    Args:
        y_true (sparse matrix): True labels.
        y_scores (dense/sparse matrix): Predicted scores.
        thresholds (list): List of thresholds to test.

    Returns:
        float: Best threshold.
        float: Best F1 score.
    """
    best_f1 = -1
    best_thresh = 0.5

    # Ensure y_scores is dense for easier comparison if it fits in memory
    if sparse.issparse(y_scores):
        y_scores = y_scores.toarray()

    if sparse.issparse(y_true):
        y_true = y_true.toarray()

    print("Evaluating thresholds...")
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        score = f1_score(y_true, y_pred, average="samples", zero_division=0)
        print(f"Threshold: {thresh:.2f}, F1-Score: {score:.10f}")

        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh, best_f1


def train_model(
    train_limit=None,
    val_limit=None,
    top_k_tags=5000,
    max_features=100000,
    cache_dir="./working/idea_2",
):
    """
    Pipeline to train the SparseNCC model and tune the threshold.
    """
    print("=== Starting Training Pipeline ===")

    # 1. Load and Process Data
    print("--- Loading Data ---")
    df_train = get_processed_data("train", limit=train_limit)
    df_val = get_processed_data("val", limit=val_limit)

    # 2. Feature Extraction (TF-IDF)
    print("--- Feature Extraction ---")
    embedder = TfidfEmbedder(max_features=max_features)

    # Fit on train, transform train
    X_train = get_tfidf_features(
        df_train, "train", embedder=embedder, cache_dir=cache_dir
    )
    # Transform val
    X_val = get_tfidf_features(df_val, "val", embedder=embedder, cache_dir=cache_dir)

    # Save embedder
    embedder.save(os.path.join(cache_dir, "tfidf_embedder.pkl"))

    # 3. Target Extraction
    print("--- Target Extraction ---")
    encoder = TagEncoder(top_k=top_k_tags)

    # Fit on train tags, transform train
    y_train = get_target_matrix(df_train, "train", encoder=encoder, cache_dir=cache_dir)
    # Transform val tags
    y_val = get_target_matrix(df_val, "val", encoder=encoder, cache_dir=cache_dir)

    # Save encoder
    encoder.save(os.path.join(cache_dir, "tag_encoder.pkl"))

    # Clean up DataFrames to save memory
    del df_train, df_val
    gc.collect()

    # 4. Train Model
    print("--- Training Model ---")
    model = SparseNCC()
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(cache_dir, "ncc_model.pkl")
    model.save(model_path)

    # Clean up train matrices
    del X_train, y_train
    gc.collect()

    # 5. Evaluate and Tune
    print("--- Evaluating ---")
    # Predict scores on validation
    val_scores = model.predict_proba(X_val)

    # Define thresholds to scan
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best_thresh, best_f1 = evaluate_thresholds(y_val, val_scores, thresholds)

    print(f"\nBest Threshold: {best_thresh}")
    print(f"Best Validation F1-Score: {best_f1:.10f}")

    # Save best threshold
    with open(os.path.join(cache_dir, "best_threshold.txt"), "w") as f:
        f.write(str(best_thresh))

    return model, best_thresh


def generate_submission_file(
    test_limit=None,
    cache_dir="./working/idea_2",
    submission_path="./submission/submission.csv",
):
    """
    Pipeline to generate submission for the test set.
    """
    print("\n=== Starting Submission Pipeline ===")

    # 1. Load Artifacts
    print("Loading artifacts...")
    try:
        embedder = TfidfEmbedder().load(os.path.join(cache_dir, "tfidf_embedder.pkl"))
        encoder = TagEncoder.load(os.path.join(cache_dir, "tag_encoder.pkl"))
        model = SparseNCC().load(os.path.join(cache_dir, "ncc_model.pkl"))

        with open(os.path.join(cache_dir, "best_threshold.txt"), "r") as f:
            threshold = float(f.read().strip())
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. Please run training first.")
        return

    print(f"Using Threshold: {threshold}")

    # 2. Load and Process Test Data
    print("Loading test data...")
    df_test = get_processed_data("test", limit=test_limit)
    ids = df_test["Id"].values

    # 3. Feature Extraction
    print("Extracting features for test set...")
    # Note: We don't cache test features in the same way or we need to be careful with naming if we do.
    # The library function get_tfidf_features handles caching based on split name.
    X_test = get_tfidf_features(df_test, "test", embedder=embedder, cache_dir=cache_dir)

    # 4. Prediction
    print("Predicting...")
    scores = model.predict_proba(X_test)

    # Apply threshold
    # If scores is sparse, this returns a sparse boolean matrix? No, usually dense boolean.
    # Let's handle it carefully.
    if sparse.issparse(scores):
        # Convert to dense chunk by chunk if needed, but 600k * 5k is ~3GB. Safe to densify.
        scores = scores.toarray()

    binary_preds = scores > threshold

    # 5. Decode Tags
    print("Decoding tags...")
    # inverse_transform takes binary matrix
    pred_tag_strings = encoder.inverse_transform(binary_preds)

    # 6. Save Submission
    print(f"Saving submission to {submission_path}...")
    save_submission(ids, pred_tag_strings, submission_path)
    print("Submission saved successfully.")
