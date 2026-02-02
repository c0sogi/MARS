import sys
import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score

# Import provided library modules
from library.data_processing import get_processed_data
from library.feature_extraction import get_tfidf_features, TfidfEmbedder
from library.label_manager import get_target_matrix, TagEncoder
from library.centroid_classifier import SparseNCC
from library.evaluation import optimize_threshold, compute_f1_score
from library.utils import save_submission

# Set random seed for reproducibility
np.random.seed(42)


def calculate_per_sample_f1(y_true, y_pred):
    """
    Calculates F1 score for each sample efficiently.
    y_true and y_pred can be sparse matrices or numpy arrays.
    """
    # Ensure dense boolean/int arrays for calculation if memory permits,
    # or use sparse operations. Given the size, sparse operations are safer.

    if not sparse.issparse(y_true):
        y_true = sparse.csr_matrix(y_true)
    if not sparse.issparse(y_pred):
        y_pred = sparse.csr_matrix(y_pred)

    # Intersection: element-wise multiplication
    # For binary vectors, this is the number of True Positives
    intersection = y_true.multiply(y_pred).sum(axis=1).A1

    # Sum of true labels
    true_sum = y_true.sum(axis=1).A1

    # Sum of predicted labels
    pred_sum = y_pred.sum(axis=1).A1

    # F1 = 2 * intersection / (true_sum + pred_sum)
    # Handle division by zero
    denominator = true_sum + pred_sum

    # Initialize f1 array
    f1_scores = np.zeros_like(denominator, dtype=np.float64)

    mask = denominator > 0
    f1_scores[mask] = 2 * intersection[mask] / denominator[mask]

    return f1_scores


def main():
    # --- Configuration ---
    TRAIN_LIMIT = None  # Limit training data for fast baseline execution
    VAL_LIMIT = None  # Limit validation data for speed
    TEST_LIMIT = None  # Predict on full test set
    TOP_K_TAGS = 5000
    MAX_FEATURES = 100000
    CACHE_DIR = "./working/idea_3"
    SUBMISSION_PATH = "./submission/submission.csv"

    print("=== Starting Runfile Execution ===")

    # ---------------------------------------------------------
    # 1. Data Loading & Processing
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Data...")
    df_train = get_processed_data("train", limit=TRAIN_LIMIT, load_cached_data=True)
    df_val = get_processed_data("val", limit=VAL_LIMIT, load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Feature Extraction
    # ---------------------------------------------------------
    print("\n[Step 2] Extracting Features...")
    embedder = TfidfEmbedder(max_features=MAX_FEATURES)

    # Fit on train, transform train
    # Note: get_tfidf_features handles caching. If cache exists, embedder might not be fitted.
    # We force fit by passing the embedder instance.
    X_train = get_tfidf_features(
        df_train, "train", embedder=embedder, cache_dir=CACHE_DIR
    )

    # Transform validation
    X_val = get_tfidf_features(df_val, "val", embedder=embedder, cache_dir=CACHE_DIR)

    # ---------------------------------------------------------
    # 3. Target Encoding
    # ---------------------------------------------------------
    print("\n[Step 3] Encoding Targets...")
    encoder = TagEncoder(top_k=TOP_K_TAGS)

    # Fit on train tags, transform train
    y_train = get_target_matrix(df_train, "train", encoder=encoder, cache_dir=CACHE_DIR)

    # Transform validation tags
    y_val = get_target_matrix(df_val, "val", encoder=encoder, cache_dir=CACHE_DIR)

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 4] Training SparseNCC Model...")
    model = SparseNCC()
    model.fit(X_train, y_train)

    # Clean up training data to free memory
    del df_train, X_train, y_train
    gc.collect()

    # ---------------------------------------------------------
    # 5. Validation & Threshold Optimization
    # ---------------------------------------------------------
    print("\n[Step 5] Validating...")
    # Predict probabilities (scores)
    val_scores = model.predict_proba(X_val)

    # Optimize threshold
    best_threshold, best_f1 = optimize_threshold(y_val, val_scores)

    print(f"Final Validation Metric: {best_f1}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 6] Performing Failure Analysis...")

    # Convert scores to binary predictions using best threshold
    if sparse.issparse(val_scores):
        val_scores_dense = val_scores.toarray()
    else:
        val_scores_dense = val_scores

    val_preds = (val_scores_dense > best_threshold).astype(int)

    # Calculate per-sample F1 score
    sample_f1s = calculate_per_sample_f1(y_val, val_preds)

    # Error magnitude
    error_magnitude = 1.0 - sample_f1s

    # Input feature: Text Length
    # df_val has 'text' column
    text_lengths = df_val["text"].fillna("").astype(str).apply(len).values

    # Calculate correlation
    correlation = np.corrcoef(error_magnitude, text_lengths)[0, 1]

    print(
        f"Correlation between Error Magnitude and Input Text Length: {correlation:.6f}"
    )

    # Clean up validation data
    del (
        df_val,
        X_val,
        y_val,
        val_scores,
        val_preds,
        sample_f1s,
        error_magnitude,
        text_lengths,
    )
    gc.collect()

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 7] Generating Submission...")

    # Load Test Data
    df_test = get_processed_data("test", limit=TEST_LIMIT, load_cached_data=True)
    ids = df_test["Id"].values

    # Extract Features for Test
    # Use the same embedder fitted on training data
    # Note: get_tfidf_features will try to load cache for 'test' or compute using embedder
    X_test = get_tfidf_features(df_test, "test", embedder=embedder, cache_dir=CACHE_DIR)

    # Predict
    print("Predicting on test set...")
    test_scores = model.predict_proba(X_test)

    # Apply threshold
    if sparse.issparse(test_scores):
        # Process in chunks if memory is tight, but 220GB is plenty for 600k rows
        test_scores = test_scores.toarray()

    binary_preds = test_scores > best_threshold

    # Decode Tags
    print("Decoding tags...")
    pred_tag_strings = encoder.inverse_transform(binary_preds)

    # Save Submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    save_submission(ids, pred_tag_strings, SUBMISSION_PATH)

    print("Runfile execution completed successfully.")


if __name__ == "__main__":
    main()
