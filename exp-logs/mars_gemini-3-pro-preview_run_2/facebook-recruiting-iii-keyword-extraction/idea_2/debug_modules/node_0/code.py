import os
import sys
import numpy as np
import pandas as pd
import shutil
from scipy import sparse

# Import library modules
import library.data_processing as dp
from library.data_processing import get_processed_data
from library.feature_extraction import TfidfEmbedder, get_tfidf_features
from library.label_manager import TagEncoder, get_target_matrix
from library.centroid_classifier import SparseNCC
from library.evaluation import optimize_threshold, compute_f1_score
from library.utils import save_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # Set random seed
    np.random.seed(42)

    # Define demo directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch the cache directory in data_processing to isolate this run
    dp.CACHE_DIR = DEMO_DIR
    print(f"Working directory set to: {DEMO_DIR}")

    # Parameters for the demo (optimized for speed)
    TRAIN_LIMIT = 2000
    VAL_LIMIT = 500
    TEST_LIMIT = 100
    MAX_FEATURES = 1000
    TOP_K_TAGS = 50

    # 2. Data Processing
    # ---------------------------------------------------------
    print("\n--- Step 1: Data Processing ---")

    # Load and process training data
    print(f"Loading {TRAIN_LIMIT} rows of training data...")
    df_train = get_processed_data("train", limit=TRAIN_LIMIT, load_cached_data=False)

    # Validate DataFrame structure
    assert isinstance(df_train, pd.DataFrame)
    assert "text" in df_train.columns
    assert "Tags" in df_train.columns
    assert len(df_train) == TRAIN_LIMIT
    print("Training data loaded and verified.")

    # Load and process validation data
    print(f"Loading {VAL_LIMIT} rows of validation data...")
    df_val = get_processed_data("val", limit=VAL_LIMIT, load_cached_data=False)
    assert len(df_val) == VAL_LIMIT
    print("Validation data loaded and verified.")

    # 3. Feature Extraction (TF-IDF)
    # ---------------------------------------------------------
    print("\n--- Step 2: Feature Extraction ---")

    # Instantiate Embedder
    embedder = TfidfEmbedder(max_features=MAX_FEATURES, verbose=True)

    # Compute features for training
    # Note: We pass load_cached_data=False to ensure we run the logic
    X_train = get_tfidf_features(
        df_train, "train", embedder=embedder, load_cached_data=False, cache_dir=DEMO_DIR
    )

    # Validate Feature Matrix
    assert sparse.issparse(X_train)
    assert X_train.shape == (TRAIN_LIMIT, MAX_FEATURES)
    print(f"Training features shape: {X_train.shape}")

    # Compute features for validation (transform only)
    X_val = get_tfidf_features(
        df_val, "val", embedder=embedder, load_cached_data=False, cache_dir=DEMO_DIR
    )
    assert X_val.shape == (VAL_LIMIT, MAX_FEATURES)
    print(f"Validation features shape: {X_val.shape}")

    # Save embedder for later use
    embedder.save(os.path.join(DEMO_DIR, "tfidf_embedder.pkl"))

    # 4. Target Extraction (Label Encoding)
    # ---------------------------------------------------------
    print("\n--- Step 3: Target Extraction ---")

    # Instantiate Encoder
    encoder = TagEncoder(top_k=TOP_K_TAGS)

    # Compute targets for training
    y_train = get_target_matrix(
        df_train, "train", encoder=encoder, load_cached_data=False, cache_dir=DEMO_DIR
    )

    # Validate Target Matrix
    assert sparse.issparse(y_train)
    assert y_train.shape == (TRAIN_LIMIT, TOP_K_TAGS)
    print(f"Training targets shape: {y_train.shape}")

    # Compute targets for validation
    y_val = get_target_matrix(
        df_val, "val", encoder=encoder, load_cached_data=False, cache_dir=DEMO_DIR
    )
    assert y_val.shape == (VAL_LIMIT, TOP_K_TAGS)

    # Verify Inverse Transform Logic
    print("Verifying label decoding...")
    sample_indices = [0, 1, 2]
    decoded_tags = encoder.inverse_transform(y_train[sample_indices])
    assert len(decoded_tags) == 3
    assert isinstance(decoded_tags[0], str)
    print(f"Sample decoded tags: {decoded_tags}")

    # Save encoder
    encoder.save(os.path.join(DEMO_DIR, "tag_encoder.pkl"))

    # 5. Model Training (SparseNCC)
    # ---------------------------------------------------------
    print("\n--- Step 4: Model Training ---")

    model = SparseNCC()
    model.fit(X_train, y_train)

    # Save model
    model.save(os.path.join(DEMO_DIR, "ncc_model.pkl"))

    # Predict on validation
    print("Predicting on validation set...")
    val_probs = model.predict_proba(X_val)

    # Validate predictions
    assert val_probs.shape == (VAL_LIMIT, TOP_K_TAGS)
    print("Prediction shape verified.")

    # 6. Evaluation
    # ---------------------------------------------------------
    print("\n--- Step 5: Evaluation ---")

    # Optimize threshold
    best_thresh, best_f1 = optimize_threshold(y_val, val_probs)

    print(
        f"Optimization complete. Best Threshold: {best_thresh}, Best F1: {best_f1:.4f}"
    )
    assert 0 <= best_f1 <= 1.0, "F1 Score out of range"

    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\n--- Step 6: Submission Generation ---")

    # Load Test Data
    df_test = get_processed_data("test", limit=TEST_LIMIT, load_cached_data=False)

    # Extract Features (using loaded embedder logic simulation)
    # In a real scenario, we would load the embedder, but we have it in memory.
    X_test = get_tfidf_features(
        df_test, "test", embedder=embedder, load_cached_data=False, cache_dir=DEMO_DIR
    )

    # Predict
    test_probs = model.predict_proba(X_test)

    # Apply Threshold
    # Convert to dense for thresholding if needed, though sparse works too usually.
    # Here we ensure it matches the logic in centroid_classifier.py
    if sparse.issparse(test_probs):
        test_probs = test_probs.toarray()

    binary_preds = (test_probs > best_thresh).astype(int)

    # Decode
    pred_tag_strings = encoder.inverse_transform(binary_preds)

    # Save Submission
    submission_path = os.path.join(DEMO_DIR, "submission.csv")
    save_submission(df_test["Id"].values, pred_tag_strings, submission_path)

    # Verify File
    assert os.path.exists(submission_path)

    # Check content
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["Id", "Tags"]
    assert len(df_sub) == TEST_LIMIT
    print(f"Submission file created at {submission_path} with {len(df_sub)} rows.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
