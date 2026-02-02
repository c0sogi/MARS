import os
import numpy as np
import pandas as pd
import warnings
import joblib
import shutil

# Import from the provided library
from library.config import Config
from library.data_loader import load_dataset
from library.text_processing import SBERTEncoder
from library.feature_engineering import (
    SubredditPLSProjector,
    MetadataScaler,
    assemble_feature_matrix,
)
from library.model_factory import create_classifier
from library.training_pipeline import run_stratified_cv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def test_data_loader():
    print("\n=== Testing Data Loader ===")

    # Load a small sample of the training data
    sample_size = 50
    df = load_dataset("train", load_cached_data=False, sample_size=sample_size)

    # Assertions
    assert isinstance(df, pd.DataFrame), "Data loader should return a DataFrame"
    assert len(df) == sample_size, f"Expected {sample_size} rows, got {len(df)}"

    # Check required columns
    expected_cols = (
        Config.TEXT_COLS
        + [Config.SUBREDDIT_COL]
        + Config.NUMERICAL_COLS
        + [Config.TARGET_COL]
    )
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check data types
    assert pd.api.types.is_numeric_dtype(
        df[Config.TARGET_COL]
    ), "Target column should be numeric"
    print("Data Loader tests passed.")


def test_text_processing():
    print("\n=== Testing Text Processing (SBERTEncoder) ===")

    encoder = SBERTEncoder()

    # Dummy text data
    texts = ["Pizza is great", "I am hungry", "Requesting help"]

    # Encode
    embeddings = encoder.encode(texts)

    # Assertions
    assert isinstance(embeddings, np.ndarray), "Encoder should return numpy array"
    assert embeddings.shape[0] == len(texts), "Should have one embedding per text"
    assert (
        embeddings.shape[1] == 384
    ), f"SBERT MiniLM should return 384 dimensions, got {embeddings.shape[1]}"

    # Check L2 Normalization (norm should be close to 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings should be L2 normalized"

    print("Text Processing tests passed.")


def test_feature_engineering():
    print("\n=== Testing Feature Engineering ===")

    n_samples = 20

    # 1. Test SubredditPLSProjector
    print("Testing SubredditPLSProjector...")
    # Create dummy subreddit strings (space separated)
    subreddits = ["subA subB", "subB subC", "subA", "subC subD"] * (n_samples // 4)
    # Create dummy target
    y = np.random.randint(0, 2, size=n_samples)

    n_components = 2
    pls = SubredditPLSProjector(n_components=n_components)

    # Fit
    pls.fit(subreddits, y)

    # Transform
    X_pls = pls.transform(subreddits)

    assert X_pls.shape == (
        n_samples,
        n_components,
    ), f"PLS output shape mismatch. Expected {(n_samples, n_components)}, got {X_pls.shape}"

    # 2. Test MetadataScaler
    print("Testing MetadataScaler...")
    # Create dummy numerical data
    n_features = 5
    X_meta_raw = np.random.rand(n_samples, n_features) * 100

    scaler = MetadataScaler()
    scaler.fit(X_meta_raw)
    X_meta_scaled = scaler.transform(X_meta_raw)

    assert (
        X_meta_scaled.shape == X_meta_raw.shape
    ), "Scaler output shape should match input"
    # RankGauss (QuantileTransformer output_distribution='normal') should produce values roughly in normal range
    assert -5 < X_meta_scaled.mean() < 5, "Scaled mean is wildly out of expected range"

    # 3. Test Feature Assembly
    print("Testing Feature Assembly...")
    # Create dummy views
    dim_emb = 10
    X_emb = np.random.rand(n_samples, dim_emb)

    X_fused = assemble_feature_matrix(X_emb, X_pls, X_meta_scaled)

    expected_dim = dim_emb + n_components + n_features
    assert X_fused.shape == (
        n_samples,
        expected_dim,
    ), f"Fused matrix shape mismatch. Expected {(n_samples, expected_dim)}, got {X_fused.shape}"

    print("Feature Engineering tests passed.")


def test_model_factory():
    print("\n=== Testing Model Factory ===")

    n_samples = 50
    n_features = 20

    # Generate dummy data
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)

    # Create classifier
    clf = create_classifier(n_estimators=5, C=1.0)

    # Fit
    clf.fit(X, y)

    # Predict
    probs = clf.predict_proba(X)

    # Assertions
    assert probs.shape == (n_samples, 2), "Predict proba should return (n_samples, 2)"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"

    print("Model Factory tests passed.")


def run_pipeline_demo():
    print("\n=== Running Full Pipeline (Debug Mode) ===")

    # Ensure working directory is clean or exists
    if os.path.exists(Config.WORKING_DIR):
        # We don't delete it to avoid removing pre-computed embeddings if they exist,
        # but for a clean demo, we rely on the pipeline's logic.
        pass

    # Run the stratified CV pipeline with debug=True
    # This triggers:
    # 1. Data Loading (subset)
    # 2. Embedding Generation (full or subset depending on cache, but sliced later)
    # 3. Feature Engineering (PLS, Scaling)
    # 4. Model Training (5 folds)
    # 5. Submission Generation
    try:
        run_stratified_cv(debug=True)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    # Verify Output
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns
    assert len(df_sub) > 0, "Submission file is empty"


if __name__ == "__main__":
    set_seed(42)

    # Run Unit Tests
    test_data_loader()
    test_text_processing()
    test_feature_engineering()
    test_model_factory()

    # Run Integration Test
    run_pipeline_demo()

    print("\nAll demonstrations passed successfully.")
