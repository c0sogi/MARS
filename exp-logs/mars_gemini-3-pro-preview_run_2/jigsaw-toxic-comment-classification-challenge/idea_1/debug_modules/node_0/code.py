import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
from scipy import sparse

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_datasets
from library.features import FeatureExtractor
from library.model import ToxicityClassifier
from library.evaluation import compute_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Solution Demonstration ===\n")

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to run a fast, lightweight demo.
    print("Step 1: Configuring environment for fast demonstration...")

    # Enable debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 1000

    # Reduce feature dimensionality for speed
    Config.WORD_MAX_FEATURES = 2000
    Config.CHAR_MAX_FEATURES = 2000

    # Reduce model iterations
    Config.LR_MAX_ITER = 50

    # Use a specific cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, Sample Size=1000.\n")

    # 2. Data Loading
    print("Step 2: Loading Datasets...")
    # Force reload to ensure we apply debug slicing logic correctly on fresh data
    train_df, val_df, test_df = load_datasets(load_cached_data=False, debug=True)

    # Validation: Check shapes
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_df)}"
    assert (
        len(val_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(val_df)}"
    assert (
        len(test_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(test_df)}"

    print(f"Data Loaded Successfully. Train shape: {train_df.shape}\n")

    # 3. Feature Extraction
    print("Step 3: Extracting Features (TF-IDF)...")
    extractor = FeatureExtractor()

    # We pass load_cached_data=False to force computation for the demo
    X_train, X_val, X_test = extractor.extract_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation: Check feature matrix properties
    assert sparse.issparse(X_train), "X_train should be a sparse matrix"
    assert X_train.shape[0] == len(train_df), "X_train row count mismatch"
    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature dimension mismatch across splits"

    print(f"Features Extracted. Feature Matrix Shape: {X_train.shape}\n")

    # 4. Model Training
    print("Step 4: Training Toxicity Classifier (Logistic Regression)...")
    classifier = ToxicityClassifier()

    # Train the model
    # Note: We extract the label columns from train_df and val_df
    y_train = train_df[Config.TARGET_COLS]
    y_val = val_df[Config.TARGET_COLS]

    mean_auc = classifier.train(X_train, y_train, X_val, y_val)

    # Validation: Check model state
    assert len(classifier.models) == len(
        Config.TARGET_COLS
    ), "Not all models were trained"
    assert 0.0 <= mean_auc <= 1.0, f"Invalid AUC score: {mean_auc}"

    print(f"Training Complete. Validation Mean AUC: {mean_auc:.4f}\n")

    # 5. Evaluation Verification
    print("Step 5: Verifying Evaluation Metric...")
    # Generate probabilities on validation set to manually verify compute_score
    val_probs_df = classifier.predict_proba(X_val)

    # Verify output structure
    assert (
        list(val_probs_df.columns) == Config.TARGET_COLS
    ), "Prediction columns mismatch"
    assert len(val_probs_df) == len(val_df), "Prediction row count mismatch"

    # Re-compute score using the library function
    calculated_score = compute_score(y_val, val_probs_df)

    # Allow for floating point minor differences, but they should be effectively identical
    assert (
        abs(mean_auc - calculated_score) < 1e-9
    ), "Mismatch between training returned score and evaluation function"

    print(f"Evaluation verified. Score: {calculated_score:.4f}\n")

    # 6. Submission Generation
    print("Step 6: Generating Submission...")
    test_ids = test_df["id"]
    classifier.generate_submission(X_test, test_ids)

    # Validation: Check submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    submission_df = pd.read_csv(submission_path)

    # Check dimensions: rows = test samples, cols = id + 6 labels
    expected_rows = len(test_df)
    expected_cols = 1 + len(Config.TARGET_COLS)

    assert submission_df.shape == (
        expected_rows,
        expected_cols,
    ), f"Submission shape mismatch. Expected {(expected_rows, expected_cols)}, got {submission_df.shape}"

    # Check column names
    expected_columns = ["id"] + Config.TARGET_COLS
    assert (
        list(submission_df.columns) == expected_columns
    ), "Submission columns mismatch"

    print(f"Submission generated successfully at {submission_path}")
    print(f"Submission Head:\n{submission_df.head(3)}\n")

    # 7. Cleanup
    print("Step 7: Cleaning up temporary files...")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
        print(f"Removed cache directory: {Config.CACHE_DIR}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
