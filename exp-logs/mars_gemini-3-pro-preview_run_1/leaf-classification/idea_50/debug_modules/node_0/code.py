import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import make_classification

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import (
    SEED,
    WORKING_DIR,
    GEOMETRIC_FEATURES,
    METADATA_DIR,
    INPUT_DIR,
    ID_COL,
    TARGET_COL,
    IMAGE_PATH_COL,
)
from library.feature_engineering import (
    extract_single_image_features,
    load_and_process_data,
)
from library.preprocessing import RobustPipeline
from library.model import ParsimoniousOASClassifier, run_training_pipeline
from library.data_loader import load_data

# Configuration for the run
warnings.filterwarnings("ignore")
np.random.seed(SEED)


def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def demo_feature_engineering():
    print_header("DEMO: Feature Engineering")

    # 1. Test Single Image Feature Extraction
    print("Testing 'extract_single_image_features'...")

    # Load one row from metadata to get a valid image path
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df_meta = pd.read_csv(train_meta_path)
        # Get first image path
        sample_rel_path = df_meta.iloc[0][IMAGE_PATH_COL]
        print(f"Sample image path: {sample_rel_path}")

        # Extract features
        features = extract_single_image_features(sample_rel_path)

        # Validation
        assert isinstance(features, dict), "Output must be a dictionary"
        assert all(
            k in features for k in GEOMETRIC_FEATURES
        ), "Missing geometric features"
        assert not any(np.isnan(v) for v in features.values()), "Features contain NaNs"

        print("Successfully extracted features:")
        for k, v in list(features.items())[:3]:  # Print first 3
            print(f"  - {k}: {v:.4f}")
        print("  ... (and others)")
    else:
        print("Skipping image extraction test (metadata not found).")

    # 2. Test Batch Processing with a subset
    print("\nTesting 'load_and_process_data' with a subset...")

    # Create a temporary subset metadata file
    subset_csv_path = os.path.join(WORKING_DIR, "temp_subset_train.csv")
    if os.path.exists(train_meta_path):
        df_subset = pd.read_csv(train_meta_path).head(5)
        df_subset.to_csv(subset_csv_path, index=False)

        # Run processing (disable cache to force computation)
        df_processed = load_and_process_data(subset_csv_path, load_cached_data=False)

        # Validation
        assert len(df_processed) == 5, "Processed dataframe length mismatch"
        # Check if shape columns are dropped (assuming shape_1 was in original)
        assert "shape_1" not in df_processed.columns, "Shape columns were not dropped"
        # Check if geometric features are added
        assert "Area" in df_processed.columns, "Geometric features not added"

        print(
            "Batch processing successful. Shape columns dropped, Geometric features added."
        )

        # Cleanup
        if os.path.exists(subset_csv_path):
            os.remove(subset_csv_path)


def demo_preprocessing():
    print_header("DEMO: Robust Preprocessing Pipeline")

    # Generate synthetic data (float64)
    # 100 samples, 5 features
    X_dummy = np.random.rand(100, 5).astype(np.float64) * 100
    # Add some skew
    X_dummy = np.exp(X_dummy / 50.0)

    pipeline = RobustPipeline()

    # Test Fit
    print("Fitting pipeline...")
    pipeline.fit(X_dummy)
    assert pipeline.is_fitted, "Pipeline should be fitted"

    # Test Transform
    print("Transforming data...")
    X_trans = pipeline.transform(X_dummy)

    # Validation
    assert X_trans.shape == X_dummy.shape, "Output shape mismatch"
    assert X_trans.dtype == np.float64, "Output dtype must be float64"

    # Check statistics (StandardScaler should make mean ~0 and std ~1)
    means = np.mean(X_trans, axis=0)
    stds = np.std(X_trans, axis=0)

    print(f"Transformed Means (should be ~0): {means}")
    print(f"Transformed Stds (should be ~1): {stds}")

    assert np.allclose(means, 0, atol=1e-6), "Means are not centered"
    assert np.allclose(stds, 1, atol=1e-6), "Stds are not scaled"

    print("Preprocessing pipeline verification passed.")


def demo_model():
    print_header("DEMO: Parsimonious OAS Classifier")

    # Generate synthetic classification data
    # n_samples=200, n_features=20, n_classes=3
    X_train, y_train = make_classification(
        n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=SEED
    )
    X_test, _ = make_classification(
        n_samples=50,
        n_features=20,
        n_informative=10,
        n_classes=3,
        random_state=SEED + 1,
    )

    # Convert to pandas for compatibility with model expectations (optional but good practice)
    X_train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(20)])
    X_test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(20)])

    model = ParsimoniousOASClassifier(assume_centered=True)

    # Test Fit
    print("Fitting model...")
    model.fit(X_train_df, y_train)

    assert model.W_ is not None, "Weights W_ not computed"
    assert model.b_ is not None, "Bias b_ not computed"

    # Test Predict Proba
    print("Predicting probabilities...")
    probs = model.predict_proba(X_test_df)

    # Validation
    assert probs.shape == (50, 3), f"Probability shape mismatch: {probs.shape}"
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Test Predict
    print("Predicting labels...")
    preds = model.predict(X_test_df)
    assert len(preds) == 50, "Prediction length mismatch"

    print("Model verification passed.")


def demo_full_integration():
    print_header("DEMO: Full Integration (Data Loader + Pipeline)")

    # 1. Test Data Loader (which calls feature engineering and preprocessing)
    print("Calling 'load_data' (this processes the real dataset)...")
    # We use cached data if available to speed up, but the first run will compute
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(load_cached_data=True)

    print(f"Loaded Train: {X_train.shape}")
    print(f"Loaded Val:   {X_val.shape}")
    print(f"Loaded Test:  {X_test.shape}")

    # Verify Data Integrity
    assert not X_train.isnull().values.any(), "X_train contains NaNs"
    assert not X_val.isnull().values.any(), "X_val contains NaNs"

    # Verify Feature Engineering Result in Loaded Data
    # Check if 'Area' is present (one of the geometric features)
    assert (
        "Area" in X_train.columns
    ), "Geometric feature 'Area' missing from loaded data"

    # 2. Run the Full Training Pipeline Wrapper
    # This function in library/model.py orchestrates everything and creates a submission
    print("\nRunning 'run_training_pipeline'...")
    try:
        run_training_pipeline(load_cached_data=True)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise e

    # Verify Submission File Creation
    submission_path = os.path.join("./submission", "submission.csv")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created at {submission_path}")
        print(f"Submission shape: {df_sub.shape}")

        # Basic submission checks
        assert ID_COL in df_sub.columns, "ID column missing in submission"
        assert (
            df_sub.shape[0] == 99
        ), "Submission row count mismatch (expected 99 for test set)"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    demo_feature_engineering()
    demo_preprocessing()
    demo_model()
    demo_full_integration()
