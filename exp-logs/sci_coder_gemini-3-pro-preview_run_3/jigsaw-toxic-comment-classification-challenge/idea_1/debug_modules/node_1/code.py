import os
import sys
import pandas as pd
import numpy as np
import warnings
from scipy import sparse

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.utils import set_seed, get_logger
from library.data_processing import load_data, FeatureEngineer
from library.model import MultiLabelNBSVM
from library.workflow import train_validate, generate_submission

# Configuration
SAMPLE_SIZE = 1000  # Small subset for speed
SEED = 42
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def run_demo():
    # 1. Setup
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    logger = get_logger("demo_main")
    logger.info("Starting demonstration script...")

    # =========================================================================
    # PART 1: Component-Level Verification
    # =========================================================================
    logger.info("--- Part 1: Component-Level Verification ---")

    # A. Data Loading
    logger.info("Testing data loading...")
    # Load training data
    df_train = load_data("train", load_cached_data=False)

    # Verify Data Structure
    assert isinstance(df_train, pd.DataFrame), "load_data should return a DataFrame"
    assert "comment_text" in df_train.columns, "DataFrame must contain 'comment_text'"
    for col in LABEL_COLS:
        assert col in df_train.columns, f"DataFrame must contain label column '{col}'"

    # Slice for speed in component test
    df_subset = df_train.iloc[:500].copy()
    y_subset = df_subset[LABEL_COLS]
    logger.info(f"Loaded and sliced training data: {df_subset.shape}")

    # B. Feature Engineering
    logger.info("Testing FeatureEngineer...")
    # Initialize with limited features for speed
    fe = FeatureEngineer(max_features_word=1000, max_features_char=1000)

    # Fit and Transform
    X_features = fe.fit_transform(
        df_subset["comment_text"], load_cached_data=False, cache_suffix="demo"
    )

    # Verify Feature Matrix
    assert sparse.issparse(
        X_features
    ), "Output of FeatureEngineer should be a sparse matrix"
    assert (
        X_features.shape[0] == 500
    ), f"Feature matrix rows {X_features.shape[0]} != 500"
    assert X_features.shape[1] <= 2000, "Feature dimension exceeds max_features limit"
    logger.info(f"Feature matrix shape: {X_features.shape}")

    # C. Model Training (NBSVM)
    logger.info("Testing MultiLabelNBSVM...")
    model = MultiLabelNBSVM(C=1.0, max_iter=50, random_state=SEED)

    # Fit Model
    model.fit(X_features, y_subset)

    # Predict
    probs = model.predict_proba(X_features)

    # Verify Predictions
    assert probs.shape == (500, 6), f"Prediction shape {probs.shape} mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"
    logger.info("Model training and prediction successful.")

    # =========================================================================
    # PART 2: Workflow Verification
    # =========================================================================
    logger.info("\n--- Part 2: Workflow Verification ---")

    # A. Train and Validate Pipeline
    # This function handles loading (with caching), feature engineering, training, and scoring
    logger.info(f"Running train_validate with sample_size={SAMPLE_SIZE}...")
    trained_model, fitted_fe = train_validate(
        load_cached_data=False,  # Force re-compute for demo
        sample_size=SAMPLE_SIZE,
        C=1.0,
        max_iter=50,
        max_features_word=2000,
        max_features_char=2000,
        seed=SEED,
    )

    # Verify Workflow Outputs
    assert isinstance(
        trained_model, MultiLabelNBSVM
    ), "train_validate should return a model"
    assert isinstance(
        fitted_fe, FeatureEngineer
    ), "train_validate should return a feature engineer"
    logger.info("train_validate completed successfully.")

    # B. Submission Generation
    # This function loads test data, transforms it using the fitted FE, predicts, and saves CSV
    logger.info(f"Running generate_submission with sample_size={SAMPLE_SIZE}...")
    generate_submission(
        model=trained_model,
        feature_engineer=fitted_fe,
        load_cached_data=False,
        sample_size=SAMPLE_SIZE,
    )

    # =========================================================================
    # PART 3: Submission File Validation
    # =========================================================================
    logger.info("\n--- Part 3: Submission Validation ---")

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    logger.info(f"Loaded submission file: {df_sub.shape}")

    # Verify Dimensions
    # We used sample_size=SAMPLE_SIZE for the test set in generate_submission
    assert len(df_sub) == SAMPLE_SIZE, f"Submission rows {len(df_sub)} != {SAMPLE_SIZE}"

    # Verify Columns
    expected_cols = ["id"] + LABEL_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Columns mismatch. Found: {df_sub.columns}"

    # Verify IDs
    assert df_sub["id"].dtype == object, "ID column should be object/string"
    assert df_sub["id"].nunique() == SAMPLE_SIZE, "IDs must be unique"

    # Verify Values
    for col in LABEL_COLS:
        assert df_sub[col].min() >= 0, f"Column {col} has values < 0"
        assert df_sub[col].max() <= 1, f"Column {col} has values > 1"
        assert df_sub[col].dtype == float, f"Column {col} should be float"

    logger.info("Submission file passed all validation checks.")
    logger.info("Demonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
