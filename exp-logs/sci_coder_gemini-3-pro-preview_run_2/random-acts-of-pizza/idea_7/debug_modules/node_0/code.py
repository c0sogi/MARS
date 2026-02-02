import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_and_merge_data
from library.feature_extraction import extract_features
from library.stacking_manager import StackingEngine


def run_demo():
    print("Initializing Demo Execution...")

    # ==========================================
    # 1. Configuration Overrides for Demo
    # ==========================================
    # We modify the Config class attributes directly to adapt the pipeline
    # for a fast, small-scale demonstration.

    # Set up a separate working directory for the demo
    demo_working_dir = "./working/demo_execution"
    demo_submission_dir = "./working/demo_submission"
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "demo_submission.csv")

    # Redirect cache paths to the demo directory
    Config.CACHE_TEXT_TRAIN = os.path.join(demo_working_dir, "X_text_train.npy")
    Config.CACHE_TEXT_VAL = os.path.join(demo_working_dir, "X_text_val.npy")
    Config.CACHE_TEXT_TEST = os.path.join(demo_working_dir, "X_text_test.npy")
    Config.CACHE_META_TRAIN = os.path.join(demo_working_dir, "df_meta_train.parquet")
    Config.CACHE_META_VAL = os.path.join(demo_working_dir, "df_meta_val.parquet")
    Config.CACHE_META_TEST = os.path.join(demo_working_dir, "df_meta_test.parquet")

    # Reduce complexity for speed
    Config.N_FOLDS = 2  # Use 2 folds instead of 5
    Config.BAGGING_PARAMS["n_estimators"] = 2  # Reduce bagging iterations

    # Minimize Hyperparameter Search Space
    # We provide a single option to skip actual grid search time
    Config.TEXT_EXPERT_GRID = {
        "C": [0.01],
        "penalty": ["l2"],
        "solver": ["liblinear"],
        "class_weight": ["balanced"],
    }
    Config.META_EXPERT_GRID = {
        "C": [1.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "class_weight": [None],
    }

    print("Configuration updated for fast execution.")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[Step 1] Loading Data...")
    # Load a small subset (100 samples) to ensure quick processing
    df_train, df_val, df_test = load_and_merge_data(debug=True, debug_size=100)

    # Validation
    assert len(df_train) == 100, f"Expected 100 train samples, got {len(df_train)}"
    assert len(df_val) == 100, f"Expected 100 val samples, got {len(df_val)}"
    assert len(df_test) == 100, f"Expected 100 test samples, got {len(df_test)}"
    assert "requester_received_pizza" in df_train.columns
    print("Data loaded successfully.")

    # Create temporary metadata file for the debug test set.
    # The StackingEngine reads the test metadata from disk to align request_ids for submission.
    # Since we are using a subset, we must provide a matching subset metadata file.
    debug_test_meta_path = os.path.join(demo_working_dir, "test_debug.csv")
    df_test[["request_id", "sample_index", "source_file"]].to_csv(
        debug_test_meta_path, index=False
    )
    Config.TEST_META_PATH = debug_test_meta_path

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print("\n[Step 2] Extracting Features...")
    # We disable cache loading to force the feature extraction logic to run
    X_text_train, X_meta_train, X_text_val, X_meta_val, X_text_test, X_meta_test = (
        extract_features(df_train, df_val, df_test, load_cached_data=False)
    )

    # Validation
    # Text features should be (N, 384) for MiniLM-L6-v2
    assert X_text_train.shape == (
        100,
        384,
    ), f"Unexpected text feature shape: {X_text_train.shape}"
    # Meta features should match the number of numerical columns
    assert X_meta_train.shape == (
        100,
        len(Config.NUMERICAL_COLS),
    ), f"Unexpected meta feature shape: {X_meta_train.shape}"

    # Check for NaNs
    assert not np.isnan(X_text_train).any(), "Text features contain NaNs"
    assert not X_meta_train.isnull().values.any(), "Meta features contain NaNs"
    print("Feature extraction complete.")

    # ==========================================
    # 4. Model Training & Stacking
    # ==========================================
    print("\n[Step 3] Running Stacking Engine...")

    # Extract targets
    y_train = df_train["requester_received_pizza"].values
    y_val = df_val["requester_received_pizza"].values

    # Initialize Engine
    engine = StackingEngine(n_folds=Config.N_FOLDS, random_state=Config.RANDOM_SEED)

    # Run Pipeline
    # This handles CV, Base Learner Training, Meta Learner Training, and Prediction
    engine.run(
        X_text_train,
        X_meta_train,
        y_train,
        X_text_val,
        X_meta_val,
        y_val,
        X_text_test,
        X_meta_test,
    )
    print("Stacking pipeline finished.")

    # ==========================================
    # 5. Submission Verification
    # ==========================================
    print("\n[Step 4] Verifying Submission...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Columns
    expected_cols = ["request_id", "requester_received_pizza"]
    if list(df_submission.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_submission.columns)}"
        )

    # Check Length (Should match test set size, which is 100 in debug mode)
    if len(df_submission) != 100:
        raise ValueError(
            f"Submission length mismatch. Expected 100, got {len(df_submission)}"
        )

    # Check Values (Probabilities between 0 and 1)
    if not df_submission["requester_received_pizza"].between(0, 1).all():
        raise ValueError("Submission contains values outside [0, 1] range.")

    print("Submission verified successfully.")
    print(f"File saved to: {Config.SUBMISSION_PATH}")
    print("\nDemo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
