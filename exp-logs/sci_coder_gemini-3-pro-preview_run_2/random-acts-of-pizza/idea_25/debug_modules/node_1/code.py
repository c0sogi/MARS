import sys
import os
import shutil
import numpy as np
import pandas as pd
import logging
import warnings

# Add current directory to path so library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_manager import DataManager
from library.feature_engine import TextEmbedder, HomophilyTargetEncoder
from library.model_factory import ModelFactory
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Setup & Config Overrides for Speed
    # ==========================================
    print(">>> Setting up environment and overriding config for demo...")
    set_seed(42)

    # Override Config settings to run a fast, small-scale demo
    Config.DEBUG = True
    Config.DEV_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.N_FOLDS = 2  # Use 2 folds instead of 5
    Config.INNER_CV_FOLDS = 2  # Use 2 inner folds
    Config.BAGGING_N_ESTIMATORS = 2  # Reduce ensemble size
    Config.SBERT_BATCH_SIZE = 4  # Small batch size

    # Redirect working directories to a temporary demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update derived paths
    Config.TRAIN_EMBEDDINGS_PATH = os.path.join(
        Config.WORKING_DIR, "train_demo_subset_embeddings.npy"
    )
    Config.VAL_EMBEDDINGS_PATH = os.path.join(
        Config.WORKING_DIR, "val_demo_subset_embeddings.npy"
    )
    Config.TEST_EMBEDDINGS_PATH = os.path.join(
        Config.WORKING_DIR, "test_debug_embeddings.npy"
    )
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_merged.parquet"
    )
    Config.TEST_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "test_merged.parquet")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Manager Demo
    # ==========================================
    print("\n>>> Testing DataManager...")
    dm = DataManager()

    # Load train data (forced reload to test processing logic)
    df_train = dm.load_dataset("train", load_cached_data=False)
    print(f"Loaded Train Data Shape: {df_train.shape}")

    # Load test data
    df_test = dm.load_dataset("test", load_cached_data=False)
    print(f"Loaded Test Data Shape: {df_test.shape}")

    # Verification
    assert (
        len(df_train) == Config.DEV_SAMPLE_SIZE
    ), f"Expected {Config.DEV_SAMPLE_SIZE} train samples, got {len(df_train)}"
    assert (
        Config.TARGET_COL in df_train.columns
    ), "Target column missing in training data"
    assert Config.SUBREDDIT_COL in df_train.columns, "Subreddit column missing"
    assert Config.TEXT_COLS[0] in df_train.columns, "Text column missing"

    # ==========================================
    # 3. Text Embedder Demo
    # ==========================================
    print("\n>>> Testing TextEmbedder...")
    # Using CPU for demo to ensure compatibility if GPU is busy/unavailable, though GPU works too
    embedder = TextEmbedder(device="cpu")

    # Generate embeddings for the small training subset
    embeddings = embedder.generate_embeddings(
        df_train, save_path=None, load_cached=False
    )

    print(f"Generated Embeddings Shape: {embeddings.shape}")

    # Verification
    # SBERT 'all-MiniLM-L6-v2' outputs 384-dimensional vectors
    assert embeddings.shape == (len(df_train), 384), "Incorrect embedding dimensions"
    # Check L2 normalization (norm should be approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not L2 normalized"

    # ==========================================
    # 4. Homophily Encoder Demo (Logic Check)
    # ==========================================
    print("\n>>> Testing HomophilyTargetEncoder...")
    # Create synthetic data to verify Bayesian smoothing logic
    # Subreddit 'A': 1 success, 1 failure -> mean 0.5
    # Subreddit 'B': 2 successes -> mean 1.0
    # Global Mean: (1+0+1+1)/4 = 0.75
    dummy_df = pd.DataFrame(
        {
            Config.SUBREDDIT_COL: [["A"], ["A"], ["B"], ["B"]],
            Config.TARGET_COL: [1, 0, 1, 1],
        }
    )

    encoder = HomophilyTargetEncoder(smoothing=10.0)
    encoder.fit(dummy_df)

    # Manual Calculation
    # Smoothed Mean = (Sum + alpha * GlobalMean) / (Count + alpha)
    alpha = 10.0
    global_mean = 0.75

    # For Subreddit 'A' (Sum=1, Count=2)
    expected_score_a = (1 + alpha * global_mean) / (
        2 + alpha
    )  # (1 + 7.5) / 12 = 8.5/12 = 0.70833

    # Transform a test row containing 'A'
    test_dummy = pd.DataFrame({Config.SUBREDDIT_COL: [["A"]]})
    transformed = encoder.transform(test_dummy)

    print(f"Expected Score for 'A': {expected_score_a:.5f}")
    print(f"Calculated Score for 'A': {transformed[0, 0]:.5f}")

    assert np.isclose(
        transformed[0, 0], expected_score_a
    ), "Homophily encoder calculation mismatch"

    # ==========================================
    # 5. Model Factory Demo
    # ==========================================
    print("\n>>> Testing ModelFactory...")
    model = ModelFactory.get_classifier(n_estimators=2)
    print(f"Model Type: {type(model).__name__}")

    assert hasattr(model, "fit"), "Model object missing 'fit' method"
    assert hasattr(
        model, "predict_proba"
    ), "Model object missing 'predict_proba' method"

    # ==========================================
    # 6. Full Trainer Pipeline Demo
    # ==========================================
    print("\n>>> Testing Full Trainer Pipeline...")
    # The Trainer initializes its own components. Since we modified Config class attributes,
    # the Trainer will use our debug settings (small N, small folds, etc.)
    trainer = Trainer()

    # Run the cross-validation pipeline
    # This will load data, generate features, run CV, and save submission
    trainer.run_cross_validation(load_cached_data=False)

    # Verify Artifacts
    print("\n>>> Verifying Outputs...")

    # Check Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created with {len(sub_df)} rows.")
        assert (
            len(sub_df) == Config.DEV_SAMPLE_SIZE
        ), "Submission row count mismatch (debug mode)"
    else:
        raise FileNotFoundError("Submission file was not created.")

    # Check Model Artifacts for Fold 0
    model_fold_0 = os.path.join(Config.WORKING_DIR, "models", "model_fold_0.joblib")
    if os.path.exists(model_fold_0):
        print(f"Model artifact found: {model_fold_0}")
    else:
        raise FileNotFoundError(f"Model artifact missing: {model_fold_0}")

    print("\n>>> Demo execution completed successfully!")


if __name__ == "__main__":
    main()
