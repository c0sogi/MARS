import sys
import os
import pandas as pd
import numpy as np
import warnings
import torch

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, compute_kendall_tau
from library.trainer import Trainer
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def demo_metric_verification():
    """
    Validates the Kendall Tau metric calculation with known inputs.
    """
    print("\n=== 1. Metric Verification ===")

    # Case A: Perfect Prediction
    # Ground Truth: a b c
    # Prediction: a b c
    df_gt = pd.DataFrame({"id": ["nb_1"], "cell_order": ["a b c"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb_1"], "cell_order": ["a b c"]})

    score_perfect = compute_kendall_tau(df_pred_perfect, df_gt)
    print(f"Perfect Match Score: {score_perfect}")
    assert score_perfect == 1.0, f"Expected 1.0 for perfect match, got {score_perfect}"

    # Case B: Completely Reversed
    # Ground Truth: a b c (pairs: ab, ac, bc) -> 3 pairs
    # Prediction: c b a (pairs: cb, ca, ba) -> 3 inversions
    # Score = 1 - 4 * (3 / (3*2)) = 1 - 4 * (0.5) = -1.0
    df_pred_reverse = pd.DataFrame({"id": ["nb_1"], "cell_order": ["c b a"]})
    score_reverse = compute_kendall_tau(df_pred_reverse, df_gt)
    print(f"Reversed Match Score: {score_reverse}")
    assert (
        score_reverse == -1.0
    ), f"Expected -1.0 for reversed match, got {score_reverse}"

    print("Metric logic verified successfully.")


def configure_environment_for_demo():
    """
    Overrides Config defaults to ensure the demo runs quickly and
    does not overwrite production artifacts.
    """
    print("\n=== 2. Configuring Environment ===")

    # Set a demo-specific working directory
    demo_dir = "./working/demo_output"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir

    # Point cache files to demo dir to force re-processing on small subset
    Config.TRAIN_CACHE_PATH = os.path.join(demo_dir, "train_processed.parquet")
    Config.VAL_CACHE_PATH = os.path.join(demo_dir, "val_processed.parquet")
    Config.TEST_CACHE_PATH = os.path.join(demo_dir, "test_processed.parquet")

    # Point model artifacts to demo dir
    Config.TFIDF_VECTORIZER_PATH = os.path.join(demo_dir, "tfidf_vectorizer.joblib")
    Config.RIDGE_MODEL_PATH = os.path.join(demo_dir, "ridge_model.joblib")
    Config.CODE_TFIDF_VECTORIZER_PATH = os.path.join(
        demo_dir, "code_tfidf_vectorizer.joblib"
    )
    Config.TRANSFORMER_MODEL_PATH = os.path.join(demo_dir, "transformer_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VAL_BATCH_SIZE = 4
    Config.VOCAB_SIZE = 500  # Smaller TF-IDF vocab
    Config.MAX_LEN = 32  # Shorter sequence length for Transformer
    Config.MAX_CODE_TOKENS_CONTEXT = 5

    # Set seed
    seed_everything(Config.SEED)
    print(f"Config updated. Working directory: {Config.WORKING_DIR}")


def demo_training_pipeline():
    """
    Demonstrates the Trainer class: loading data, feature engineering, and training.
    """
    print("\n=== 3. Running Training Pipeline ===")

    # Initialize Trainer with debug=True to use a small subset (e.g., 100 samples)
    # This automatically triggers load_data() and feature creation
    trainer = Trainer(debug=True)

    # Assertion: Verify data loading
    print(f"Train Data Shape: {trainer.train_df.shape}")
    assert not trainer.train_df.empty, "Training dataframe should not be empty."
    assert (
        "context" in trainer.train_df.columns
    ), "Context feature missing from dataframe."

    # Run the training loop (Ridge + Transformer)
    # This calls fit() on RidgeRanker and runs one epoch of TransformerRanker
    trainer.fit()

    # Assertion: Verify models are saved
    assert os.path.exists(Config.RIDGE_MODEL_PATH), "Ridge model file was not created."
    # Transformer might not save if validation score is -inf, but our code in Trainer
    # initializes best_score = -inf, so the first validation should trigger a save.
    assert os.path.exists(
        Config.TRANSFORMER_MODEL_PATH
    ), "Transformer model file was not created."

    print("Training pipeline completed successfully.")


def demo_inference_pipeline():
    """
    Demonstrates the Inference pipeline: loading models and generating submission.
    """
    print("\n=== 4. Running Inference Pipeline ===")

    # Run inference with debug=True to process a subset of test data
    run_inference(debug=True)

    # Assertion: Verify submission file creation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Assertion: Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Data Shape: {df_sub.shape}")
    print(f"Sample Submission:\n{df_sub.head(2)}")

    assert list(df_sub.columns) == [
        "id",
        "cell_order",
    ], "Submission columns are incorrect."
    assert len(df_sub) > 0, "Submission dataframe is empty."

    # Verify values are strings
    assert isinstance(
        df_sub.iloc[0]["cell_order"], str
    ), "cell_order should be a string."

    print("Inference pipeline completed successfully.")


if __name__ == "__main__":
    try:
        demo_metric_verification()
        configure_environment_for_demo()
        demo_training_pipeline()
        demo_inference_pipeline()
        print("\nAll demonstrations passed successfully.")
    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nEXECUTION ERROR: {e}")
        sys.exit(1)
