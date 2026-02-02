import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import warnings

# Import provided library modules
from library.config import Config
from library.data_loader import get_data_splits, read_notebook, create_relaxed_pairs
from library.fine_tuning import train_semantic_model
from library.feature_engineering import generate_features_pipeline
from library.regressor import train_lgbm_regressor
from library.inference import generate_submission_file
from library.metrics import kendall_tau_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a temporary working directory and overrides Config parameters
    to run the pipeline on a tiny subset of data for demonstration purposes.
    """
    print(">>> Setting up demo environment...")

    # Define new paths
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    metadata_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.METADATA_DIR = metadata_dir
    Config.SUBMISSION_DIR = demo_dir

    Config.TRAIN_METADATA_PATH = os.path.join(metadata_dir, "train.csv")
    Config.VAL_METADATA_PATH = os.path.join(metadata_dir, "val.csv")
    Config.TEST_METADATA_PATH = os.path.join(metadata_dir, "test.csv")

    Config.MODEL_OUTPUT_PATH = os.path.join(demo_dir, "dsapr_model")
    Config.LGBM_MODEL_PATH = os.path.join(demo_dir, "lgbm_model.txt")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(demo_dir, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "test_features.parquet")

    # Override Hyperparameters for speed
    Config.NUM_FINE_TUNE_NOTEBOOKS = 20  # Only use 20 notebooks for fine-tuning
    Config.EPOCHS = 1
    Config.WARMUP_STEPS = 2
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead/issues in demo

    Config.LGBM_NUM_BOOST_ROUND = 10
    Config.LGBM_EARLY_STOPPING_ROUNDS = 5
    Config.LGBM_PARAMS["verbosity"] = -1

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # Create subset metadata files
    print(">>> Creating metadata subsets...")
    original_train = pd.read_csv("./metadata/train.csv")
    original_val = pd.read_csv("./metadata/val.csv")
    original_test = pd.read_csv("./metadata/test.csv")

    # Sample 20 rows for each split
    subset_train = original_train.head(20)
    subset_val = original_val.head(20)
    subset_test = original_test.head(20)

    subset_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    subset_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    subset_test.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(f"Subset train size: {len(subset_train)}")
    print(f"Subset val size: {len(subset_val)}")
    print(f"Subset test size: {len(subset_test)}")


def demonstrate_data_loading():
    print("\n>>> Demonstrating Data Loading...")

    # 1. Test get_data_splits
    df_train, df_val, df_test = get_data_splits()
    assert len(df_train) == 20
    assert len(df_val) == 20
    assert len(df_test) == 20
    print("Data splits loaded successfully.")

    # 2. Test read_notebook
    sample_path = df_train.iloc[0]["file_path"]
    nb_data = read_notebook(sample_path)
    assert "cell_type" in nb_data
    assert "source" in nb_data
    print(f"Successfully read notebook: {sample_path}")

    # 3. Test create_relaxed_pairs
    # This uses the subset metadata we created
    pairs_df = create_relaxed_pairs(df_train, load_cached_data=False)
    assert "markdown" in pairs_df.columns
    assert "code" in pairs_df.columns
    print(f"Generated {len(pairs_df)} pairs for fine-tuning.")


def demonstrate_fine_tuning():
    print("\n>>> Demonstrating Semantic Model Fine-Tuning...")

    # This will train on the pairs generated from the 20 notebooks
    # Config.EPOCHS is set to 1, so it should be fast.
    train_semantic_model(load_cached_data=False)

    assert os.path.exists(
        Config.MODEL_OUTPUT_PATH
    ), "Fine-tuned model directory not created."
    print(f"Model saved to {Config.MODEL_OUTPUT_PATH}")


def demonstrate_regressor_training():
    print("\n>>> Demonstrating Feature Engineering and Regressor Training...")

    # train_lgbm_regressor calls generate_features_pipeline internally.
    # Because we updated Config.TRAIN_METADATA_PATH, it uses the subset.
    bst = train_lgbm_regressor(load_cached_data=False)

    assert os.path.exists(Config.LGBM_MODEL_PATH), "LightGBM model file not created."
    assert bst is not None, "Regressor training returned None."
    print("Regressor trained and saved.")

    # Verify feature cache was created
    assert os.path.exists(
        Config.TRAIN_FEATURES_PATH
    ), "Train features parquet not found."
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features parquet not found."


def demonstrate_inference():
    print("\n>>> Demonstrating Inference...")

    generate_submission_file(load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df_sub.columns
    assert "cell_order" in df_sub.columns
    assert len(df_sub) == 20, f"Expected 20 predictions, got {len(df_sub)}"

    print(f"Submission generated with {len(df_sub)} rows.")
    print("Sample prediction:")
    print(df_sub.head(1))


def demonstrate_metrics():
    print("\n>>> Demonstrating Kendall Tau Metric...")

    # Case 1: Perfect match
    # Notebook with 3 cells: A, B, C
    # Ground Truth: A B C
    # Prediction: A B C
    # Swaps = 0. Pairs = 3*2 = 6. Score = 1 - 4*(0/6) = 1.0

    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["A B C"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["A B C"]})

    score_perfect = kendall_tau_metric(df_pred_perfect, df_gt)
    print(f"Perfect Match Score: {score_perfect}")
    assert np.isclose(score_perfect, 1.0), "Metric failed on perfect match."

    # Case 2: Complete inversion
    # Ground Truth: A B C
    # Prediction: C B A
    # Inversions: (C,B), (C,A), (B,A) -> 3.
    # Score = 1 - 4*(3/6) = 1 - 2 = -1.0

    df_pred_inverse = pd.DataFrame({"id": ["nb1"], "cell_order": ["C B A"]})

    score_inverse = kendall_tau_metric(df_pred_inverse, df_gt)
    print(f"Inverse Match Score: {score_inverse}")
    assert np.isclose(score_inverse, -1.0), "Metric failed on inverse match."

    print("Metric verification passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Data Loading
    demonstrate_data_loading()

    # 3. Fine Tuning
    # Note: This might take a minute or two depending on CPU/GPU
    demonstrate_fine_tuning()

    # 4. Regressor Training
    demonstrate_regressor_training()

    # 5. Inference
    demonstrate_inference()

    # 6. Metrics
    demonstrate_metrics()

    print("\n>>> All demonstrations completed successfully.")
