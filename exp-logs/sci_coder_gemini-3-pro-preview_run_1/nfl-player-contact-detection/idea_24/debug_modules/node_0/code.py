import os
import shutil
import pandas as pd
import numpy as np
import warnings
import joblib

# Import from the provided library
from library.config import Config
from library.workflow_manager import WorkflowManager
from library.data_loader import DataLoader
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def setup_demo_config():
    """
    Overrides the default Config parameters to ensure the demo runs quickly
    and uses a separate working directory.
    """
    print("Setting up demo configuration...")

    # 1. Paths
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update dependent paths
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.WORKING_DIR, "cache/features_train.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(
        Config.WORKING_DIR, "cache/features_val.parquet"
    )
    Config.CACHE_TEST_FEATURES = os.path.join(
        Config.WORKING_DIR, "cache/features_test.parquet"
    )
    Config.CACHE_HARD_NEGATIVES = os.path.join(
        Config.WORKING_DIR, "cache/hard_negative_indices.npy"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # 2. Runtime Parameters
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for speed
    Config.NUM_BOOST_ROUND = 5  # Minimal iterations
    Config.EARLY_STOPPING_ROUNDS = 2
    Config.VERBOSE_EVAL = -1  # Silent

    # 3. Model Parameters (Force CPU for tiny data to avoid overhead)
    Config.LGBM_PARAMS["device"] = "cpu"
    Config.LGBM_PARAMS["verbosity"] = -1

    Config.XGB_PARAMS["device"] = "cpu"
    Config.XGB_PARAMS["tree_method"] = "hist"
    Config.XGB_PARAMS["verbosity"] = 0

    Config.CAT_PARAMS["task_type"] = "CPU"
    Config.CAT_PARAMS["verbose"] = 0

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated.")


def verify_data_loader():
    """
    Verifies that the DataLoader correctly loads and processes data.
    """
    print("\n--- Verifying DataLoader ---")
    loader = DataLoader()

    # Load a small slice of training data
    # We disable cache loading to force feature computation
    df = loader.get_train_data(debug=True, load_cached_data=False)

    # Basic Checks
    assert isinstance(df, pd.DataFrame), "DataLoader returned incorrect type"
    assert not df.empty, "DataFrame is empty"
    assert "contact" in df.columns, "Target column 'contact' missing"
    assert "distance" in df.columns, "Feature 'distance' missing"

    # Check Vector Features
    expected_vectors = ["radial_velocity", "tangential_velocity"]
    for feat in expected_vectors:
        assert feat in df.columns, f"Vector feature {feat} missing"

    print(f"Data loaded successfully. Shape: {df.shape}")
    print("DataLoader verification passed.")


def run_full_workflow():
    """
    Executes the WorkflowManager to run training and inference phases.
    """
    print("\n--- Running Full Workflow (Training + Inference) ---")

    wm = WorkflowManager()

    # 1. Training Phase
    # This will train Scouts, Mine Hard Negatives, and Train Experts
    wm.run_training_phase(debug=True, load_cached_data=False)

    # Verify Models were saved
    assert os.path.exists(wm.expert_lgbm_path), "LGBM Expert model not saved"
    assert os.path.exists(wm.expert_xgb_path), "XGB Expert model not saved"
    assert os.path.exists(wm.expert_cat_path), "CatBoost Expert model not saved"
    assert os.path.exists(wm.threshold_path), "Threshold file not saved"

    print("Training phase completed and artifacts verified.")

    # 2. Inference Phase
    wm.run_inference_phase(debug=True, load_cached_data=False)

    print("Inference phase completed.")


def verify_submission():
    """
    Verifies the generated submission file format and content.
    """
    print("\n--- Verifying Submission ---")

    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)

    # Check Columns
    assert "contact_id" in df_sub.columns, "contact_id column missing"
    assert "contact" in df_sub.columns, "contact column missing"

    # Check Values
    unique_vals = df_sub["contact"].unique()
    valid_vals = {0, 1}
    assert set(unique_vals).issubset(
        valid_vals
    ), f"Invalid values in contact column: {unique_vals}"

    # Check Length (Should match the debug sample size logic roughly,
    # though exact rows depend on how many steps are in the sampled plays)
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Submission verified. Rows: {len(df_sub)}")
    print(df_sub.head())


if __name__ == "__main__":
    try:
        # 1. Setup
        setup_demo_config()

        # 2. Verify Data Loading logic
        verify_data_loader()

        # 3. Run Workflow
        run_full_workflow()

        # 4. Verify Output
        verify_submission()

        print("\nSUCCESS: All demonstration steps completed without error.")

    except AssertionError as e:
        print(f"\nFAILURE: Assertion failed - {e}")
        exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred - {e}")
        import traceback

        traceback.print_exc()
        exit(1)
