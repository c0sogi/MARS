import os
import sys
import numpy as np
import pandas as pd
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ====================================================
# 1. Setup & Monkey Patching Configuration
# ====================================================
# We need to modify the configuration to run on a small subset of data
# and use a temporary directory for outputs to avoid overwriting real work.
import library.config as config

# Define temporary paths
DEMO_DIR = "./working/demo_execution"
os.makedirs(DEMO_DIR, exist_ok=True)

MINI_TRAIN_META = os.path.join(DEMO_DIR, "mini_train_metadata.csv")
MINI_VAL_META = os.path.join(DEMO_DIR, "mini_val_metadata.csv")
MINI_TEST_META = os.path.join(DEMO_DIR, "mini_test_metadata.csv")
DEMO_SUBMISSION = os.path.join(DEMO_DIR, "submission.csv")

# Patch config variables
config.IDEA_DIR = DEMO_DIR
config.CACHE_DIR = DEMO_DIR
config.SUBMISSION_DIR = DEMO_DIR
config.SUBMISSION_PATH = DEMO_SUBMISSION
config.TRAIN_META_PATH = MINI_TRAIN_META
config.VAL_META_PATH = MINI_VAL_META
config.TEST_META_PATH = MINI_TEST_META

# Reduce PCA components to ensure it works with very few samples
config.PCA_COMPONENTS = 2

# Import library modules after patching config
from library.utils import seed_everything, load_numpy
from library.feature_generation import run_feature_generation
from library.preprocessing import run_preprocessing
from library.modeling import train_models, generate_submission


def create_mini_metadata():
    """
    Creates small subsets of the original metadata to allow for fast execution.
    """
    print("Creating mini-metadata for demonstration...")

    # Load original metadata
    # Note: We assume the original metadata files exist as per the task description
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Sample patients (ensure we have enough for the pipeline to run without errors)
    # We take 5 unique patients for train, 2 for val, 2 for test
    train_patients = orig_train["Patient"].unique()[:5]
    val_patients = orig_val["Patient"].unique()[:2]
    test_patients = orig_test["Patient"].unique()[:2]

    mini_train = orig_train[orig_train["Patient"].isin(train_patients)].copy()
    mini_val = orig_val[orig_val["Patient"].isin(val_patients)].copy()
    mini_test = orig_test[orig_test["Patient"].isin(test_patients)].copy()

    # Save to demo directory
    mini_train.to_csv(MINI_TRAIN_META, index=False)
    mini_val.to_csv(MINI_VAL_META, index=False)
    mini_test.to_csv(MINI_TEST_META, index=False)

    print(
        f"Mini-metadata created: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
    )
    return train_patients, val_patients, test_patients


def main():
    # Ensure reproducibility
    seed_everything(config.SEED)

    # 1. Prepare Data
    create_mini_metadata()

    # 2. Feature Generation
    # We set load_cached_data=False to demonstrate the actual extraction logic
    print("\n=== Step 1: Feature Generation ===")
    train_feats, val_feats, test_feats = run_feature_generation(load_cached_data=False)

    # Verification
    print("Verifying Feature Generation...")
    # Check if we got a dictionary
    assert isinstance(train_feats, dict), "Train features should be a dictionary"

    # Check feature dimensions: PCA_COMPONENTS (2) + Volume (1) = 3
    sample_feat = next(iter(train_feats.values()))
    expected_dim = config.PCA_COMPONENTS + 1
    assert sample_feat.shape == (
        expected_dim,
    ), f"Expected feature dim {expected_dim}, got {sample_feat.shape}"

    print(f"Success. Extracted features for {len(train_feats)} training patients.")

    # 3. Preprocessing
    print("\n=== Step 2: Preprocessing ===")
    # This fuses image features with tabular data and creates interaction terms
    data_dict = run_preprocessing(
        train_feats, val_feats, test_feats, load_cached_data=False
    )

    # Verification
    print("Verifying Preprocessing...")
    required_keys = ["X_fvc_train", "y_train", "X_unc_train", "X_fvc_test"]
    for key in required_keys:
        assert key in data_dict, f"Missing key in data_dict: {key}"
        assert isinstance(data_dict[key], np.ndarray), f"{key} should be a numpy array"

    # Check samples align with mini-metadata rows
    # Note: mini_train has multiple rows per patient (history)
    mini_train_df = pd.read_csv(MINI_TRAIN_META)
    assert len(data_dict["X_fvc_train"]) == len(
        mini_train_df
    ), f"Mismatch in training samples: {len(data_dict['X_fvc_train'])} vs {len(mini_train_df)}"

    print(
        f"Success. Preprocessed data shapes: X_fvc_train={data_dict['X_fvc_train'].shape}"
    )

    # 4. Modeling
    print("\n=== Step 3: Modeling ===")
    fvc_model, unc_model = train_models(data_dict)

    # Verification
    print("Verifying Models...")
    assert fvc_model is not None, "FVC Model is None"
    assert unc_model is not None, "Uncertainty Model is None"
    assert fvc_model.result is not None, "FVC Model is not fitted"
    assert unc_model.result is not None, "Uncertainty Model is not fitted"

    print("Success. Models trained.")

    # 5. Inference / Submission
    print("\n=== Step 4: Inference & Submission ===")
    generate_submission(
        fvc_model, unc_model, data_dict["X_fvc_test"], data_dict["X_unc_test"]
    )

    # Verification
    print("Verifying Submission...")
    assert os.path.exists(DEMO_SUBMISSION), "Submission file was not created"

    sub_df = pd.read_csv(DEMO_SUBMISSION)
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check values are reasonable (FVC > 0, Confidence > 0)
    assert (sub_df["FVC"] > 0).all(), "Negative FVC predictions found"
    assert (sub_df["Confidence"] > 0).all(), "Negative Confidence predictions found"

    print(f"Success. Submission generated with {len(sub_df)} rows.")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
