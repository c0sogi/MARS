import os
import sys
import numpy as np
import pandas as pd
import shutil

# 1. Setup and Configuration Override
# We need to override the configuration to use a smaller dataset and faster model settings
# for this demonstration.
import library.config as config

# Define a temporary working directory for this demo
DEMO_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
os.makedirs(DEMO_DIR, exist_ok=True)

# Override config paths to point to our sample metadata (which we will create shortly)
config.WORKING_DIR = DEMO_DIR
config.TRAIN_METADATA_PATH = os.path.join(DEMO_DIR, "train_metadata_sample.csv")
config.VAL_METADATA_PATH = os.path.join(DEMO_DIR, "val_metadata_sample.csv")
config.TEST_METADATA_PATH = os.path.join(DEMO_DIR, "test_metadata_sample.csv")
config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.parquet")
config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

# Override Model Hyperparameters for speed
config.XGB_PARAMS["n_estimators"] = 10
config.XGB_PARAMS["early_stopping_rounds"] = 2
config.XGB_PARAMS["max_depth"] = 3
config.XGB_PARAMS["learning_rate"] = 0.1

# Import Library Modules after config setup
from library.data_manager import DataManager
from library.model_factory import DualModelWrapper, generate_submission


def create_sample_data():
    """
    Creates a small subset of the metadata to allow rapid feature extraction and training.
    """
    print("Creating sample metadata for demonstration...")

    # Read original metadata
    # Note: We assume the metadata files exist as per the problem description.
    orig_train_path = "./metadata/train_metadata.csv"
    orig_val_path = "./metadata/val_metadata.csv"
    orig_test_path = "./metadata/test_metadata.csv"

    if not os.path.exists(orig_train_path):
        raise FileNotFoundError(f"Original metadata not found at {orig_train_path}")

    # Sample size
    N_SAMPLE = 50

    # Train
    df_train = pd.read_csv(orig_train_path)
    df_train_sample = df_train.head(N_SAMPLE).copy()
    df_train_sample.to_csv(config.TRAIN_METADATA_PATH, index=False)

    # Val
    df_val = pd.read_csv(orig_val_path)
    df_val_sample = df_val.head(20).copy()  # Smaller val set
    df_val_sample.to_csv(config.VAL_METADATA_PATH, index=False)

    # Test
    df_test = pd.read_csv(orig_test_path)
    df_test_sample = df_test.head(20).copy()
    df_test_sample.to_csv(config.TEST_METADATA_PATH, index=False)

    print(f"Sample metadata created in {DEMO_DIR}")
    return len(df_train_sample), len(df_val_sample), len(df_test_sample)


def main():
    # Set seed
    np.random.seed(42)

    # 1. Prepare Data
    n_train, n_val, n_test = create_sample_data()

    # 2. Instantiate DataManager
    print("\n--- Initializing DataManager ---")
    dm = DataManager(working_dir=DEMO_DIR)

    # 3. Load and Process Data
    # We set load_cached_data=False to demonstrate the feature engineering pipeline running.
    # In a real scenario, this extracts RDF, Strain, and Topology features from .xyz files.
    print("Processing data (Feature Engineering)...")
    (X_train, y_train), (X_val, y_val), (X_test, ids_test) = dm.load_and_process_data(
        load_cached_data=False
    )

    # Validation of Data Processing
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    assert (
        len(X_train) == n_train
    ), f"Expected {n_train} training samples, got {len(X_train)}"
    assert len(X_val) == n_val, f"Expected {n_val} validation samples, got {len(X_val)}"
    assert len(X_test) == n_test, f"Expected {n_test} test samples, got {len(X_test)}"

    # Check if features were actually generated (columns > metadata columns)
    # Metadata has ~14 columns. Processed should have many more (RDF bins etc.)
    assert (
        X_train.shape[1] > 20
    ), "Feature extraction seems to have failed to generate enough features."

    # Check for NaNs
    assert not X_train.isnull().values.any(), "X_train contains NaNs"
    assert not y_train.isnull().values.any(), "y_train contains NaNs"

    # 4. Model Training
    print("\n--- Initializing and Training Model ---")
    model_wrapper = DualModelWrapper()

    # Train the model
    model_wrapper.train(X_train, y_train, X_val, y_val)

    # 5. Prediction
    print("\n--- Generating Predictions ---")
    preds = model_wrapper.predict(X_test)

    # Validation of Predictions
    assert "formation_energy_ev_natom" in preds
    assert "bandgap_energy_ev" in preds
    assert len(preds["formation_energy_ev_natom"]) == n_test

    # Check for non-negative values (physics constraint enforced in predict)
    assert np.all(
        preds["formation_energy_ev_natom"] >= 0
    ), "Negative formation energy predicted"
    assert np.all(preds["bandgap_energy_ev"] >= 0), "Negative bandgap energy predicted"

    print(
        "Sample Predictions (Formation Energy):", preds["formation_energy_ev_natom"][:5]
    )

    # 6. Submission Generation
    print("\n--- Creating Submission File ---")
    generate_submission(
        model_wrapper, X_test, ids_test, output_path=config.SUBMISSION_PATH
    )

    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Successfully generated submission at {config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
