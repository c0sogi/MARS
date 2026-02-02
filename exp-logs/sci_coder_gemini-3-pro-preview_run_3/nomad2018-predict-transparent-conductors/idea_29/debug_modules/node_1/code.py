import os
import pandas as pd
import numpy as np
import shutil
import library.config as config
import library.data_manager as data_manager
import library.model_handler as model_handler


def run_demo():
    print("Starting demonstration of library usage...")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    # Define a separate working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config paths to use our sampled metadata (which we will create shortly)
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_working_dir
    config.TRAIN_METADATA_PATH = os.path.join(
        demo_working_dir, "train_metadata_sample.csv"
    )
    config.VAL_METADATA_PATH = os.path.join(demo_working_dir, "val_metadata_sample.csv")
    config.TEST_METADATA_PATH = os.path.join(
        demo_working_dir, "test_metadata_sample.csv"
    )
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "demo_submission.csv")

    # Optimize hyperparameters for speed in this demo
    config.XGB_PARAMS["n_estimators"] = 10  # Reduce from 3000
    config.XGB_PARAMS["max_depth"] = 3  # Reduce depth
    config.RDF_BINS = 20  # Reduce bins for faster feature extraction
    config.NEIGHBOR_CUTOFF = 5.0  # Slightly reduce cutoff

    print(f"Configuration overridden. Working directory: {config.WORKING_DIR}")

    # ==========================================
    # 2. Create Sample Metadata
    # ==========================================
    # We load the original metadata and sample a few rows to create a mini-dataset
    # This ensures the file paths point to actual existing files in ./input

    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Take a small sample (e.g., 50 training, 20 validation, 20 test)
    sample_train = orig_train_meta.head(50)
    sample_val = orig_val_meta.head(20)
    sample_test = orig_test_meta.head(20)

    # Save these samples to the new paths defined in config
    sample_train.to_csv(config.TRAIN_METADATA_PATH, index=False)
    sample_val.to_csv(config.VAL_METADATA_PATH, index=False)
    sample_test.to_csv(config.TEST_METADATA_PATH, index=False)

    print(
        f"Created sample metadata: Train({len(sample_train)}), Val({len(sample_val)}), Test({len(sample_test)})"
    )

    # ==========================================
    # 3. Data Loading & Feature Extraction
    # ==========================================
    print("\n--- Building Datasets (Feature Extraction) ---")

    # Build Training Data
    # load_cached_data=False forces re-calculation to demonstrate the logic
    train_df = data_manager.build_dataset("train", load_cached_data=False)
    print(f"Train dataset built. Shape: {train_df.shape}")

    # Verify features were created
    # We expect columns starting with 'rdf_', 'bvs_', etc.
    feature_cols = [
        c for c in train_df.columns if c.startswith("rdf_") or c.startswith("bvs_")
    ]
    assert (
        len(feature_cols) > 0
    ), "No features were extracted! Check feature_extraction.py logic."
    assert (
        "formation_energy_ev_natom" in train_df.columns
    ), "Target column missing in train_df"

    # Build Validation Data
    val_df = data_manager.build_dataset("val", load_cached_data=False)
    print(f"Validation dataset built. Shape: {val_df.shape}")
    assert len(val_df) == len(sample_val), "Validation dataset size mismatch."

    # Build Test Data
    test_df = data_manager.build_dataset("test", load_cached_data=False)
    print(f"Test dataset built. Shape: {test_df.shape}")
    assert len(test_df) == len(sample_test), "Test dataset size mismatch."

    # ==========================================
    # 4. Model Training
    # ==========================================
    print("\n--- Training Models ---")

    predictor = model_handler.EnergyPredictor()

    # Train the predictor
    # This handles log-transformation internally
    predictor.train(train_df, val_df)

    # Verify models are fitted
    assert hasattr(
        predictor.model_formation, "feature_importances_"
    ), "Formation energy model not fitted."
    assert hasattr(
        predictor.model_bandgap, "feature_importances_"
    ), "Bandgap energy model not fitted."
    print("Models trained successfully.")

    # ==========================================
    # 5. Prediction & Submission
    # ==========================================
    print("\n--- Generating Predictions ---")

    submission_df = predictor.predict(test_df)

    # Verify submission format
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"
    assert len(submission_df) == len(sample_test), "Submission length mismatch."

    # Check for non-negative values (physically required)
    assert (
        submission_df["formation_energy_ev_natom"] >= 0
    ).all(), "Negative formation energy predicted."
    assert (
        submission_df["bandgap_energy_ev"] >= 0
    ).all(), "Negative bandgap energy predicted."

    print("Predictions generated and verified.")
    print(submission_df.head())

    # Save submission
    model_handler.save_submission(submission_df)

    # Verify file existence
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"File successfully created at {config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
