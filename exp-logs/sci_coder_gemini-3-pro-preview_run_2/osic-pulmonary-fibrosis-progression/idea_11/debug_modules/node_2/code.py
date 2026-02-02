import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config, seed_everything
from library.dicom_processing import process_patient
from library.feature_extraction import run_feature_extraction
from library.modeling import run_modeling


def main():
    # 1. Setup and Configuration
    print("=== Setting up Demonstration ===")
    seed_everything(Config.SEED)

    # Override Config for speed (Debug Mode)
    # This limits the number of patients processed to ensure quick runtime
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5
    print(
        f"Configured for speed: DEBUG={Config.DEBUG}, SAMPLE_SIZE={Config.DEBUG_SAMPLE_SIZE}"
    )

    # Ensure working directory is clean for a fresh run logic check (optional but good for demo)
    if os.path.exists(Config.WORKING_DIR):
        print(f"Cleaning working directory: {Config.WORKING_DIR}")
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Demonstrate DICOM Processing
    print("\n=== 1. Testing DICOM Processing Module ===")

    # Load metadata to find a valid patient
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {Config.TRAIN_METADATA_PATH}"
        )

    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = train_meta.iloc[0]
    patient_id = sample_row["Patient"]
    dcm_path = sample_row["dcm_path"]

    print(f"Processing single patient: {patient_id}")

    # Process patient (force no cache to test processing logic)
    processed_data = process_patient(patient_id, dcm_path, load_cached_data=False)

    images = processed_data["images"]
    density = processed_data["density"]

    print(f"  Output Images Shape: {images.shape}")
    print(f"  Output Density Profile: {density}")

    # Validation
    expected_channels = 3
    expected_size = Config.IMAGE_SIZE
    # Expected views: NUM_AXIAL_SLICES + 1 (if CORONAL is True)
    expected_views = Config.NUM_AXIAL_SLICES + (1 if Config.USE_CORONAL else 0)

    # Check dimensions
    assert images.ndim == 4, f"Images should be 4D array, got {images.ndim}"
    assert images.shape[1:] == (
        expected_size,
        expected_size,
        expected_channels,
    ), f"Image spatial dimensions mismatch. Expected {(expected_size, expected_size, expected_channels)}, got {images.shape[1:]}"

    # Check density profile
    assert density.shape == (
        4,
    ), f"Density profile should have 4 bins, got {density.shape}"
    # Sum should be close to 1.0 (normalized) or 0.0 (if volume loading failed completely)
    assert np.isclose(density.sum(), 1.0) or np.isclose(
        density.sum(), 0.0
    ), f"Density profile sum invalid: {density.sum()}"

    print("DICOM Processing verification passed.")

    # 3. Demonstrate Feature Extraction
    print("\n=== 2. Testing Feature Extraction Pipeline ===")

    # Run the full feature extraction pipeline
    # This handles image embedding (EfficientNet), Tabular encoding, and PCA
    # load_cached_data=False ensures we actually run the generation code
    data_dict = run_feature_extraction(load_cached_data=False)

    # Validation of Data Structures
    for split in ["train", "val", "test"]:
        assert split in data_dict, f"Missing {split} split in feature dictionary"
        df, X_pca = data_dict[split]

        print(
            f"  [{split.upper()}] DataFrame Shape: {df.shape}, Feature Matrix Shape: {X_pca.shape}"
        )

        # Check alignment
        assert (
            len(df) == X_pca.shape[0]
        ), f"Row count mismatch for {split}: DataFrame {len(df)} vs Matrix {X_pca.shape[0]}"

        # Check dimensionality reduction
        # Cite debug_lesson_7: Adapt Static Hyperparameters to Runtime Data Dimensions
        # PCA is fitted on the training set, so components are limited by training sample count
        n_train = len(data_dict["train"][0])
        expected_components = min(Config.N_PCA_COMPONENTS, n_train)

        assert (
            X_pca.shape[1] == expected_components
        ), f"PCA components mismatch. Expected {expected_components}, got {X_pca.shape[1]}"

    print("Feature Extraction verification passed.")

    # 4. Demonstrate Modeling and Inference
    print("\n=== 3. Testing Modeling and Inference ===")

    # Run modeling (Training -> Validation -> Inference -> Submission)
    run_modeling(data_dict)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    print(f"  Loading submission file: {submission_path}")
    sub_df = pd.read_csv(submission_path)

    print("  Submission Head:")
    print(sub_df.head())

    # Validation of Submission Format
    required_columns = {"Patient_Week", "FVC", "Confidence"}
    assert required_columns.issubset(
        sub_df.columns
    ), f"Submission missing required columns. Found: {sub_df.columns}"

    # Validation of Value Constraints
    # Confidence should be >= 70 as per metric definition (clipped in modeling)
    min_conf = sub_df["Confidence"].min()
    assert min_conf >= 70, f"Found Confidence value < 70: {min_conf}"

    # FVC should be positive
    min_fvc = sub_df["FVC"].min()
    assert min_fvc > 0, f"Found non-positive FVC prediction: {min_fvc}"

    # Check row count matches test set size (in Debug mode, this is small)
    # The test set in data_dict['test'] corresponds to the processed rows.
    test_df_processed = data_dict["test"][0]
    assert len(sub_df) == len(
        test_df_processed
    ), f"Submission row count ({len(sub_df)}) does not match processed test set size ({len(test_df_processed)})"

    print("Modeling and Submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
