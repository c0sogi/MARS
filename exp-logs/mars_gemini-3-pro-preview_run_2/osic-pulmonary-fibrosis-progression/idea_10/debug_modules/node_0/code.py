import os
import numpy as np
import pandas as pd
import torch
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dicom_loader import load_scan, read_dicom_slice
from library.image_processing import (
    get_patient_zones,
    select_variance_slice,
    compute_density_hist,
    preprocess_image,
)
from library.feature_extractor import FeatureExtractor
from library.data_pipeline import DataPipeline
from library.model_factory import LaplaceSolver, train_laplace_solver
from library.workflow import run_workflow

# Setup
warnings.filterwarnings("ignore")
seed_everything(Config.SEED)


def create_mini_metadata(n_patients=2):
    """
    Creates smaller metadata files to ensure the demonstration runs quickly.
    """
    print(f"\n[Demo] Creating mini metadata files with {n_patients} patients...")

    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select subset of unique patients
    train_patients = train_df["Patient"].unique()[:n_patients]
    val_patients = val_df["Patient"].unique()[:n_patients]
    test_patients = test_df["Patient"].unique()[:n_patients]

    # Filter DataFrames
    mini_train = train_df[train_df["Patient"].isin(train_patients)]
    mini_val = val_df[val_df["Patient"].isin(val_patients)]
    mini_test = test_df[test_df["Patient"].isin(test_patients)]

    # Define paths for mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train_metadata.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val_metadata.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")

    # Save
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"[Demo] Mini metadata saved to {Config.WORKING_DIR}")
    return mini_train_path, mini_val_path, mini_test_path


def test_dicom_and_image_processing(metadata_path):
    """
    Verifies DICOM loading and image processing logic.
    """
    print("\n[Demo] Testing DICOM Loader and Image Processing...")

    df = pd.read_csv(metadata_path)
    # Get first patient info
    patient_id = df.iloc[0]["Patient"]
    rel_path = df.iloc[0]["dcm_path"]
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    # 1. Test load_scan
    print(f"Loading scan for patient: {patient_id}")
    volume = load_scan(full_path, load_cached_data=False)

    # Assertions
    assert isinstance(volume, np.ndarray), "Volume should be a numpy array"
    assert volume.ndim == 3, f"Volume should be 3D, got {volume.ndim}"
    # CT slices are typically 512x512
    assert (
        volume.shape[1] == 512 and volume.shape[2] == 512
    ), "Slice dims should be 512x512"
    print(f"Volume shape verified: {volume.shape}")

    # 2. Test get_patient_zones
    zones = get_patient_zones(volume)
    assert len(zones) == Config.N_ZONES, f"Should have {Config.N_ZONES} zones"
    print(f"Zones split verified: {len(zones)} zones")

    # 3. Test select_variance_slice and preprocess_image
    # Use the middle zone
    mid_zone = zones[1]
    if mid_zone.shape[0] > 0:
        best_slice = select_variance_slice(mid_zone)
        assert best_slice.shape == (512, 512), "Best slice should be 512x512"

        processed_img = preprocess_image(best_slice)
        # Expected shape: (3, Config.IMG_SIZE, Config.IMG_SIZE) -> (3, 224, 224)
        expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
        assert (
            processed_img.shape == expected_shape
        ), f"Processed image shape mismatch: {processed_img.shape}"
        assert processed_img.dtype == np.float32, "Processed image should be float32"
        print(f"Image preprocessing verified: {processed_img.shape}")

        # 4. Test compute_density_hist
        hist = compute_density_hist(mid_zone)
        assert hist.shape == (Config.DENSITY_BINS,), "Histogram shape mismatch"
        assert np.isclose(hist.sum(), 1.0) or np.isclose(
            hist.sum(), 0.0
        ), "Histogram should sum to 1 (or 0 if empty)"
        print("Density histogram verified.")
    else:
        print("Warning: Middle zone was empty, skipping specific slice tests.")


def test_feature_extractor(metadata_path):
    """
    Verifies the FeatureExtractor class.
    """
    print("\n[Demo] Testing Feature Extractor...")

    df = pd.read_csv(metadata_path)
    patient_id = df.iloc[0]["Patient"]
    rel_path = df.iloc[0]["dcm_path"]

    # Initialize extractor
    extractor = FeatureExtractor(
        device=torch.device("cpu")
    )  # Force CPU for demo stability

    # Extract features
    print(f"Extracting features for {patient_id}...")
    features = extractor.extract_patient_features(patient_id, rel_path)

    # Expected dimension calculation:
    # 3 zones * 1280 (EfficientNetB0) + 3 zones * 4 (Histogram) = 3840 + 12 = 3852
    expected_dim = 3 * 1280 + 3 * 4

    assert isinstance(features, np.ndarray), "Features should be numpy array"
    assert features.shape == (
        expected_dim,
    ), f"Feature vector shape mismatch. Expected ({expected_dim},), got {features.shape}"
    print(f"Feature extraction verified. Vector shape: {features.shape}")


def test_full_workflow(train_path, val_path, test_path):
    """
    Runs the full workflow using the mini metadata files.
    Monkey-patches the Config paths to point to the mini files.
    """
    print("\n[Demo] Running Full Workflow with Mini Dataset...")

    # Monkey-patch Config to use mini metadata
    # This affects the library modules since they import the same Config object
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    # Run workflow
    # load_cached_data=False ensures we actually run the processing logic
    model = run_workflow(load_cached_data=False, debug=False)

    # Verify Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), "Submission columns mismatch"

    # Check values
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"
    assert (sub_df["Confidence"] >= 0).all(), "Confidence values must be non-negative"

    print("Workflow verification successful.")


if __name__ == "__main__":
    # 1. Create Mini Metadata for Speed
    mini_train, mini_val, mini_test = create_mini_metadata(n_patients=2)

    # 2. Test Low-Level Components
    test_dicom_and_image_processing(mini_train)

    # 3. Test Feature Extraction
    test_feature_extractor(mini_train)

    # 4. Test High-Level Workflow
    test_full_workflow(mini_train, mini_val, mini_test)

    print("\n[Demo] All tests passed successfully.")
