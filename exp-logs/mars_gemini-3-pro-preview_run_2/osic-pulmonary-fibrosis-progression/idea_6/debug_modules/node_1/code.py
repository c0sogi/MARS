import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dicom_handler import DicomHandler
from library.feature_extractor import FeatureExtractor
from library.data_processor import DataProcessor
from library.modeling import DualMomentGLM

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demo...")

    # Enable Debug mode to use a small subset of data (5 patients)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5

    # Use a temporary directory for this demo to avoid loading existing large cache files
    # and to ensure we demonstrate the actual processing logic.
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create these new directories
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Cache Dir: {Config.CACHE_DIR}")
    print("   Configuration complete.\n")

    # -------------------------------------------------------------------------
    # 2. Verify Metric Calculation
    # -------------------------------------------------------------------------
    print("2. Verifying Metric Calculation...")
    # Test case: Perfect prediction
    # FVC_true = 2000, FVC_pred = 2000, Sigma = 100
    # Delta = 0
    # Sigma_clipped = max(100, 70) = 100
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    score_perfect = calculate_metric([2000], [2000], [100])
    expected_score = -np.log(np.sqrt(2) * 100)

    assert np.isclose(
        score_perfect, expected_score, atol=1e-4
    ), f"Metric mismatch! Got {score_perfect}, expected {expected_score}"

    print(f"   Perfect prediction score: {score_perfect:.4f}")
    print("   Metric verification passed.\n")

    # -------------------------------------------------------------------------
    # 3. Demonstrate DICOM Handling
    # -------------------------------------------------------------------------
    print("3. Demonstrating DicomHandler...")
    # Get a sample patient ID from the training metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    sample_patient_id = train_meta["Patient"].iloc[0]
    print(f"   Processing patient: {sample_patient_id}")

    # Process patient (Load -> Select Slices -> Preprocess)
    # This returns a numpy array of shape (N_SLICES, 3, IMG_SIZE, IMG_SIZE)
    img_tensor = DicomHandler.process_patient(
        sample_patient_id, subset="train", load_cached_data=False
    )

    print(f"   Output Tensor Shape: {img_tensor.shape}")

    # Assertions
    expected_shape = (Config.N_SLICES, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Shape mismatch! Expected {expected_shape}, got {img_tensor.shape}"
    assert img_tensor.dtype == np.float32, "Data type should be float32"

    # Check normalization (values should not be raw HU like -1000)
    # Preprocessing normalizes to roughly N(0,1) via ImageNet stats, so values typically range -3 to 3
    assert (
        img_tensor.max() < 100 and img_tensor.min() > -100
    ), "Image values appear to be out of normalized range."

    print("   DicomHandler verification passed.\n")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Feature Extraction
    # -------------------------------------------------------------------------
    print("4. Demonstrating FeatureExtractor...")
    extractor = FeatureExtractor()

    # Process the 'train' subset. Due to DEBUG=True, this processes only 5 patients.
    # We set load_cached_data=False to force execution.
    features, ids = extractor.process_dataset("train", load_cached_data=False)

    print(f"   Extracted Features Shape: {features.shape}")
    print(f"   Patient IDs Shape: {ids.shape}")

    # Assertions
    # EfficientNet-B0 features (1280) * 2 (Mean + Std) = 2560
    assert features.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        2560,
    ), f"Feature shape mismatch! Expected ({Config.DEBUG_SAMPLE_SIZE}, 2560), got {features.shape}"
    assert len(ids) == Config.DEBUG_SAMPLE_SIZE, "ID count mismatch"

    print("   FeatureExtractor verification passed.\n")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Data Processor
    # -------------------------------------------------------------------------
    print("5. Demonstrating DataProcessor...")
    processor = DataProcessor()

    # Run the full pipeline:
    # 1. Extract features for Train/Val/Test (using debug size)
    # 2. PCA
    # 3. Tabular Merge
    # 4. Scaling
    data_dict = processor.process(load_cached_data=False)

    print("   Keys in processed data dictionary:", list(data_dict.keys()))

    # Assertions
    required_keys = [
        "X_train_fvc",
        "X_train_unc",
        "y_train",
        "X_val_fvc",
        "X_val_unc",
        "y_val",
        "X_test_fvc",
        "X_test_unc",
        "test_ids",
    ]
    for key in required_keys:
        assert key in data_dict, f"Missing key in data dictionary: {key}"
        assert isinstance(data_dict[key], np.ndarray), f"{key} is not a numpy array"

    # Check X_train_fvc dimensions
    # Rows should equal total samples in the debug subset (patients * visits)
    # Since we only took 5 patients, and each has multiple visits, exact rows vary,
    # but columns should be fixed based on logic:
    # Static(8) + PCA(40) + Relative_Weeks(1) + Interactions(48) = ~97 columns
    # Let's just verify it's not empty.
    X_train = data_dict["X_train_fvc"]
    print(f"   X_train_fvc shape: {X_train.shape}")
    assert X_train.shape[0] > 0, "X_train_fvc is empty"
    assert X_train.shape[1] > Config.PCA_COMPONENTS, "Feature count seems too low"

    print("   DataProcessor verification passed.\n")

    # -------------------------------------------------------------------------
    # 6. Demonstrate Modeling (DualMomentGLM)
    # -------------------------------------------------------------------------
    print("6. Demonstrating DualMomentGLM (Training & Inference)...")
    model = DualMomentGLM()

    # Train
    # Note: We pass the data_dict we just generated.
    # In a real run, model.run() calls processor.process(), but we can call train/predict manually
    # to use the data we already verified.
    model.train(data_dict)
    print("   Training complete.")

    # Predict on Test
    model.predict_test(data_dict)
    print("   Inference complete.")

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission file loaded. Rows: {len(sub_df)}")
    print(f"   Columns: {list(sub_df.columns)}")

    # Check format
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    # Check values
    assert sub_df["Confidence"].min() >= 0, "Confidence values must be non-negative"

    print("   Modeling and Submission verification passed.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
