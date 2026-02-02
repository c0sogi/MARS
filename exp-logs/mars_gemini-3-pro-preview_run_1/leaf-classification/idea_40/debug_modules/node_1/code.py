import os
import sys
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import log_loss

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

# Import library modules
from library import config
from library import utils
from library import features
from library import preprocessing
from library import data
from library import model


def demo_config_and_utils():
    print("\n=== 1. Validating Config and Utils ===")

    # Check Config
    print(f"Seed: {config.SEED}")
    print(f"Working Dir: {config.WORKING_DIR}")
    print(f"EFD Harmonics: {config.EFD_HARMONICS}")

    # Check Utils: Seeding
    utils.set_seed(42)
    r1 = np.random.rand()
    utils.set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Random seed setting failed reproducibility check."
    print("Random seed reproducibility verified.")

    # Check Utils: Config Hash
    hash_val = utils.get_config_hash()
    assert (
        isinstance(hash_val, str) and len(hash_val) > 0
    ), "Config hash generation failed."
    print(f"Config Hash: {hash_val}")


def demo_feature_extraction():
    print("\n=== 2. Validating Feature Extraction ===")

    # Load metadata to get a real image path
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        print("Metadata not found, skipping feature extraction demo.")
        return

    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]
    rel_path = sample_row["file_path"]
    full_path = os.path.join(config.INPUT_DIR, rel_path)

    print(f"Processing single image: {rel_path}")

    # Test process_single_image
    extracted_features = features.process_single_image(full_path)

    # Validation
    assert isinstance(
        extracted_features, dict
    ), "Feature extraction should return a dictionary."

    # Check for Spatial features
    spatial_keys = [k for k in extracted_features.keys() if k.startswith("spatial_")]
    assert len(spatial_keys) == len(
        config.SPATIAL_FEATURES
    ), f"Expected {len(config.SPATIAL_FEATURES)} spatial features, found {len(spatial_keys)}."

    # Check for EFD features
    # 15 harmonics * 4 coeffs (a,b,c,d) = 60
    efd_keys = [k for k in extracted_features.keys() if k.startswith("efd_")]
    expected_efd = config.EFD_HARMONICS * 4
    assert (
        len(efd_keys) == expected_efd
    ), f"Expected {expected_efd} EFD features, found {len(efd_keys)}."

    # Check Precision
    first_val = list(extracted_features.values())[0]
    assert isinstance(
        first_val, (float, np.floating)
    ), "Feature values should be floats."

    print(
        f"Successfully extracted {len(extracted_features)} features from sample image."
    )


def demo_preprocessing_unit():
    print("\n=== 3. Validating Preprocessing Logic (Unit Test) ===")

    # Create dummy data with float64
    X_dummy = np.random.rand(20, 5).astype(np.float64)
    # Add some scale difference to test standardization
    X_dummy[:, 0] = X_dummy[:, 0] * 100

    preprocessor = preprocessing.HighPrecisionPreprocessor()

    # Fit
    preprocessor.fit(X_dummy)

    # Transform
    X_trans = preprocessor.transform(X_dummy)

    # Validate
    assert X_trans.shape == X_dummy.shape, "Transformed shape mismatch."
    assert X_trans.dtype == np.float64, "Precision lost during preprocessing."

    # Check standardization (mean approx 0, std approx 1)
    # Note: PowerTransformer changes distribution, then StandardScaler standardizes it.
    means = np.mean(X_trans, axis=0)
    stds = np.std(X_trans, axis=0)

    assert np.allclose(means, 0, atol=1e-7), "Standardization mean is not zero."
    assert np.allclose(stds, 1, atol=1e-7), "Standardization std is not one."

    print("HighPrecisionPreprocessor unit test passed.")


def demo_data_loading_and_integration():
    print("\n=== 4. Validating Data Loading (Integration) ===")

    # We use a small max_samples to speed up the loading/processing
    # Note: The library forces full feature extraction if cache is missing,
    # but filters afterwards.
    N_SAMPLES = 50

    print(f"Loading data with max_samples={N_SAMPLES}...")
    # Force reload to verify pipeline logic (load_cached_data=False)
    # Warning: This might take a moment as it processes features for the full set
    # before slicing, depending on implementation.
    # Based on library code: features.get_dataset processes all.
    # We will use load_cached_data=True to be faster if previous runs existed,
    # or it will create it.

    try:
        data_tuple = data.load_data(load_cached_data=True, max_samples=N_SAMPLES)
        X_train, y_train, X_val, y_val, X_test, test_ids, classes = data_tuple

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Classes: {classes}")

        # Validate Dimensions
        # Total features = 64*3 (margin, shape, texture) + 8 (spatial) + 60 (efd) = 260
        EXPECTED_FEATS = 260
        assert (
            X_train.shape[1] == EXPECTED_FEATS
        ), f"Expected {EXPECTED_FEATS} features, got {X_train.shape[1]}"
        assert (
            len(X_train) <= N_SAMPLES
        ), "Max samples limit not respected for training set."
        assert X_train.dtype == np.float64, "Data dtype incorrect."

        print("Data loading integration test passed.")
        return X_train, y_train, X_val, y_val, classes

    except Exception as e:
        print(f"Data loading failed: {e}")
        raise e


def demo_model_training(X_train, y_train, X_val, y_val, classes):
    print("\n=== 5. Validating Model Training & Prediction ===")

    clf = model.SpectralSpatialOAS()

    # Fit
    print("Fitting model...")
    clf.fit(X_train, y_train)

    # Predict Proba
    print("Predicting probabilities on validation set...")
    probs = clf.predict_proba(X_val)

    # Validate Probabilities
    assert probs.shape == (
        len(X_val),
        len(classes),
    ), "Probability matrix shape mismatch."
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities do not sum to 1."

    # Predict Labels
    preds = clf.predict(X_val)
    assert len(preds) == len(X_val), "Prediction length mismatch."

    # Calculate Score
    loss = log_loss(y_val, probs, labels=clf.classes_)
    print(f"Validation Log Loss (Subset): {loss:.4f}")

    print("Model validation passed.")


def demo_full_pipeline():
    print("\n=== 6. Executing Full Pipeline Wrapper ===")

    # This function in the library orchestrates everything:
    # Load -> Train -> Eval -> Submit
    # We use a small subset to ensure it finishes quickly.

    try:
        model.train_and_predict(load_cached_data=True, max_samples=60)

        # Check if submission file exists
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        assert os.path.exists(sub_path), "Submission file was not created."

        df_sub = pd.read_csv(sub_path)
        print(f"Submission file created with shape: {df_sub.shape}")

        # Basic submission check
        assert "id" in df_sub.columns, "Submission missing 'id' column."
        assert (
            df_sub.shape[1] == 100
        ), "Submission should have 100 columns (id + 99 species)."

        print("Full pipeline execution successful.")

    except Exception as e:
        print(f"Full pipeline execution failed: {e}")
        raise e


if __name__ == "__main__":
    print("Starting Library Demo...")

    # 1. Config & Utils
    demo_config_and_utils()

    # 2. Features
    demo_feature_extraction()

    # 3. Preprocessing
    demo_preprocessing_unit()

    # 4. Data Loading
    X_tr, y_tr, X_v, y_v, cls_names = demo_data_loading_and_integration()

    # 5. Model
    demo_model_training(X_tr, y_tr, X_v, y_v, cls_names)

    # 6. Full Pipeline
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
