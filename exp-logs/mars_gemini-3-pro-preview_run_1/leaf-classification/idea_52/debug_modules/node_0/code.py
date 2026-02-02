import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import library components
from library.config import Config
from library.utils import set_seed, custom_log_loss
from library.features import ImageFeatureExtractor
from library.preprocessing import HighPrecisionTransformer
from library.data import DataManager
from library.model import OASLinearDiscriminant


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup and Configuration
    print("\n1. Initializing Configuration and Seeds...")
    set_seed(Config.SEED)

    # Verify Config constants
    assert os.path.exists(Config.INPUT_DIR), "Input directory not found."
    assert os.path.exists(Config.METADATA_DIR), "Metadata directory not found."
    print(
        f"   Config verified. Input: {Config.INPUT_DIR}, Metadata: {Config.METADATA_DIR}"
    )

    # 2. Feature Extraction Demo
    print("\n2. Demonstrating Feature Extraction (ImageFeatureExtractor)...")
    extractor = ImageFeatureExtractor()
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")

    # Process a small subset (20 samples) to verify feature generation
    # This triggers geometric feature extraction and merging with tabular data
    print("   Extracting features for 20 samples...")
    X_raw, y_raw, ids_raw = extractor.process_data(
        train_meta_path,
        dataset_name="demo_train_features",
        load_cached_data=False,  # Force computation for demo
        max_samples=20,
    )

    # Expected features: 192 tabular (3 groups * 64) + 10 geometric
    expected_features = 192 + 10
    print(f"   Extracted Shape: {X_raw.shape}")

    assert X_raw.shape == (
        20,
        expected_features,
    ), f"Expected shape (20, {expected_features}), got {X_raw.shape}"
    assert len(y_raw) == 20, "Target length mismatch."
    assert len(ids_raw) == 20, "ID length mismatch."

    # Test specific geometric feature calculation
    # Get the first file path from metadata to test single-image extraction
    df_meta = pd.read_csv(train_meta_path)
    sample_image_path = os.path.join(Config.INPUT_DIR, df_meta.iloc[0]["file_path"])
    thickness = extractor.compute_integral_thickness(sample_image_path)
    print(f"   Sample Integral Thickness: {thickness:.6f}")
    assert isinstance(thickness, float), "Thickness should be a float."

    # 3. Preprocessing Demo
    print("\n3. Demonstrating Preprocessing (HighPrecisionTransformer)...")
    transformer = HighPrecisionTransformer()

    # Fit and transform the raw data extracted above
    print("   Fitting and transforming data...")
    X_trans = transformer.fit_transform(X_raw)

    # Validation
    assert X_trans.dtype == Config.FLOAT_PRECISION, "Transformed data must be float64."
    assert (
        X_trans.shape == X_raw.shape
    ), "Shape should be preserved during transformation."

    # Check Standard Scaling properties (Mean ~ 0, Std ~ 1)
    # Note: With n=20, stats might fluctuate slightly, but should be close.
    mean_val = np.mean(X_trans)
    std_val = np.std(X_trans)
    print(f"   Transformed Data Mean: {mean_val:.6f} (Expected ~0)")
    print(f"   Transformed Data Std:  {std_val:.6f} (Expected ~1)")

    assert np.abs(mean_val) < 1e-6, "Data not centered correctly."
    # We allow a small margin for std dev due to degrees of freedom in small samples
    assert np.abs(std_val - 1.0) < 0.1, "Data not scaled correctly."

    # 4. Data Manager Demo
    print("\n4. Demonstrating Data Manager (Loading All Data)...")
    dm = DataManager()

    # Load all splits with a limit of 50 samples each for speed
    # This handles extraction + preprocessing internally
    print("   Loading train, val, and test sets (max_samples=50)...")
    (train_data, val_data, test_data) = dm.load_all_data(
        load_cached_data=False, max_samples=50
    )

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    print(f"   Train Shape: {X_train.shape}")
    print(f"   Val Shape:   {X_val.shape}")
    print(f"   Test Shape:  {X_test.shape}")

    assert X_train.shape[0] == 50, "Train sample count mismatch."
    assert X_val.shape[0] == 50, "Val sample count mismatch."
    assert X_test.shape[0] == 50, "Test sample count mismatch."

    # 5. Model Demo
    print("\n5. Demonstrating Model (OASLinearDiscriminant)...")
    model = OASLinearDiscriminant()

    # Fit model on the training subset
    print("   Fitting model on training subset...")
    model.fit(X_train, y_train)

    # Check learned attributes
    n_classes_learned = len(model.classes_)
    print(f"   Model learned {n_classes_learned} classes from the subset.")

    # Predict probabilities on the training set (to ensure class consistency for demo)
    # In a real scenario, we predict on Val/Test, but for this demo, we want to guarantee
    # that the rows correspond to known classes to avoid LabelEncoder errors if Val has
    # unseen classes due to the small random subset.
    print("   Predicting probabilities...")
    y_pred_proba = model.predict_proba(X_train)

    assert y_pred_proba.shape == (
        50,
        n_classes_learned,
    ), f"Prediction shape mismatch. Expected (50, {n_classes_learned}), got {y_pred_proba.shape}"

    # Verify probabilities sum to 1
    row_sums = np.sum(y_pred_proba, axis=1)
    print(f"   Mean Row Sum: {np.mean(row_sums):.6f}")
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."

    # 6. Metric Demo
    print("\n6. Demonstrating Metric (Custom Log Loss)...")

    # For log loss, we need integer encoded labels matching the model's classes
    # or one-hot. The custom_log_loss wrapper handles standard inputs.
    # We use the y_train and y_pred_proba from the previous step.

    # First, we need to ensure y_train is encoded to match the columns of y_pred_proba
    # The custom_log_loss function uses sklearn.metrics.log_loss internally.
    # We pass the raw string labels (y_train) and the probabilities.
    # However, sklearn log_loss requires the columns of proba to be strictly ordered
    # and all classes in y_true to be present in model.classes_.

    # Since we predicted on X_train, y_train is guaranteed to be a subset of model.classes_.
    loss = custom_log_loss(y_train, y_pred_proba)
    print(f"   Calculated Log Loss: {loss:.6f}")

    assert loss >= 0, "Log loss cannot be negative."

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
