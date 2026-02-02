import os
import sys
import numpy as np
import pandas as pd
import shutil
from library.config import Config
import library.feature_extraction as fe
import library.data_manager as dm
from library.preprocessor import HighPrecisionPipeline
from library.model import IntegralInertialDiscriminant

# Ensure reproducibility
np.random.seed(42)


def clean_debug_cache():
    """Removes temporary debug cache files created during the demo."""
    if os.path.exists(Config.CACHE_DIR):
        for f in os.listdir(Config.CACHE_DIR):
            if "debug" in f:
                os.remove(os.path.join(Config.CACHE_DIR, f))
        # Try to remove the directory if empty, but ignore if not
        try:
            os.rmdir(Config.CACHE_DIR)
        except OSError:
            pass


def demo_config_and_setup():
    print("1. Verifying Configuration...")
    # Check if input directory exists
    assert os.path.exists(Config.INPUT_DIR), "Input directory not found."
    # Check if metadata exists
    assert os.path.exists(Config.TRAIN_METADATA_PATH), "Train metadata not found."

    print(f"   Input Dir: {Config.INPUT_DIR}")
    print(f"   Float Precision: {Config.FLOAT_PRECISION}")
    print("   Configuration verified.\n")


def demo_feature_extraction():
    print("2. Demonstrating Feature Extraction (Single Image)...")

    # Load metadata to get a valid image path
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]
    image_rel_path = sample_row["file_path"]
    full_path = os.path.join(Config.INPUT_DIR, image_rel_path)

    print(f"   Extracting features from: {image_rel_path}")

    # Call the library function
    features = fe.extract_inertial_features(full_path)

    # Validation
    assert isinstance(features, dict), "Features should be returned as a dictionary."
    assert "Area" in features, "Feature 'Area' missing."
    assert "Inertial_Major_Axis" in features, "Feature 'Inertial_Major_Axis' missing."
    assert features["Area"] >= 0, "Area cannot be negative."

    print(f"   Extracted {len(features)} geometric features.")
    print(f"   Sample Area: {features['Area']:.4f}")
    print("   Feature extraction verified.\n")


def demo_data_manager():
    print("3. Demonstrating Data Manager (Parallel Loading & Merging)...")

    subset_size = 20
    print(f"   Loading subset of {subset_size} samples using DataManager...")

    # Use data_manager to load a subset (this triggers parallel feature extraction)
    # This creates a cache file named 'train_data_debug_20.parquet'
    df_subset = dm.get_train_data(load_cached_data=False, sample_size=subset_size)

    # Validation
    assert (
        len(df_subset) == subset_size
    ), f"Expected {subset_size} rows, got {len(df_subset)}."
    assert "species" in df_subset.columns, "Target column 'species' missing."
    assert "id" in df_subset.columns, "ID column missing."

    # Check if extracted features are merged
    # 'Area' is an extracted feature, 'margin_1' is a raw feature
    assert "Area" in df_subset.columns, "Extracted features not merged correctly."
    assert "margin_1" in df_subset.columns, "Raw features not preserved."

    print("   Data loaded and merged successfully.")
    print(f"   DataFrame Shape: {df_subset.shape}")
    print("   Data Manager verified.\n")

    return df_subset


def demo_preprocessing(df):
    print("4. Demonstrating HighPrecisionPipeline...")

    # Prepare X and y
    # Drop non-feature columns
    drop_cols = ["id", "species"]
    # Also drop excluded features defined in Config
    drop_cols += [c for c in Config.EXCLUDED_FEATURES if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["species"].values

    print(f"   Input Feature Shape: {X.shape}")

    # Instantiate Pipeline
    pipeline = HighPrecisionPipeline()

    # Fit and Transform
    X_trans = pipeline.fit_transform(X)

    # Validation
    assert isinstance(X_trans, np.ndarray), "Output must be a numpy array."
    assert (
        X_trans.dtype == Config.FLOAT_PRECISION
    ), f"Output dtype must be {Config.FLOAT_PRECISION}."
    assert X_trans.shape == X.shape, "Output shape mismatch."
    assert not np.isnan(X_trans).any(), "Output contains NaNs."

    # Check statistics (StandardScaler should make mean ~0 and std ~1)
    # Note: PowerTransformer is applied first, so exact 0/1 depends on distribution,
    # but StandardScaler is the last step.
    means = np.mean(X_trans, axis=0)
    stds = np.std(X_trans, axis=0)

    # Allow some tolerance for numerical noise
    assert np.allclose(means, 0, atol=1e-6), "Features not centered."
    assert np.allclose(stds, 1, atol=1e-6), "Features not scaled."

    print("   Preprocessing successful (Yeo-Johnson + Scaling).")
    print("   Pipeline verified.\n")

    return X_trans, y


def demo_model(X, y):
    print("5. Demonstrating IntegralInertialDiscriminant Model...")

    # Instantiate Model
    model = IntegralInertialDiscriminant()

    # Fit Model
    print("   Fitting model...")
    model.fit(X, y)

    assert model.classes_ is not None, "Classes not learned."
    assert model.W is not None, "Weights W not computed."
    assert model.b is not None, "Bias b not computed."

    # Predict Probabilities
    print("   Predicting probabilities...")
    probs = model.predict_proba(X)

    # Validation
    assert probs.shape == (len(X), len(model.classes_)), "Probability shape mismatch."

    # Check probability properties
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities do not sum to 1."
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]."

    # Check if precision matrix is float64
    assert (
        model.precision_.dtype == Config.FLOAT_PRECISION
    ), "Precision matrix is not high precision."

    print(f"   Prediction Shape: {probs.shape}")
    print("   Model verified.\n")


if __name__ == "__main__":
    try:
        # Clean up any previous debug runs
        clean_debug_cache()

        # 1. Config
        demo_config_and_setup()

        # 2. Feature Extraction
        demo_feature_extraction()

        # 3. Data Manager (Get Subset)
        df_subset = demo_data_manager()

        # 4. Preprocessing
        X_processed, y_subset = demo_preprocessing(df_subset)

        # 5. Model
        demo_model(X_processed, y_subset)

        print("=" * 40)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 40)

    except Exception as e:
        print(f"\nFAILED: {e}")
        # Clean up on failure as well
        clean_debug_cache()
        sys.exit(1)
    finally:
        # Final cleanup
        clean_debug_cache()
