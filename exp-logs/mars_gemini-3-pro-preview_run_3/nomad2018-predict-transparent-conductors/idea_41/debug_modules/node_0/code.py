import os
import sys
import numpy as np
import pandas as pd
import ase.io
import xgboost as xgb

# Import from the provided library
from library.config import INPUT_DIR, CACHE_DIR
from library.feature_engine import FingerprintGenerator
from library.data_handler import get_train_data
from library.preprocessor import (
    load_and_preprocess_data,
    TargetTransformer,
    FeatureCleaner,
)
from library.model_wrapper import XGBRegressorWrapper

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def demo_fingerprint_generator():
    print("\n=== Demo: FingerprintGenerator ===")

    # Path to a sample geometry file
    # Assuming standard structure, picking id=1 from train
    sample_geom_path = os.path.join(INPUT_DIR, "train", "1", "geometry.xyz")

    if not os.path.exists(sample_geom_path):
        # Fallback to finding first available file if specific ID missing
        for root, dirs, files in os.walk(os.path.join(INPUT_DIR, "train")):
            for f in files:
                if f.endswith(".xyz"):
                    sample_geom_path = os.path.join(root, f)
                    break
            if os.path.exists(sample_geom_path):
                break

    print(f"Loading atoms from: {sample_geom_path}")
    atoms = ase.io.read(sample_geom_path)

    # Instantiate generator
    generator = FingerprintGenerator()

    # Generate features
    features = generator.generate(atoms)

    # Verification
    print(f"Generated {len(features)} features.")

    # Check for specific expected keys based on logic
    expected_keys_subset = [
        "vol_per_atom",
        "density",
        "RDF_Ga_O_bin_0",
        "RDF_Al_O_bin_0",
        "local_Ga_BVS_p50",
        "topo_OMO_Angles_p50",
    ]

    for key in expected_keys_subset:
        assert key in features, f"Missing expected feature key: {key}"

    # Check values are not NaN (unless empty distribution, which might happen for rare pairs,
    # but density/vol should be real)
    assert not np.isnan(features["vol_per_atom"]), "Volume per atom is NaN"
    assert not np.isnan(features["density"]), "Density is NaN"

    print("FingerprintGenerator check passed.")


def demo_data_handler_and_preprocessor():
    print("\n=== Demo: Data Handler & Preprocessor ===")

    # Use a small sample size for speed
    sample_size = 50

    # 1. Test get_train_data (Data Handler)
    # This internally calls process_dataset -> FingerprintGenerator
    # We force load_cached_data=False to ensure the code runs,
    # though it might save to cache for subsequent calls.
    print(f"Loading training data (sample_size={sample_size})...")
    df_train_raw = get_train_data(load_cached_data=False, sample_size=sample_size)

    assert (
        len(df_train_raw) == sample_size
    ), f"Expected {sample_size} rows, got {len(df_train_raw)}"
    assert (
        "formation_energy_ev_natom" in df_train_raw.columns
    ), "Target column missing in raw train data"

    # 2. Test load_and_preprocess_data (Preprocessor)
    # This handles splitting (X, y) and cleaning
    print("Running load_and_preprocess_data...")
    (X_train, y_train), (X_val, y_val), (X_test, test_ids) = load_and_preprocess_data(
        load_cached_data=False, sample_size=sample_size
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Verification
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in Train X and y rows"
    assert X_val.shape[0] == y_val.shape[0], "Mismatch in Val X and y rows"
    # Ensure no object columns remain (cleaning step)
    assert X_train.select_dtypes(
        include=["object"]
    ).empty, "X_train contains non-numeric data"

    # 3. Test TargetTransformer
    print("Testing TargetTransformer...")
    transformer = TargetTransformer()
    original_val = np.array([0.0, 1.0, 10.0])
    transformed = transformer.transform(original_val)
    inverted = transformer.inverse_transform(transformed)

    assert np.allclose(original_val, inverted), "TargetTransformer inverse failed"
    # Check log1p logic: log(1+0)=0, log(1+1)=0.693...
    assert np.isclose(transformed[0], 0.0), "TargetTransformer log1p(0) incorrect"

    print("Data Handler & Preprocessor checks passed.")

    return X_train, y_train, X_val, y_val, X_test


def demo_model_wrapper(X_train, y_train, X_val, y_val, X_test):
    print("\n=== Demo: Model Wrapper (XGBoost) ===")

    wrapper = XGBRegressorWrapper()

    # Optimize for speed: Reduce n_estimators for this demo
    # Access internal models dictionary
    print("Reducing n_estimators for fast demonstration...")
    for target_name in wrapper.models:
        wrapper.models[target_name].set_params(n_estimators=5, max_depth=3)

    # Train
    print("Training models...")
    metrics = wrapper.train(X_train, y_train, X_val, y_val)

    # Verify metrics dictionary
    assert "formation_energy_ev_natom" in metrics, "Missing formation energy metric"
    assert "bandgap_energy_ev" in metrics, "Missing bandgap energy metric"
    print("Training metrics:", metrics)

    # Predict
    print("Running inference on test set...")
    preds = wrapper.predict(X_test)

    # Verification
    assert len(preds) == len(X_test), "Prediction length mismatch"
    assert (
        "formation_energy_ev_natom" in preds.columns
    ), "Missing formation energy prediction"
    assert "bandgap_energy_ev" in preds.columns, "Missing bandgap energy prediction"

    # Check values are reasonable (e.g., non-negative, though model can predict negative,
    # physical energies here are typically positive or near zero).
    # Since we use log1p transform internally, the inverse expm1 should be > -1.
    # Usually formation energy and bandgap are > 0.
    print("Sample predictions:\n", preds.head())

    print("Model Wrapper check passed.")


if __name__ == "__main__":
    # Ensure working directory exists for cache
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Feature Generation Demo
    demo_fingerprint_generator()

    # 2. Data Loading & Preprocessing Demo
    X_train, y_train, X_val, y_val, X_test = demo_data_handler_and_preprocessor()

    # 3. Model Training & Inference Demo
    demo_model_wrapper(X_train, y_train, X_val, y_val, X_test)

    print("\nAll demonstrations completed successfully.")
