import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging

# =============================================================================
# 1. Configuration Patching
# =============================================================================
# We import config first and patch it so that subsequent imports of library modules
# pick up the modified values (e.g. for speed and directory redirection).
import library.config as config

# Setup demo directory
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

print(f"Configuration: Redirecting outputs to {DEMO_DIR}")
print("Configuration: Patching parameters for fast execution...")

# Patch Constants
config.DEBUG_SAMPLE_SIZE = 2000  # Process only 2000 events for feature gen
config.N_ESTIMATORS = 10  # Very few trees for demo
config.EARLY_STOPPING_ROUNDS = 5
config.IDEA_DIR = DEMO_DIR

# Patch Paths
config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.parquet")
config.MODEL_X_PATH = os.path.join(DEMO_DIR, "lgbm_model_x.txt")
config.MODEL_Y_PATH = os.path.join(DEMO_DIR, "lgbm_model_y.txt")
config.MODEL_Z_PATH = os.path.join(DEMO_DIR, "lgbm_model_z.txt")
config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

# Patch Mutable Params
config.LGBM_PARAMS["n_estimators"] = 10
config.LGBM_PARAMS["verbosity"] = -1

# =============================================================================
# 2. Import Library Modules
# =============================================================================
# Now that config is patched, we import the rest.
from library.utils import spherical_to_cartesian, cartesian_to_spherical
from library.data_loader import IceCubeFeatureGenerator
from library.model import GradientBoostingVectorRegressor
from library.inference import generate_submission
from library.config import FEATURE_NAMES

# Silence LightGBM and other logs for the demo
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("FeatureGenerator").setLevel(logging.WARNING)
logging.getLogger("GBVR").setLevel(logging.WARNING)
logging.getLogger("Inference").setLevel(logging.WARNING)


def test_utils():
    print("\n[1] Verifying Utility Functions...")

    # Test 1: Z-axis (Zenith=0)
    # Cartesian (0, 0, 1) -> Spherical (Azimuth=Any, Zenith=0)
    az, zen = cartesian_to_spherical(0, 0, 1)
    assert np.isclose(zen, 0.0, atol=1e-6), f"Expected Zenith 0, got {zen}"

    # Test 2: X-axis (Azimuth=0, Zenith=pi/2)
    # Cartesian (1, 0, 0)
    az, zen = cartesian_to_spherical(1, 0, 0)
    assert np.isclose(zen, np.pi / 2, atol=1e-6), f"Expected Zenith pi/2, got {zen}"
    assert np.isclose(az, 0.0, atol=1e-6), f"Expected Azimuth 0, got {az}"

    # Test 3: Round Trip
    az_in, zen_in = np.array([1.0, 2.0]), np.array([0.5, 1.5])
    x, y, z = spherical_to_cartesian(az_in, zen_in)
    az_out, zen_out = cartesian_to_spherical(x, y, z)

    np.testing.assert_allclose(
        az_in, az_out, rtol=1e-5, err_msg="Azimuth round-trip failed"
    )
    np.testing.assert_allclose(
        zen_in, zen_out, rtol=1e-5, err_msg="Zenith round-trip failed"
    )

    print("    Utils verification passed.")


def run_feature_generation():
    print("\n[2] Running Feature Generation (Train Split)...")

    gen = IceCubeFeatureGenerator()

    # We force re-computation (load_cached_data=False) to test the logic
    # The patched DEBUG_SAMPLE_SIZE=2000 ensures this is fast
    train_df = gen.process_split(
        meta_path=config.TRAIN_META_PATH,
        output_path=config.TRAIN_FEATURES_PATH,
        load_cached_data=False,
    )

    # Assertions
    assert not train_df.empty, "Feature generation returned empty DataFrame"
    assert "target_x" in train_df.columns, "Targets (x) missing from training features"
    assert "target_y" in train_df.columns, "Targets (y) missing from training features"
    assert "target_z" in train_df.columns, "Targets (z) missing from training features"

    # Check if features exist
    for feat in FEATURE_NAMES:
        assert feat in train_df.columns, f"Feature {feat} missing"

    print(f"    Generated {len(train_df)} events with {len(train_df.columns)} columns.")
    return train_df


def run_model_training(train_df):
    print("\n[3] Running Model Training...")

    # Split for demo purposes
    # We just split the small train_df we generated
    val_size = int(len(train_df) * 0.2)
    train_subset = train_df.iloc[:-val_size]
    val_subset = train_df.iloc[-val_size:]

    X_train = train_subset[FEATURE_NAMES]
    y_train = train_subset  # Contains targets
    X_val = val_subset[FEATURE_NAMES]
    y_val = val_subset

    model = GradientBoostingVectorRegressor()

    # Manually update model paths to demo dir (since __init__ might have run before patch or uses defaults)
    # However, we patched config before import, so defaults in __init__ should be correct if they use config.CONST
    # Let's verify paths just in case
    assert (
        model.model_paths["x"] == config.MODEL_X_PATH
    ), "Model path not updated correctly"

    metrics = model.fit(X_train, y_train, X_val, y_val)

    print("    Training Metrics:", metrics)
    assert "x" in metrics and "y" in metrics and "z" in metrics

    # Verify artifacts exist
    assert os.path.exists(config.MODEL_X_PATH), "Model X artifact not saved"

    # Test Prediction
    print("    Testing in-memory prediction...")
    preds = model.predict(X_val)

    assert "azimuth" in preds.columns
    assert "zenith" in preds.columns
    assert (
        preds["azimuth"].between(0, 2 * np.pi).all()
    ), "Azimuth predictions out of range"
    assert preds["zenith"].between(0, np.pi).all(), "Zenith predictions out of range"

    print("    Model training and prediction verification passed.")
    return model


def run_inference_pipeline():
    print("\n[4] Running Inference Pipeline (Test Set)...")

    # We use the generate_submission function
    # We limit to 1 batch for speed using debug_sample_batches
    generate_submission(
        test_meta_path=config.TEST_META_PATH,
        output_path=config.SUBMISSION_PATH,
        debug_sample_batches=1,
    )

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    # Verify content
    sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(sub)} rows.")

    assert list(sub.columns) == [
        "event_id",
        "azimuth",
        "zenith",
    ], "Incorrect submission columns"
    assert not sub.isnull().values.any(), "Submission contains NaNs"

    print("    Inference pipeline verification passed.")


if __name__ == "__main__":
    # Set seeds for reproducibility of the demo script itself
    np.random.seed(42)

    try:
        test_utils()

        # Run pipeline
        df_train = run_feature_generation()
        model = run_model_training(df_train)
        run_inference_pipeline()

        print("\n=== Demo Completed Successfully ===")
        print(f"Outputs located in: {DEMO_DIR}")

    except AssertionError as e:
        print(f"\n!!! Verification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
