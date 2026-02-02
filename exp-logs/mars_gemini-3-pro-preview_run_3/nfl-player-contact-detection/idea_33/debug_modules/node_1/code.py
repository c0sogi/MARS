import os
import sys
import pandas as pd
import numpy as np
import shutil
import joblib

# Import library modules
from library.config import Config
from library.data_loader import load_metadata, load_tracking, load_helmets
import library.feature_engineering
from library.feature_engineering import process_data
from library.model import ContactModel
from library.pipeline import train_pipeline, inference_pipeline
from library.utils import setup_seed


def main():
    print("Starting Library Usage Demonstration...")
    setup_seed(Config.SEED)

    # =========================================================================
    # 1. Configuration & Monkeypatching for Speed
    # =========================================================================
    print("\n[1] Adjusting Configuration for Demo Speed...")

    # Reduce XGBoost estimators for extremely fast training
    Config.STREAM_A_PARAMS["n_estimators"] = 10
    Config.STREAM_A_PARAMS["early_stopping_rounds"] = 5
    Config.STREAM_B_PARAMS["n_estimators"] = 10
    Config.STREAM_B_PARAMS["early_stopping_rounds"] = 5

    # Modify cache directory to a temp location to avoid conflicts and verify creation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update cache file paths in Config to point to new working dir
    # We iterate over keys to update paths dynamically
    for key in Config.CACHE_FILES:
        filename = os.path.basename(Config.CACHE_FILES[key])
        Config.CACHE_FILES[key] = os.path.join(Config.WORKING_DIR, filename)

    # Monkeypatch library.feature_engineering.load_metadata to force small data sampling
    # This ensures FeatureEngine uses a small subset of data
    original_load_metadata = library.feature_engineering.load_metadata

    def mocked_load_metadata(split_name, sample_ratio=1.0):
        # Force 2% sampling regardless of call, to keep runtime very low
        print(
            f"   -> Mocked load_metadata called for {split_name} with forced 2% sampling."
        )
        return original_load_metadata(split_name, sample_ratio=0.02)

    library.feature_engineering.load_metadata = mocked_load_metadata
    print("   -> Configuration adjusted and data loader monkeypatched.")

    # =========================================================================
    # 2. Data Loader Verification
    # =========================================================================
    print("\n[2] Verifying Data Loader...")

    # Test Metadata Loading (using original function to test schema check)
    df_meta = original_load_metadata("train", sample_ratio=0.01)
    assert not df_meta.empty, "Metadata DataFrame should not be empty"
    assert "contact_id" in df_meta.columns, "Metadata missing contact_id"
    print(f"   -> Metadata loaded successfully. Shape: {df_meta.shape}")

    # Test Tracking Loading
    # We load a small chunk or just verify the function works.
    # load_tracking loads the whole file, which might be large, but we have 220GB RAM.
    # We'll just load it once to verify schema.
    print("   -> Loading tracking data (this may take a moment)...")
    df_tracking = load_tracking("train")
    assert not df_tracking.empty, "Tracking DataFrame should not be empty"
    required_track_cols = ["x_position", "y_position", "speed", "nfl_player_id"]
    for col in required_track_cols:
        assert col in df_tracking.columns, f"Tracking missing {col}"
    print(f"   -> Tracking data loaded. Shape: {df_tracking.shape}")

    # Test Helmets Loading
    print("   -> Loading helmets data...")
    df_helmets = load_helmets("train")
    assert not df_helmets.empty, "Helmets DataFrame should not be empty"
    assert "left" in df_helmets.columns, "Helmets missing bounding box columns"
    print(f"   -> Helmets data loaded. Shape: {df_helmets.shape}")

    # =========================================================================
    # 3. Feature Engineering Verification
    # =========================================================================
    print("\n[3] Verifying Feature Engineering...")

    # We use process_data with load_cached_data=False to force computation using our mocked data loader
    # This generates features for the small subset
    print("   -> Generating features for Train split (Stream A & B)...")
    X_a, ids_a, y_a, X_b, ids_b, y_b = process_data("train", load_cached_data=False)

    # Assertions
    assert isinstance(X_a, pd.DataFrame), "X_a should be a DataFrame"
    assert len(X_a) == len(y_a), "X_a and y_a length mismatch"
    assert len(X_a) == len(ids_a), "X_a and ids_a length mismatch"

    # Check if features were actually generated (columns > 0)
    assert X_a.shape[1] > 0, "Stream A features empty"

    # If we have ground contacts in the sample, check Stream B
    if len(X_b) > 0:
        assert len(X_b) == len(y_b), "X_b and y_b length mismatch"
        print(f"   -> Stream B samples found: {len(X_b)}")
    else:
        print("   -> No Stream B samples in this random subset (acceptable).")

    print(f"   -> Generated Stream A samples: {len(X_a)}")
    print("   -> Feature Engineering verified.")

    # =========================================================================
    # 4. Model Training Verification
    # =========================================================================
    print("\n[4] Verifying Model Training...")

    # Initialize model with Stream A params
    model = ContactModel(Config.STREAM_A_PARAMS)

    # Fit on the small feature set generated above
    print("   -> Fitting model on generated features...")
    model.fit(X_a, y_a, verbose=False)

    # Predict
    preds = model.predict(X_a)
    assert len(preds) == len(X_a), "Prediction length mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Save/Load check
    model_path = os.path.join(Config.WORKING_DIR, "test_model.joblib")
    model.save(model_path)
    assert os.path.exists(model_path), "Model file not saved"

    loaded_model = ContactModel.load(model_path)
    preds_loaded = loaded_model.predict(X_a)
    assert np.allclose(
        preds, preds_loaded
    ), "Loaded model predictions differ from original"
    print("   -> Model training, persistence, and prediction verified.")

    # =========================================================================
    # 5. Pipeline Verification
    # =========================================================================
    print("\n[5] Verifying Full Pipeline...")

    # A. Train Pipeline
    # This will re-run feature generation (hitting cache or re-computing) and train both streams
    # It saves models to Config.WORKING_DIR
    print("   -> Running Train Pipeline...")
    train_pipeline(load_cached_data=True)

    # Check artifacts
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "model_stream_a.joblib")
    ), "Stream A model missing"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "model_stream_b.joblib")
    ), "Stream B model missing"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "thresholds.joblib")
    ), "Thresholds file missing"

    # B. Inference Pipeline
    # This loads test data (mocked to be small), loads models, predicts, and creates submission
    print("   -> Running Inference Pipeline...")
    inference_pipeline(load_cached_data=True)

    # Check submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    df_sub = pd.read_csv(submission_path)
    assert "contact_id" in df_sub.columns, "Submission missing contact_id"
    assert "contact" in df_sub.columns, "Submission missing contact"
    assert df_sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values"

    # Verify submission length matches sample submission (which is large)
    # Note: Our inference pipeline merges predictions onto the sample submission.
    # Even if we only predicted for a subset (due to mocking), the pipeline fills the rest with 0.
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_sample
    ), f"Submission length {len(df_sub)} != Sample {len(df_sample)}"

    print(f"   -> Submission generated successfully with {len(df_sub)} rows.")
    print("   -> Pipeline verified.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
