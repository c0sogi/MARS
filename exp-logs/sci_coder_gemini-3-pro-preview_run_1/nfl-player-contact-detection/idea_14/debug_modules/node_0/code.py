import os
import sys
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import WORKING_DIR, MODEL_OUTPUT_DIR, SEED
from library.data_factory import DataFactory
from library.model_factory import LGBMWrapper, XGBWrapper, EnsemblePredictor
from library.training_pipeline import TrainingPipeline
from library.utils import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def demo_data_factory():
    print("\n[DEMO] Testing DataFactory and Feature Engineering...")

    # Initialize DataFactory
    factory = DataFactory()

    # Load a very small sample to test feature generation speed and logic
    sample_size = 200
    print(f"Generating features for {sample_size} rows...")

    # We use a specific mode string to avoid overwriting main cache
    df_features = factory.get_processed_dataset(
        mode="train", sample_size=sample_size, load_cached_data=False
    )

    # Assertions
    print("Verifying DataFactory output...")
    assert not df_features.empty, "Feature DataFrame should not be empty."
    assert (
        len(df_features) <= sample_size
    ), "Result length should respect sample size (gating might reduce it)."

    # Check for critical columns
    expected_cols = ["contact", "distance", "speed_p1", "n1_dist", "is_ground"]
    for col in expected_cols:
        assert col in df_features.columns, f"Missing expected column: {col}"

    # Check for NaNs in critical features
    assert not df_features["distance"].isnull().any(), "Distance feature contains NaNs."

    print("DataFactory test passed. Features generated successfully.")
    return df_features


def demo_model_factory(df_train):
    print("\n[DEMO] Testing ModelFactory (LGBM & XGB)...")

    # Prepare data
    feature_cols = [
        c
        for c in df_train.columns
        if c
        not in [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
    ]
    X = df_train[feature_cols]
    y = df_train["contact"]

    # Split (Reuse data for train/val to keep it simple for API test)
    X_train, y_train = X, y
    X_val, y_val = X, y

    # 1. Test LightGBM
    print("Testing LightGBM Wrapper...")
    lgbm_overrides = {"n_estimators": 5, "verbose": -1}
    lgbm = LGBMWrapper(overrides=lgbm_overrides)
    lgbm.train(X_train, y_train, X_val, y_val)

    preds_lgbm = lgbm.predict(X_val)
    assert len(preds_lgbm) == len(X_val), "Prediction length mismatch (LGBM)."
    assert np.all(
        (preds_lgbm >= 0) & (preds_lgbm <= 1)
    ), "Predictions out of bounds (LGBM)."

    # Test Save/Load
    lgbm.save("test_lgbm.joblib")
    lgbm_loaded = LGBMWrapper()
    lgbm_loaded.load("test_lgbm.joblib")
    assert lgbm_loaded.model is not None, "Failed to load LGBM model."

    # 2. Test XGBoost
    print("Testing XGBoost Wrapper...")
    xgb_overrides = {"n_estimators": 5, "verbosity": 0}
    xgb_model = XGBWrapper(overrides=xgb_overrides)
    xgb_model.train(X_train, y_train, X_val, y_val)

    preds_xgb = xgb_model.predict(X_val)
    assert len(preds_xgb) == len(X_val), "Prediction length mismatch (XGB)."

    xgb_model.save("test_xgb.joblib")

    print("ModelFactory test passed.")


def demo_training_pipeline():
    print("\n[DEMO] Testing TrainingPipeline (Debug Mode)...")

    # Initialize pipeline in debug mode
    # This sets n_estimators=50 and uses a sample_size=5000
    pipeline = TrainingPipeline(debug=True)

    # Execute the full pipeline
    # This covers: Scout Training -> Hard Negative Mining -> Expert Training -> Threshold Opt
    pipeline.run()

    # Verify Artifacts
    print("Verifying Pipeline Artifacts...")
    expected_files = ["expert_lgbm.joblib", "expert_xgb.joblib", "best_threshold.npy"]
    for f in expected_files:
        path = os.path.join(MODEL_OUTPUT_DIR, f)
        assert os.path.exists(path), f"Pipeline failed to generate {f}"

    print("TrainingPipeline test passed.")


def demo_inference_logic(df_sample):
    print("\n[DEMO] Testing Inference Logic (EnsemblePredictor)...")

    # Paths to the models created by the pipeline
    lgbm_path = os.path.join(MODEL_OUTPUT_DIR, "expert_lgbm.joblib")
    xgb_path = os.path.join(MODEL_OUTPUT_DIR, "expert_xgb.joblib")

    # Initialize Predictor
    predictor = EnsemblePredictor(lgbm_path=lgbm_path, xgb_path=xgb_path)

    # Prepare X
    feature_cols = [
        c
        for c in df_sample.columns
        if c
        not in [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
    ]
    X = df_sample[feature_cols]

    # Predict
    preds = predictor.predict(X)

    # Assertions
    assert isinstance(preds, np.ndarray), "Prediction should be a numpy array."
    assert preds.shape == (len(X),), "Prediction shape mismatch."
    assert not np.isnan(preds).any(), "Predictions contain NaNs."

    print(f"Inference test passed. Mean probability: {np.mean(preds):.4f}")


if __name__ == "__main__":
    set_seed(SEED)

    print("===========================================================")
    print("   NFL Contact Detection: Library Usage Demonstration      ")
    print("===========================================================")

    try:
        # 1. Test Data Loading and Feature Engineering
        df_sample = demo_data_factory()

        # 2. Test Individual Model Wrappers
        demo_model_factory(df_sample)

        # 3. Test Full Training Pipeline (Integration Test)
        demo_training_pipeline()

        # 4. Test Inference Logic using Pipeline Artifacts
        # We reuse the sample dataframe as a proxy for test data
        demo_inference_logic(df_sample)

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
