import os
import sys
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
import library.config as config
from library.utils import seed_everything
from library.data_loader import DataLoader
from library.features import FeatureFactory
from library.models import LGBMClassifierWrapper, XGBClassifierWrapper
from library.trainer import CascadeTrainer
from library.inference import Predictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting NFL Contact Detection Library Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Patch configurations for rapid execution (Demo Mode)
    print("\n[1] Patching configurations for speed...")

    # Reduce estimators and ensure silent execution to optimize runtime
    config.LGBM_SCOUT_PARAMS.update({"n_estimators": 10, "verbose": -1})
    config.LGBM_EXPERT_PARAMS.update({"n_estimators": 10, "verbose": -1})
    config.XGB_EXPERT_PARAMS.update({"n_estimators": 10, "verbosity": 0})

    # Set global seed for reproducibility
    seed_everything(42)

    # Define demo constraints
    DEMO_ROWS = 5000  # Small subset for quick processing

    # -------------------------------------------------------------------------
    # 2. Data Loader Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating DataLoader...")
    loader = DataLoader()

    # Load a subset of the training base table
    # This merges metadata with tracking data
    df_train = loader.prepare_base_table(mode="train", n_rows=DEMO_ROWS)

    # Verifications
    print(f"    Loaded Train Data Shape: {df_train.shape}")
    assert not df_train.empty, "Training dataframe is empty"
    assert "contact" in df_train.columns, "Target column 'contact' missing"
    assert (
        "x_position_p1" in df_train.columns
    ), "Tracking feature 'x_position_p1' missing"
    assert len(df_train) == DEMO_ROWS, f"Expected {DEMO_ROWS} rows, got {len(df_train)}"

    # -------------------------------------------------------------------------
    # 3. Feature Factory Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating FeatureFactory...")
    factory = FeatureFactory()

    # 3a. Compute Tier 1 Features (Instantaneous)
    print("    Computing Tier 1 Features...")
    df_tier1 = factory.compute_tier1_features(df_train, load_cached_data=False)

    # Verify Tier 1
    assert "distance" in df_tier1.columns, "Feature 'distance' missing in Tier 1"
    assert "speed_diff" in df_tier1.columns, "Feature 'speed_diff' missing in Tier 1"
    assert len(df_tier1) == len(df_train), "Tier 1 row count mismatch"

    # 3b. Compute Tier 2 Features (Contextual/Rolling)
    print("    Computing Tier 2 Features...")
    df_tier2 = factory.compute_tier2_features(df_train, load_cached_data=False)

    # Verify Tier 2
    # Check for a rolling feature (e.g., distance_lag_1)
    assert (
        "distance_lag_1" in df_tier2.columns
    ), "Rolling feature 'distance_lag_1' missing in Tier 2"
    assert "jerk_p1" in df_tier2.columns, "Derived feature 'jerk_p1' missing in Tier 2"
    assert len(df_tier2) == len(df_train), "Tier 2 row count mismatch"

    # -------------------------------------------------------------------------
    # 4. Model Wrapper Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Wrappers...")

    # Prepare dummy data for model testing
    X_demo = df_tier1
    y_demo = df_train["contact"]

    # 4a. LightGBM Wrapper
    print("    Testing LGBMClassifierWrapper...")
    lgbm_wrapper = LGBMClassifierWrapper(config.LGBM_SCOUT_PARAMS, name="demo_lgbm")
    lgbm_wrapper.fit(X_demo, y_demo)

    # Predict
    preds_lgbm = lgbm_wrapper.predict_proba(X_demo)
    assert len(preds_lgbm) == len(X_demo)
    assert preds_lgbm.min() >= 0 and preds_lgbm.max() <= 1

    # Save/Load
    demo_model_path = os.path.join(config.WORKING_DIR, "demo_lgbm.joblib")
    lgbm_wrapper.save(demo_model_path)
    assert os.path.exists(demo_model_path)

    lgbm_loaded = LGBMClassifierWrapper(
        config.LGBM_SCOUT_PARAMS, name="demo_lgbm_loaded"
    )
    lgbm_loaded.load(demo_model_path)
    print("    LGBM Save/Load successful.")

    # 4b. XGBoost Wrapper
    print("    Testing XGBClassifierWrapper...")
    xgb_wrapper = XGBClassifierWrapper(config.XGB_EXPERT_PARAMS, name="demo_xgb")
    xgb_wrapper.fit(X_demo, y_demo)

    # Predict
    preds_xgb = xgb_wrapper.predict_proba(X_demo)
    assert len(preds_xgb) == len(X_demo)

    # Save
    demo_xgb_path = os.path.join(config.WORKING_DIR, "demo_xgb.joblib")
    xgb_wrapper.save(demo_xgb_path)
    assert os.path.exists(demo_xgb_path)
    print("    XGBoost Save/Load successful.")

    # -------------------------------------------------------------------------
    # 5. Cascade Trainer Demonstration (Full Pipeline)
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating CascadeTrainer (Full Pipeline)...")
    print("    Running Scout -> Mining -> Expert -> Submission generation")

    trainer = CascadeTrainer()

    # Run the pipeline with debug_rows to ensure it finishes quickly
    # This will:
    # 1. Train Scout on balanced subset of 5000 rows
    # 2. Mine hard negatives from the 5000 rows
    # 3. Train Experts on mined data
    # 4. Generate submission for the test set
    trainer.run(debug_rows=DEMO_ROWS)

    # Verify Submission
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), "Submission file was not generated by Trainer"
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns
    print(f"    Trainer run complete. Submission shape: {df_sub.shape}")

    # -------------------------------------------------------------------------
    # 6. Predictor Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Predictor (Inference Standalone)...")

    # The Predictor class is designed to load saved models and run inference.
    # The Trainer run above has already saved 'expert_lgbm.joblib' and 'expert_xgb.joblib'
    # in the WORKING_DIR.

    predictor = Predictor(model_dir=config.WORKING_DIR)

    # Run inference explicitly
    # Note: This repeats the inference step done by trainer.run(), but isolates the class usage.
    # We use a threshold of 0.5 for demonstration.
    df_pred_sub = predictor.predict(threshold=0.5)

    assert not df_pred_sub.empty
    assert df_pred_sub.shape == df_sub.shape
    print("    Predictor run complete.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
