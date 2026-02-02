import os
import shutil
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.data_loader import DataLoader
from library.models import LGBMModel, XGBModel, EnsemblePredictor
from library.trainer import Trainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override for Speed
    # -------------------------------------------------------------------------
    print(">>> Step 1: Setup and Configuration Override")
    seed_everything(Config.SEED)
    setup_logging()

    # Create a demo working directory to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to point to demo directory
    Config.WORKING_DIR = demo_dir

    # Override Hyperparameters for fast execution
    # Reducing n_estimators to 10 ensures training finishes in seconds
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["verbose"] = -1

    Config.XGB_PARAMS["n_estimators"] = 10

    Config.EARLY_STOPPING_ROUNDS = 5

    # Set Submission Path
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    print("Configuration patched for rapid demonstration.")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Feature Engineering Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 2: Data Loading and Feature Engineering")
    loader = DataLoader()

    # Load a small sample (e.g., 500 rows)
    # This triggers FeatureEngineer -> _load_raw_data -> _merge_data -> _compute_vector_physics -> _apply_gating
    sample_size = 500
    df_train = loader.load_train_data(load_cached_data=False, sample_size=sample_size)

    # Assertions to verify structure
    assert not df_train.empty, "Training dataframe should not be empty."
    assert (
        len(df_train) <= sample_size
    ), f"Length {len(df_train)} exceeds sample size {sample_size} (gating might reduce it, which is fine)."

    # Verify Physics Features
    expected_features = [
        "distance",
        "v_radial",
        "v_tangential",
        "time_to_collision",
        "contact",
    ]
    for feat in expected_features:
        assert feat in df_train.columns, f"Missing feature: {feat}"

    # Verify Ground Sentinel Logic
    # Rows where nfl_player_id_2 == 'G' should have distance == Config.GROUND_DISTANCE_SENTINEL (-1.0)
    ground_rows = df_train[df_train["nfl_player_id_2"] == "G"]
    if not ground_rows.empty:
        assert np.allclose(
            ground_rows["distance"], Config.GROUND_DISTANCE_SENTINEL
        ), "Ground rows do not have the correct sentinel distance."

    print(f"Data Loaded Successfully. Shape: {df_train.shape}")

    # -------------------------------------------------------------------------
    # 3. Individual Model Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 3: Individual Model Verification")

    # Prepare simple X, y
    features = Config.FEATURES
    X = df_train[features]
    y = df_train["contact"]

    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train_demo, X_val_demo = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_demo, y_val_demo = y.iloc[:split_idx], y.iloc[split_idx:]

    # Test LGBM
    print("Testing LGBMModel...")
    lgbm = LGBMModel()
    lgbm.fit(X_train_demo, y_train_demo, X_val_demo, y_val_demo)
    preds_lgbm = lgbm.predict_proba(X_val_demo)
    assert preds_lgbm.shape == (len(X_val_demo),), "LGBM prediction shape mismatch."
    assert (
        preds_lgbm.min() >= 0 and preds_lgbm.max() <= 1
    ), "LGBM predictions out of probability range."

    # Test Persistence
    model_dir = os.path.join(demo_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    lgbm.save(model_dir)
    lgbm_loaded = LGBMModel()
    lgbm_loaded.load(model_dir)
    assert lgbm_loaded.model is not None, "LGBM loading failed."

    # Test XGB
    print("Testing XGBModel...")
    xgb_model = XGBModel()
    xgb_model.fit(X_train_demo, y_train_demo, X_val_demo, y_val_demo)
    preds_xgb = xgb_model.predict_proba(X_val_demo)
    assert (
        preds_xgb.min() >= 0 and preds_xgb.max() <= 1
    ), "XGB predictions out of probability range."

    # -------------------------------------------------------------------------
    # 4. Full Trainer Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n>>> Step 4: Full Trainer Pipeline Execution")

    # Instantiate Trainer
    trainer = Trainer()

    # We use a slightly larger sample size for the pipeline to ensure we get enough positives/negatives
    # for the mining and smoothing logic to work without empty set errors.
    pipeline_sample_size = 2000

    # Run the full pipeline
    # This includes:
    # 1. train_scouts (LGBM/XGB on balanced data)
    # 2. mine_hard_negatives (Using scouts to find hard examples)
    # 3. train_experts (Ensemble on Positives + Hard Negatives + Smoothing)
    # 4. optimize_threshold (Finding best MCC threshold on Val)
    # 5. generate_submission (Inference on Test)
    trainer.run_training_pipeline(sample_size=pipeline_sample_size)

    # -------------------------------------------------------------------------
    # 5. Submission Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 5: Submission Verification")

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    assert "contact_id" in df_sub.columns, "contact_id column missing in submission."
    assert "contact" in df_sub.columns, "contact column missing in submission."

    # Check values
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contact column contains non-binary values."

    # Check against sample submission length (test_metadata is derived from sample_submission)
    # Note: The provided sample_submission.csv has 463243 rows.
    # The pipeline runs inference on the full test set loaded via load_test_data.
    # load_test_data uses test_metadata.csv, which matches sample_submission.
    # So the lengths should match exactly.
    sample_sub_ref = pd.read_csv("./input/sample_submission.csv")
    assert len(df_sub) == len(
        sample_sub_ref
    ), f"Submission length {len(df_sub)} does not match sample_submission {len(sample_sub_ref)}."

    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
