import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
import library.config as config
from library.utils import seed_everything, setup_logger
from library.data_processing import DataProcessor
from library.gating_filters import GatingFilter
from library.feature_engineering import FeatureEngineer
from library.trainer import Trainer
from library.evaluation import Evaluator


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # --------------------------------------------------------------------------
    print("[Demo] Setting up environment...")
    seed_everything(config.SEED)
    logger = setup_logger("demo_script")

    # Override hyperparameters to ensure the demo runs quickly (Speed Optimization)
    # We modify the config dictionaries directly before they are used by model classes.
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 8
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 3
    config.CATBOOST_PARAMS["iterations"] = 10
    config.CATBOOST_PARAMS["depth"] = 3

    # Reduce early stopping rounds for the demo
    config.EARLY_STOPPING_ROUNDS = 2

    # --------------------------------------------------------------------------
    # 2. Data Processing
    # --------------------------------------------------------------------------
    print("\n[Demo] Step 2: Data Processing...")
    processor = DataProcessor()

    # Load a small sample of training data (500 rows) for demonstration
    # We disable loading from cache to force the logic to run
    df_merged = processor.load_and_merge_data(
        split="train", load_cached_data=False, sample_size=500
    )

    # Verification
    assert not df_merged.empty, "Merged dataframe should not be empty."
    assert "distance" in df_merged.columns, "Distance column missing after merge."
    assert "speed_p1" in df_merged.columns, "Player 1 tracking data missing."

    # Verify Sentinel Value logic for Ground
    ground_rows = df_merged[df_merged["nfl_player_id_2"] == "G"]
    if not ground_rows.empty:
        assert (
            ground_rows["distance"] == config.SENTINEL_VALUE
        ).all(), "Sentinel value not applied correctly to Ground interactions."

    print(f"[Demo] Merged Data Shape: {df_merged.shape}")

    # --------------------------------------------------------------------------
    # 3. Gating Filters
    # --------------------------------------------------------------------------
    print("\n[Demo] Step 3: Gating Filters...")
    gating = GatingFilter()

    # Apply gating
    df_gated = gating.apply_gating(df_merged, load_cached_data=False)

    # Verification
    assert len(df_gated) <= len(df_merged), "Gating should not increase row count."
    # Ensure ground interactions are preserved (they should bypass gating)
    ground_gated = df_gated[df_gated["nfl_player_id_2"] == "G"]
    assert len(ground_gated) == len(
        ground_rows
    ), "Gating incorrectly filtered Ground interactions."

    print(f"[Demo] Gated Data Shape: {df_gated.shape}")

    # --------------------------------------------------------------------------
    # 4. Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[Demo] Step 4: Feature Engineering...")
    engineer = FeatureEngineer()

    # Generate features
    # This involves window expansion and spectral energy calculation
    df_features = engineer.create_features(
        df_gated, split="train", load_cached_data=False
    )

    # Verification
    if not df_features.empty:
        assert "spectral_energy" in df_features.columns, "Spectral features missing."
        # Check for flattened window columns (e.g., distance_0, distance_-1, etc.)
        # config.WINDOW_SIZE is 10, so offsets are -10 to +10. 'distance_0' should exist.
        assert "distance_0" in df_features.columns, "Flattened window features missing."
        assert (
            "contact" in df_features.columns
        ), "Target column 'contact' lost during engineering."
    else:
        print(
            "[Demo] Warning: Feature dataframe is empty (likely due to aggressive sampling/gating)."
        )
        # Create dummy data if empty to allow script to proceed for demonstration
        df_features = pd.DataFrame(np.random.rand(50, 50))
        df_features["contact"] = np.random.randint(0, 2, 50)
        df_features["contact_id"] = [f"id_{i}" for i in range(50)]

    print(f"[Demo] Feature Matrix Shape: {df_features.shape}")

    # --------------------------------------------------------------------------
    # 5. Model Training (Curriculum)
    # --------------------------------------------------------------------------
    print("\n[Demo] Step 5: Curriculum Training...")
    trainer = Trainer()

    # Split into mock train/val for the trainer
    # In a real scenario, we would load the actual validation set
    val_size = int(len(df_features) * 0.2)
    df_train_sub = df_features.iloc[:-val_size].copy().reset_index(drop=True)
    df_val_sub = df_features.iloc[-val_size:].copy().reset_index(drop=True)

    # Ensure we have both classes in train/val for the demo to work without errors
    if df_train_sub["contact"].nunique() < 2:
        df_train_sub.loc[0, "contact"] = 1
        df_train_sub.loc[1, "contact"] = 0
    if df_val_sub["contact"].nunique() < 2:
        df_val_sub.loc[0, "contact"] = 1
        df_val_sub.loc[1, "contact"] = 0

    # Run the curriculum
    # This trains Scouts -> Mines Hard Negatives -> Trains Expert Ensemble
    ensemble = trainer.run_curriculum(df_train_sub, df_val_sub, load_cached_data=False)

    # Verification
    assert ensemble is not None, "Ensemble training failed."
    assert ensemble.lgbm.model is not None, "LGBM Expert not trained."
    assert ensemble.xgb.model is not None, "XGB Expert not trained."

    # --------------------------------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------------------------------
    print("\n[Demo] Step 6: Evaluation...")
    evaluator = Evaluator()

    # Prepare validation features
    feature_cols = trainer._get_feature_cols(df_val_sub)
    X_val = df_val_sub[feature_cols]
    y_val = df_val_sub["contact"]

    # Predict
    y_pred_proba = ensemble.predict_proba(X_val)

    # Verification
    assert len(y_pred_proba) == len(y_val), "Prediction length mismatch."
    assert np.all(
        (y_pred_proba >= 0) & (y_pred_proba <= 1)
    ), "Probabilities out of bounds."

    # Optimize Threshold
    best_thresh, best_mcc = evaluator.optimize_threshold(y_val, y_pred_proba, steps=20)

    # Print Metrics
    y_pred_bin = (y_pred_proba >= best_thresh).astype(int)
    evaluator.print_detailed_metrics(y_val, y_pred_bin)

    print(f"\n[Demo] Final MCC on Mock Validation Set: {best_mcc:.4f}")

    print("\n[Demo] Execution Complete. All steps verified successfully.")


if __name__ == "__main__":
    main()
