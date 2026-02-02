import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Import library components
import library.config as config
import library.utils as utils
import library.feature_extraction as fe
import library.data_processor as dp
import library.model_definitions as md
import library.stacking_engine as se

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration script...")

    # ---------------------------------------------------------
    # 1. Utils Demonstration
    # ---------------------------------------------------------
    print("\n--- Testing Utils ---")
    utils.seed_everything(42)

    # Verify MAE calculation
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 20, 28])
    mae = utils.calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 4 / 3), f"MAE calculation failed: {mae}"
    print("Utils: Seed set and MAE calculated correctly.")

    # ---------------------------------------------------------
    # 2. Configuration Patching for Speed
    # ---------------------------------------------------------
    print("\n--- Patching Configuration for Speed ---")
    # Patch model parameters in model_definitions to reduce training time
    # This ensures that when models are instantiated, they use these reduced settings
    md.LGBM_PARAMS["n_estimators"] = 10
    md.XGB_PARAMS["n_estimators"] = 10
    md.HGB_PARAMS["max_iter"] = 10

    # Patch N_FOLDS in stacking_engine to speed up CV
    se.N_FOLDS = 2

    print("Configuration patched: n_estimators=10, N_FOLDS=2")

    # ---------------------------------------------------------
    # 3. Data Processor Demonstration
    # ---------------------------------------------------------
    print("\n--- Testing Data Processor ---")
    # Use a temporary cache directory for this run to avoid messing with real work
    demo_cache_dir = os.path.join(config.WORKING_DIR, "demo_cache_run")
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)

    # Initialize DatasetBuilder
    builder = dp.DatasetBuilder(cache_dir=demo_cache_dir)

    # Load a small subset of training data (debug_size=10)
    # This triggers the parallel feature extraction pipeline
    print("Generating features for 10 samples...")
    X, y, seg_ids = builder.get_train_data(debug_size=10, load_cached_data=False)

    # Validation of Data Shapes
    # Expected features: 10 sensors * 43 features/sensor + 45 spatial features = 475 features
    assert X.shape[0] == 10, f"Unexpected X rows: {X.shape[0]}"
    assert X.shape[1] == 475, f"Unexpected feature count: {X.shape[1]}"
    assert y.shape == (10,), f"Unexpected y shape: {y.shape}"
    assert seg_ids.shape == (10,), f"Unexpected seg_ids shape: {seg_ids.shape}"
    assert not np.isnan(X).any(), "Features contain NaNs"

    print(f"Data loaded successfully. X: {X.shape}, y: {y.shape}")

    # Test Caching Mechanism
    print("Testing cache retrieval...")
    # Calling get_train_data again should load from the parquet file created above
    X_cached, y_cached, _ = builder.get_train_data(debug_size=10, load_cached_data=True)
    assert np.array_equal(X, X_cached), "Cached X does not match original X"
    assert np.array_equal(y, y_cached), "Cached y does not match original y"
    print("Cache retrieval verified.")

    # ---------------------------------------------------------
    # 4. Feature Extraction Direct Usage
    # ---------------------------------------------------------
    print("\n--- Testing Feature Extraction ---")
    # Manually load one file to test process_segment directly
    # This ensures the logic in feature_extraction.py is working independently of the builder
    meta_df = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    sample_row = meta_df.iloc[0]
    sample_file = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    sample_df = pd.read_csv(sample_file, dtype="float32")
    features = fe.process_segment(sample_df)

    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert features.ndim == 1, "Features should be 1D"
    assert (
        len(features) == X.shape[1]
    ), f"Feature count mismatch: {len(features)} vs {X.shape[1]}"
    print(f"Feature extraction verified. Feature vector length: {len(features)}")

    # ---------------------------------------------------------
    # 5. Model Definitions Demonstration
    # ---------------------------------------------------------
    print("\n--- Testing Model Definitions ---")
    # Verify that the patched parameters are effective
    lgbm = md.get_lgbm_regressor()
    assert (
        lgbm.n_estimators == 10
    ), f"LGBM n_estimators not patched: {lgbm.n_estimators}"

    xgb_model = md.get_xgb_regressor()
    assert (
        xgb_model.n_estimators == 10
    ), f"XGB n_estimators not patched: {xgb_model.n_estimators}"

    print("Model definitions instantiated with patched parameters.")

    # ---------------------------------------------------------
    # 6. Stacking Engine Demonstration
    # ---------------------------------------------------------
    print("\n--- Testing Stacking Engine ---")
    trainer = se.StackingTrainer()

    # A. Train Base Layer (Level 0)
    # This runs Stratified K-Fold (2 folds) and trains LGBM, XGB, and HGB
    print("Training Base Layer (CV)...")
    oof_preds = trainer.train_base_layer(X, y)

    assert oof_preds.shape == (10, 3), f"OOF preds shape mismatch: {oof_preds.shape}"
    assert not np.isnan(oof_preds).any(), "OOF predictions contain NaNs"
    print("Base layer training complete.")

    # B. Train Meta Layer (Level 1)
    # Trains Ridge regression on the OOF predictions
    print("Training Meta Layer...")
    trainer.train_meta_layer(oof_preds, y)
    assert trainer.meta_model is not None, "Meta model not stored"
    print("Meta layer training complete.")

    # C. Retrain Full Base
    # Retrains base models on the entire dataset using average best iterations
    print("Retraining Base Models on Full Data...")
    # Ensure best_iterations are populated (they should be from train_base_layer)
    assert len(trainer.best_iterations["lgbm"]) > 0
    assert len(trainer.best_iterations["xgb"]) > 0

    trainer.retrain_full_base(X, y)
    assert "lgbm" in trainer.base_models
    assert "xgb" in trainer.base_models
    assert "hgb" in trainer.base_models
    print("Full base models retrained.")

    # D. Predict
    # Uses the stacked ensemble to predict on new data
    print("Generating Predictions...")
    # Using X as test data for demonstration purposes
    preds = trainer.predict_stack(X)

    assert preds.shape == (10,), f"Prediction shape mismatch: {preds.shape}"
    mae_score = utils.calculate_mae(y, preds)
    print(f"Prediction complete. MAE on training subset: {mae_score:.4f}")

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    print("\nDemo completed successfully. Temporary files cleaned up.")


if __name__ == "__main__":
    main()
