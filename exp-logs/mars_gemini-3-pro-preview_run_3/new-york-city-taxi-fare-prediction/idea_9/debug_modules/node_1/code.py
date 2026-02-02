import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib

# Import from provided library
import library.config as config
from library.utils import seed_everything
from library.data_processing import load_data, clean_data, split_data
from library.feature_engineering import FeatureEngineer
from library.model_tree import train_xgboost, train_lgbm, predict_tree_model
from library.model_nn import train_nn_model, predict_nn_model
from library.meta_learner import train_meta_learner, predict_meta, generate_submission


def run_demo():
    print("=== Starting NYC Taxi Fare Prediction Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    seed_everything(config.SEED)

    # Override config parameters to ensure quick execution
    print("Configuring hyperparameters for fast demonstration...")
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["early_stopping_rounds"] = 5

    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["early_stopping_rounds"] = 5

    config.NN_PARAMS["epochs"] = 1
    config.NN_PARAMS["batch_size"] = 256
    config.NN_PARAMS["hidden_dims"] = [64, 32]  # Smaller network for demo

    # Define a small sample size for the demo
    DEMO_SAMPLE_SIZE = 5000

    # 2. Data Processing
    print(f"\n[Data Processing] Loading {DEMO_SAMPLE_SIZE} samples...")

    # Load Train (Sampled)
    df_train_full = load_data(config.PATH_TRAIN, sample_size=DEMO_SAMPLE_SIZE)
    assert len(df_train_full) == DEMO_SAMPLE_SIZE, "Training data sampling failed."

    # Load Val (Sampled)
    df_val = load_data(config.PATH_VAL, sample_size=DEMO_SAMPLE_SIZE)

    # Load Test (Full - it's small enough)
    df_test = load_data(config.PATH_TEST, sample_size=None)

    print("[Data Processing] Cleaning data...")
    df_train_full = clean_data(df_train_full, is_train=True)
    df_val = clean_data(df_val, is_train=True)
    df_test = clean_data(df_test, is_train=False)

    # Verify cleaning
    assert not df_train_full.empty, "Training data is empty after cleaning."
    assert "fare_amount" in df_train_full.columns, "Target column missing."

    print("[Data Processing] Splitting training data into Base and Meta sets...")
    df_train_base, df_train_meta = split_data(df_train_full)

    print(f"  Base Train Shape: {df_train_base.shape}")
    print(f"  Meta Train Shape: {df_train_meta.shape}")

    # 3. Feature Engineering
    print("\n[Feature Engineering] Initializing and Fitting Scaler...")
    engineer = FeatureEngineer()

    # Fit scaler on base training data
    engineer.fit_scaler(df_train_base)
    assert os.path.exists(engineer.scaler_path), "Scaler file was not saved."

    # Helper to transform and verify
    def transform_and_verify(df, name):
        print(f"  Transforming {name}...")
        tree_data = engineer.transform_tree(df)
        nn_data = engineer.transform_nn(df)

        # assertions
        assert (
            "haversine_dist" in tree_data.columns
        ), f"Haversine missing in {name} tree data"
        assert (
            "grid_pickup_lat" in nn_data.columns
        ), f"Grid embedding missing in {name} nn data"

        # Check for NaNs in tree data (NN data might have them if scaler encountered new range, but shouldn't with standard scaler)
        assert not tree_data.isnull().any().any(), f"NaNs found in {name} tree data"

        return tree_data, nn_data

    # Transform all sets
    train_base_tree, train_base_nn = transform_and_verify(df_train_base, "Train Base")
    train_meta_tree, train_meta_nn = transform_and_verify(df_train_meta, "Train Meta")
    val_tree, val_nn = transform_and_verify(df_val, "Validation")
    test_tree, test_nn = transform_and_verify(df_test, "Test")

    # 4. Model Training (Base Models)
    models = {}

    # A. XGBoost
    print("\n[Model Training] XGBoost...")
    models["xgboost"] = train_xgboost(
        train_base_tree, val_tree, load_cached_model=False  # Force retrain for demo
    )

    # Verify XGB prediction
    xgb_preds = predict_tree_model(models["xgboost"], val_tree)
    assert len(xgb_preds) == len(val_tree), "XGBoost prediction length mismatch"
    print(
        f"  XGBoost Val RMSE: {np.sqrt(((val_tree['fare_amount'] - xgb_preds)**2).mean()):.4f}"
    )

    # B. LightGBM
    print("\n[Model Training] LightGBM...")
    models["lgbm"] = train_lgbm(train_base_tree, val_tree, load_cached_model=False)

    # Verify LGBM prediction
    lgbm_preds = predict_tree_model(models["lgbm"], val_tree)
    assert len(lgbm_preds) == len(val_tree), "LightGBM prediction length mismatch"

    # C. Neural Network
    print("\n[Model Training] Spatial ResNet...")
    models["nn"] = train_nn_model(train_base_nn, val_nn, load_cached_model=False)

    # Verify NN prediction
    nn_preds = predict_nn_model(models["nn"], val_nn)
    assert len(nn_preds) == len(val_nn), "NN prediction length mismatch"

    # 5. Meta Learner
    print("\n[Meta Learner] Training Stacking Model...")
    meta_model = train_meta_learner(
        models, train_meta_tree, train_meta_nn, load_cached_model=False
    )

    # 6. Final Inference and Submission
    print("\n[Inference] Generating Final Predictions...")
    final_preds = predict_meta(models, meta_model, test_tree, test_nn)

    assert len(final_preds) == len(
        df_test
    ), "Final prediction count does not match test set size."
    assert not np.isnan(final_preds).any(), "Final predictions contain NaNs."

    print(f"  Predictions generated. Sample: {final_preds[:5]}")

    # Generate Submission File
    generate_submission(df_test, final_preds)

    assert os.path.exists(config.PATH_SUBMISSION), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(config.PATH_SUBMISSION)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns are incorrect."
    assert len(sub_df) == len(df_test), "Submission row count mismatch."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
