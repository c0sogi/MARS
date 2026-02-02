import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library import (
    config,
    utils,
    preprocessing,
    features,
    gbdt_models,
    nn_model,
    ensemble,
)


def main():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ==========================================
    print("Initializing demonstration...")
    utils.seed_everything(config.SEED)

    # Override config for speed (Demonstration Mode)
    print("Overriding configuration for fast execution...")
    config.DEBUG_SAMPLE_SIZE = 5000  # Use only 5000 rows

    # Reduce Tree Estimators
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["early_stopping_rounds"] = 5

    config.LGBM_PARAMS["n_estimators"] = 10

    # Reduce NN Complexity and Epochs
    config.RESNET_PARAMS["epochs"] = 1
    config.RESNET_PARAMS["batch_size"] = 128
    config.RESNET_PARAMS["hidden_dims"] = [64, 32]  # Smaller network for demo

    # Clean up working directory to ensure fresh run for demo
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. DATA LOADING & PREPROCESSING
    # ==========================================
    print("\nStep 2: Loading and Cleaning Data...")
    # Force reload from metadata to verify cleaning logic
    train_df, val_df, test_df = preprocessing.load_and_clean(load_cached_data=False)

    # Validation
    assert (
        len(train_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert not train_df.isnull().values.any(), "Cleaned train data contains NaNs"
    assert "fare_amount" in train_df.columns, "Target column missing in train"
    print("Data loaded and cleaned successfully.")

    # ==========================================
    # 3. FEATURE ENGINEERING
    # ==========================================
    print("\nStep 3: Feature Engineering...")
    # Process features for both pipelines
    data_dict = features.process_data(train_df, val_df, test_df, load_cached_data=False)

    # Validation
    expected_keys = [
        "train_tree",
        "val_tree",
        "test_tree",
        "train_nn",
        "val_nn",
        "test_nn",
    ]
    for key in expected_keys:
        assert key in data_dict, f"Missing key in feature dict: {key}"
        assert len(data_dict[key]) > 0, f"Feature dataframe {key} is empty"

    # Check specific feature existence
    assert (
        "haversine_dist" in data_dict["train_tree"].columns
    ), "Engineered feature missing"
    assert "hour_sin" in data_dict["train_nn"].columns, "NN cyclical feature missing"
    print("Feature engineering completed successfully.")

    # ==========================================
    # 4. MODEL TRAINING (GBDT)
    # ==========================================
    print("\nStep 4: Training GBDT Models...")

    # XGBoost
    xgb_model, xgb_val_preds = gbdt_models.train_xgboost(
        data_dict["train_tree"], data_dict["val_tree"]
    )
    assert xgb_model is not None
    assert len(xgb_val_preds) == len(data_dict["val_tree"])

    # LightGBM
    lgbm_model, lgbm_val_preds = gbdt_models.train_lgbm(
        data_dict["train_tree"], data_dict["val_tree"]
    )
    assert lgbm_model is not None
    assert len(lgbm_val_preds) == len(data_dict["val_tree"])
    print("GBDT training completed.")

    # ==========================================
    # 5. MODEL TRAINING (NEURAL NETWORK)
    # ==========================================
    print("\nStep 5: Training Neural Network...")

    resnet_model, resnet_val_preds = nn_model.train_resnet(
        data_dict["train_nn"], data_dict["val_nn"]
    )
    assert isinstance(resnet_model, torch.nn.Module)
    assert len(resnet_val_preds) == len(data_dict["val_nn"])
    print("Neural Network training completed.")

    # ==========================================
    # 6. ENSEMBLING (META LEARNER)
    # ==========================================
    print("\nStep 6: Training Meta-Learner...")

    # Collect Base Predictions
    base_preds_val = {
        "xgb": xgb_val_preds,
        "lgbm": lgbm_val_preds,
        "resnet": resnet_val_preds,
    }

    # Get True Targets
    y_val_true = data_dict["val_tree"]["fare_amount"].values

    # Train Meta Learner
    meta_learner = ensemble.train_meta_learner(base_preds_val, y_val_true)
    assert meta_learner is not None
    print("Meta-Learner trained successfully.")

    # ==========================================
    # 7. INFERENCE & SUBMISSION
    # ==========================================
    print("\nStep 7: Generating Submission...")

    # A. Base Model Inference on Test Set
    print("Generating base model predictions on test set...")

    # XGBoost
    # Note: XGBoost model from library is wrapped or raw.
    # train_xgboost returns the xgb.XGBRegressor object which has .predict
    xgb_test_preds = xgb_model.predict(data_dict["test_tree"][config.TREE_FEATURES])

    # LightGBM
    lgbm_test_preds = lgbm_model.predict(data_dict["test_tree"][config.TREE_FEATURES])

    # ResNet
    resnet_test_preds = nn_model.predict_resnet(resnet_model, data_dict["test_nn"])

    # Validation of prediction shapes
    test_len = len(test_df)
    assert len(xgb_test_preds) == test_len
    assert len(lgbm_test_preds) == test_len
    assert len(resnet_test_preds) == test_len

    # B. Ensemble Inference
    base_preds_test = {
        "xgb": xgb_test_preds,
        "lgbm": lgbm_test_preds,
        "resnet": resnet_test_preds,
    }

    final_preds = ensemble.predict_meta(meta_learner, base_preds_test)
    assert len(final_preds) == test_len

    # C. Save Submission
    # We need the 'key' column from the original test dataframe
    # The 'test_df' returned by preprocessing has the keys.
    # Note: The 'test_tree' dataframe might not have 'key' if transform_tree only selects features.
    # We rely on the row order preservation which is guaranteed by the pipeline.

    # Reload test.csv to ensure we have the exact keys required for submission
    # (preprocessing.load_and_clean returns cleaned dataframes, but let's be safe and read keys from metadata/test.parquet)
    # Actually, preprocessing.load_and_clean returns test_df which comes from metadata/test.parquet and has 'key'.
    keys = test_df["key"].values

    utils.save_submission(keys, final_preds, config.SUBMISSION_PATH)

    # Verify File Creation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Verify File Content Format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["key", "fare_amount"]
    assert len(sub_df) == test_len

    print(f"\nSubmission generated at: {config.SUBMISSION_PATH}")
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()
