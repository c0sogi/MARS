import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_squared_error

# Import from provided libraries
import library.config as config
from library.utils import seed_everything, get_device
from library.data_processing import process_data
from library.feature_engineering import process_features
from library.model_tree import train_xgboost, train_lgbm, predict_tree_model
from library.model_nn import train_nn_model, predict_nn_model
from library.meta_learner import train_meta_learner, predict_meta, generate_submission


def main():
    # 1. Setup
    seed_everything(config.SEED)

    # Monkey-patch config for fast baseline execution
    # Limit epochs and estimators to ensure run completes in < 2 hours
    # We modify the dictionaries in place
    config.NN_PARAMS["epochs"] = 5
    config.XGB_PARAMS["n_estimators"] = 1000
    config.LGBM_PARAMS["n_estimators"] = 1000

    # 2. Data Processing
    print("Step 2: Loading and Processing Data...")
    # We load cached data if available to save time on raw parsing
    # process_data returns: df_train_base, df_train_meta, df_val, df_test
    df_train_base, df_train_meta, df_val, df_test = process_data(load_cached_data=True)

    # SAMPLING FOR FAST BASELINE
    # We sample the training sets to ensure the pipeline runs quickly.
    # We must keep validation and test sets full as per requirements.
    SAMPLE_SIZE_BASE = 200_000
    SAMPLE_SIZE_META = 50_000

    if len(df_train_base) > SAMPLE_SIZE_BASE:
        print(f"Sampling Base Train from {len(df_train_base)} to {SAMPLE_SIZE_BASE}...")
        df_train_base = df_train_base.sample(
            n=SAMPLE_SIZE_BASE, random_state=config.SEED
        ).reset_index(drop=True)

    if len(df_train_meta) > SAMPLE_SIZE_META:
        print(f"Sampling Meta Train from {len(df_train_meta)} to {SAMPLE_SIZE_META}...")
        df_train_meta = df_train_meta.sample(
            n=SAMPLE_SIZE_META, random_state=config.SEED
        ).reset_index(drop=True)

    # 3. Feature Engineering
    print("Step 3: Feature Engineering...")
    # We force load_cached_data=False because we just sampled the dataframes in memory
    # and we want the features to correspond to these samples.
    # Note: This will overwrite cache in working/ with the sampled versions.
    feature_data = process_features(
        df_train_base, df_train_meta, df_val, df_test, load_cached_data=False
    )

    # 4. Train Base Models (Level 0)
    print("Step 4: Training Base Models...")
    models = {}

    # XGBoost
    # We force load_cached_model=False to ensure we train on our sampled data
    models["xgboost"] = train_xgboost(
        feature_data["train_base_tree"],
        feature_data["val_tree"],
        load_cached_model=False,
    )

    # LightGBM
    models["lgbm"] = train_lgbm(
        feature_data["train_base_tree"],
        feature_data["val_tree"],
        load_cached_model=False,
    )

    # Spatial ResNet
    models["nn"] = train_nn_model(
        feature_data["train_base_nn"], feature_data["val_nn"], load_cached_model=False
    )

    # 5. Train Meta Learner (Level 1)
    print("Step 5: Training Meta Learner...")
    meta_model = train_meta_learner(
        models,
        feature_data["train_meta_tree"],
        feature_data["train_meta_nn"],
        load_cached_model=False,
    )

    # 6. Validation
    print("Step 6: Final Validation...")
    # Predict on full validation set
    val_preds = predict_meta(
        models, meta_model, feature_data["val_tree"], feature_data["val_nn"]
    )

    # Calculate RMSE
    # We extract the target from the validation dataframe
    y_val = feature_data["val_tree"]["fare_amount"].values
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_rmse}")

    # 7. Failure Analysis
    print("Step 7: Failure Analysis...")
    residuals = np.abs(y_val - val_preds)

    # Select features for correlation analysis (using tree features as they are raw/interpretable)
    analysis_df = feature_data["val_tree"].copy()
    analysis_df["residuals"] = residuals

    # Define features to check (Spatial and Temporal)
    check_feats = [
        "haversine_dist",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "hour",
        "year",
    ]

    print("Correlation between Residuals and Features:")
    for feat in check_feats:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["residuals"])
            print(f"  {feat}: {corr:.4f}")

    # 8. Submission
    print("Step 8: Submission Generation...")
    # Threshold defined in task
    THRESHOLD = 3.3898257003113574

    if val_rmse < THRESHOLD:
        print(
            f"Validation RMSE ({val_rmse}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        # Get raw test data for keys (preserved in df_test from process_data)
        df_test_raw = df_test

        test_preds = predict_meta(
            models, meta_model, feature_data["test_tree"], feature_data["test_nn"]
        )

        generate_submission(df_test_raw, test_preds)
    else:
        print(
            f"Validation RMSE ({val_rmse}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
