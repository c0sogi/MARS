import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Import library modules
from library import (
    config,
    utils,
    preprocessing,
    features,
    gbdt_models,
    nn_model,
    ensemble,
)

# ==========================================
# CONFIGURATION OVERRIDES FOR FAST BASELINE
# ==========================================
# Limit training data to 2 million rows to ensure execution within 2 hours
config.DEBUG_SAMPLE_SIZE = 2000000
# Reduce epochs for ResNet to speed up training
config.RESNET_PARAMS["epochs"] = 8
# Adjust Tree estimators for speed vs performance trade-off in baseline
config.XGB_PARAMS["n_estimators"] = 1000
config.LGBM_PARAMS["n_estimators"] = 1000


def main():
    # Set seeds for reproducibility
    utils.seed_everything(config.SEED)

    print("Starting Runfile execution...")
    print(
        f"Configuration: Sample Size={config.DEBUG_SAMPLE_SIZE}, Device={utils.get_device()}"
    )

    # ==========================================
    # 1. DATA LOADING & PREPROCESSING
    # ==========================================
    # Load data from metadata, clean it, and apply sampling if configured
    print("\n[Step 1] Loading and Cleaning Data...")
    train_raw, val_raw, test_raw = preprocessing.load_and_clean(load_cached_data=True)

    # ==========================================
    # 2. FEATURE ENGINEERING
    # ==========================================
    # Generate features for both Tree (raw/integer) and NN (scaled/cyclical) pipelines
    print("\n[Step 2] Feature Engineering...")
    data_dict = features.process_data(
        train_raw, val_raw, test_raw, load_cached_data=True
    )

    # ==========================================
    # 3. SPLIT DATA (BASE vs META)
    # ==========================================
    print("\n[Step 3] Splitting Data for Stacking...")
    # We split the training set into Base-Train (for base models) and Meta-Train (for meta learner)
    # We must split both Tree and NN datasets consistently using indices
    indices = np.arange(len(data_dict["train_tree"]))
    train_idx, meta_idx = train_test_split(
        indices, test_size=config.META_TRAIN_SIZE, random_state=config.SEED
    )

    # Create Tree-based subsets
    base_train_tree = data_dict["train_tree"].iloc[train_idx].reset_index(drop=True)
    meta_train_tree = data_dict["train_tree"].iloc[meta_idx].reset_index(drop=True)

    # Create NN-based subsets
    base_train_nn = data_dict["train_nn"].iloc[train_idx].reset_index(drop=True)
    meta_train_nn = data_dict["train_nn"].iloc[meta_idx].reset_index(drop=True)

    # Validation and Test sets (Full)
    val_tree = data_dict["val_tree"]
    val_nn = data_dict["val_nn"]
    test_tree = data_dict["test_tree"]
    test_nn = data_dict["test_nn"]

    print(f"Base Train Size: {len(base_train_tree)}")
    print(f"Meta Train Size: {len(meta_train_tree)}")

    # ==========================================
    # 4. TRAIN BASE MODELS
    # ==========================================
    print("\n[Step 4] Training Base Models...")

    # We use the full hold-out validation set for early stopping to ensure the models
    # generalize well to unseen data, rather than overfitting to the meta-train split.

    # 4.1 XGBoost
    xgb_model, _ = gbdt_models.train_xgboost(base_train_tree, val_tree)

    # 4.2 LightGBM
    lgbm_model, _ = gbdt_models.train_lgbm(base_train_tree, val_tree)

    # 4.3 ResNet
    resnet_model, _ = nn_model.train_resnet(base_train_nn, val_nn)

    # ==========================================
    # 5. TRAIN META LEARNER
    # ==========================================
    print("\n[Step 5] Training Meta-Learner...")

    # Generate predictions on Meta-Train set to form the training data for the meta-learner
    print("Generating predictions on Meta-Train set...")

    # XGBoost Prediction
    meta_pred_xgb = xgb_model.predict(meta_train_tree[config.TREE_FEATURES])

    # LightGBM Prediction
    meta_pred_lgbm = lgbm_model.predict(meta_train_tree[config.TREE_FEATURES])

    # ResNet Prediction
    meta_pred_resnet = nn_model.predict_resnet(resnet_model, meta_train_nn)

    # Stack predictions
    meta_preds_dict = {
        "xgb": meta_pred_xgb,
        "lgbm": meta_pred_lgbm,
        "resnet": meta_pred_resnet,
    }

    # Get targets (same for tree/nn)
    meta_targets = meta_train_tree["fare_amount"].values

    # Train Meta-Learner (Ridge)
    meta_learner = ensemble.train_meta_learner(meta_preds_dict, meta_targets)

    # ==========================================
    # 6. FINAL EVALUATION
    # ==========================================
    print("\n[Step 6] Final Evaluation on Hold-out Validation Set...")

    # Generate base predictions on the full validation set
    val_pred_xgb = xgb_model.predict(val_tree[config.TREE_FEATURES])
    val_pred_lgbm = lgbm_model.predict(val_tree[config.TREE_FEATURES])
    val_pred_resnet = nn_model.predict_resnet(resnet_model, val_nn)

    val_preds_dict = {
        "xgb": val_pred_xgb,
        "lgbm": val_pred_lgbm,
        "resnet": val_pred_resnet,
    }

    # Generate ensemble predictions
    final_val_preds = ensemble.predict_meta(meta_learner, val_preds_dict)

    # Compute Metric
    y_val_true = val_tree["fare_amount"].values
    final_rmse = utils.compute_rmse(y_val_true, final_val_preds)

    print(f"Final Validation Metric: {final_rmse}")

    # ==========================================
    # 7. FAILURE ANALYSIS
    # ==========================================
    print("\n[Step 7] Failure Analysis...")

    # Calculate absolute errors
    errors = np.abs(y_val_true - final_val_preds)

    # Use the tree validation dataframe for analysis (contains raw-ish features)
    analysis_df = val_tree.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate correlation of features with error magnitude
    # Filter to numeric columns only
    numeric_df = analysis_df.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(numeric_df["error_magnitude"]).sort_values(
        ascending=False
    )

    print("Top 10 features correlated with prediction error:")
    print(correlations.head(10))

    # ==========================================
    # 8. SUBMISSION GENERATION
    # ==========================================
    THRESHOLD = 3.3898257003113574

    if final_rmse < THRESHOLD:
        print(
            f"\n[Step 8] Validation RMSE {final_rmse} < {THRESHOLD}. Generating Submission..."
        )

        # Generate base predictions on Test set
        test_pred_xgb = xgb_model.predict(test_tree[config.TREE_FEATURES])
        test_pred_lgbm = lgbm_model.predict(test_tree[config.TREE_FEATURES])
        test_pred_resnet = nn_model.predict_resnet(resnet_model, test_nn)

        test_preds_dict = {
            "xgb": test_pred_xgb,
            "lgbm": test_pred_lgbm,
            "resnet": test_pred_resnet,
        }

        # Generate ensemble predictions
        final_test_preds = ensemble.predict_meta(meta_learner, test_preds_dict)

        # Get keys from raw test data (features dataframes might not have 'key')
        keys = test_raw["key"].values

        # Save submission
        utils.save_submission(keys, final_test_preds, config.SUBMISSION_PATH)
        print(f"Submission saved successfully to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\n[Step 8] Validation RMSE {final_rmse} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
