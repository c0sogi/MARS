import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import SEED, TARGET_COL, ID_COL
from library.data_processor import process_data
from library.model_wrappers import LGBMWrapper, XGBWrapper
from library.cross_validator import run_stratified_kfold
from library.ensembler import optimize_weights, blend_predictions, generate_submission


def main():
    # Set seed for reproducibility
    np.random.seed(SEED)

    print("=== Starting Task Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Data Processing
    # -------------------------------------------------------------------------
    print("\n[1/5] Processing Data...")
    # We force load_cached_data=False to demonstrate the feature engineering logic
    df_train, df_test = process_data(load_cached_data=False)

    # Validation: Check if critical columns exist
    assert (
        TARGET_COL in df_train.columns
    ), f"Target column '{TARGET_COL}' missing from train data."
    assert ID_COL in df_train.columns, f"ID column '{ID_COL}' missing from train data."

    # Validation: Check if interaction features were created
    # Based on config.py, 'Wilderness_Area1_x_Elevation' should exist
    expected_feat = "Wilderness_Area1_x_Elevation"
    assert (
        expected_feat in df_train.columns
    ), f"Feature engineering failed: '{expected_feat}' not found."

    print(
        f"Data processed. Full Train Shape: {df_train.shape}, Full Test Shape: {df_test.shape}"
    )

    # -------------------------------------------------------------------------
    # 2. Subsampling for Speed
    # -------------------------------------------------------------------------
    print("\n[2/5] Subsampling for Fast Demonstration...")

    # Filter out classes with too few samples to survive stratification in a small subset
    class_counts = df_train[TARGET_COL].value_counts()
    valid_classes = class_counts[class_counts >= 20].index
    df_train_filtered = df_train[df_train[TARGET_COL].isin(valid_classes)].copy()

    # Create a small subset: 2000 train rows, 500 test rows
    # Stratify train split to ensure all classes are represented for CV
    df_train_sub, _ = train_test_split(
        df_train_filtered,
        train_size=2000,
        stratify=df_train_filtered[TARGET_COL],
        random_state=SEED,
    )

    df_test_sub = df_test.sample(n=500, random_state=SEED).copy()

    print(f"Subset Train Shape: {df_train_sub.shape}")
    print(f"Subset Test Shape: {df_test_sub.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Execution (Cross-Validation)
    # -------------------------------------------------------------------------
    print("\n[3/5] Running Cross-Validation...")

    # Define lightweight parameters for fast CPU execution
    # We override the heavy GPU params from config.py for this demo
    lgbm_demo_params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "n_estimators": 10,  # Very few trees for speed
        "num_leaves": 15,
        "learning_rate": 0.1,
        "verbose": -1,
        "device": "cpu",  # Use CPU for small data to avoid overhead
        "n_jobs": 4,
        "random_state": SEED,
    }

    xgb_demo_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": 10,  # Very few trees for speed
        "max_depth": 3,
        "learning_rate": 0.1,
        "tree_method": "hist",  # CPU histogram method
        "device": "cpu",
        "n_jobs": 4,
        "random_state": SEED,
    }

    # Run LGBM
    print("  -> Training LightGBM...")
    oof_lgbm, test_lgbm, classes_lgbm = run_stratified_kfold(
        LGBMWrapper,
        lgbm_demo_params,
        df_train_sub,
        df_test_sub,
        n_folds=2,
        verbose=True,
    )

    # Run XGBoost
    print("  -> Training XGBoost...")
    oof_xgb, test_xgb, classes_xgb = run_stratified_kfold(
        XGBWrapper, xgb_demo_params, df_train_sub, df_test_sub, n_folds=2, verbose=True
    )

    # Validation: Check consistency between models
    assert np.array_equal(
        classes_lgbm, classes_xgb
    ), "Class labels mismatch between models."
    classes = classes_lgbm
    n_classes = len(classes)

    # Validation: Check output shapes
    assert oof_lgbm.shape == (len(df_train_sub), n_classes), "LGBM OOF shape incorrect."
    assert test_lgbm.shape == (
        len(df_test_sub),
        n_classes,
    ), "LGBM Test shape incorrect."
    assert oof_xgb.shape == (len(df_train_sub), n_classes), "XGB OOF shape incorrect."

    # -------------------------------------------------------------------------
    # 4. Ensembling
    # -------------------------------------------------------------------------
    print("\n[4/5] Optimizing Ensemble Weights...")

    oof_preds_dict = {"lgbm": oof_lgbm, "xgb": oof_xgb}

    test_preds_dict = {"lgbm": test_lgbm, "xgb": test_xgb}

    y_true = df_train_sub[TARGET_COL]

    # Optimize
    weights = optimize_weights(oof_preds_dict, y_true, classes)

    # Validation: Weights must sum to ~1
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-4, f"Weights do not sum to 1: {total_weight}"

    # Blend Test Predictions
    blended_preds = blend_predictions(test_preds_dict, weights)
    assert blended_preds.shape == (
        len(df_test_sub),
        n_classes,
    ), "Blended predictions shape incorrect."

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Submission...")

    output_dir = "./working"
    output_file = "demo_submission.csv"
    output_path = os.path.join(output_dir, output_file)

    test_ids = df_test_sub[ID_COL]

    generate_submission(test_ids, blended_preds, classes, output_path)

    # Validation: Verify file creation and content
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Submission file was not created at {output_path}")

    df_sub_check = pd.read_csv(output_path)
    assert df_sub_check.shape == (
        len(df_test_sub),
        2,
    ), "Submission file has incorrect dimensions."
    assert list(df_sub_check.columns) == [
        ID_COL,
        TARGET_COL,
    ], "Submission file has incorrect columns."
    assert (
        df_sub_check[ID_COL].dtype == "int64" or df_sub_check[ID_COL].dtype == "int32"
    ), "Id column should be integer."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
