import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Import library modules
import library.config
from library.data import DataManager
from library.model import StratifiedModel
from library.utils import calculate_log_mae, timer

# Set Seeds for Reproducibility
np.random.seed(library.config.RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration Patching for Fast Baseline
# ==========================================
# We override the default parameters to balance high capacity with the 4-hour time limit.
# Using Lesson 8 (Cite solution_lesson_node_00008) and Lesson 13 (Cite solution_lesson_node_00013).
print("Patching XGBoost parameters for high-capacity execution...")
library.config.XGB_PARAMS.update(
    {
        "n_estimators": 15000,  # Increased to ensure convergence for hard types
        "learning_rate": 0.02,  # Lower LR for better precision
        "early_stopping_rounds": 100,
        "n_jobs": 12,
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 10,  # Cite solution_lesson_node_00017: Depth is critical
    }
)


def main():
    with timer("Full Pipeline Execution"):

        # ==========================================
        # 2. Data Loading
        # ==========================================
        print("\n[Step 1/4] Loading Data...")
        dm = DataManager()
        # We use the full dataset (debug_mode=False) to maximize score,
        # relying on the GPU and optimized params for speed.
        X_train, y_train, X_val, y_val = dm.get_train_data(
            load_cached_data=True, debug_mode=False
        )

        # ==========================================
        # 3. Model Training
        # ==========================================
        print("\n[Step 2/4] Training Stratified Models...")
        model = StratifiedModel()
        model.train(X_train, y_train, X_val, y_val)

        # Free memory before validation to prevent OOM
        del X_train, y_train
        gc.collect()

        # ==========================================
        # 4. Validation & Failure Analysis
        # ==========================================
        print("\n[Step 3/4] Performing Validation and Failure Analysis...")

        # We need to manually generate predictions on the validation set to:
        # 1. Calculate the exact final metric for the threshold check.
        # 2. Correlate errors with features for failure analysis.

        val_preds = []
        val_targets = []
        val_types = []
        val_features_list = []

        # Iterate by coupling type to use the correct stratified model
        for c_type in library.config.COUPLING_TYPES:
            mask = X_val["type"] == c_type
            if not mask.any():
                continue

            # Extract subset for this type
            # We drop 'type' as it is not an input feature for the model
            X_v_type = X_val.loc[mask].drop(columns=["type"])
            y_v_type = y_val.loc[mask]

            # Load the specific trained model
            model_path = os.path.join(library.config.MODEL_SAVE_DIR, f"{c_type}.json")
            if not os.path.exists(model_path):
                print(f"Warning: Model for {c_type} not found.")
                continue

            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(model_path)

            # Ensure inference runs on GPU
            if library.config.XGB_PARAMS.get("device") == "cuda":
                xgb_model.set_params(device="cuda")

            # Predict
            # Note: predict() handles moving data to GPU internally in XGBoost
            y_pred = xgb_model.predict(X_v_type)

            # Collect results
            val_preds.extend(y_pred)
            val_targets.extend(y_v_type.values)
            val_types.extend([c_type] * len(y_v_type))

            # Store features for analysis (keep as dataframe slice)
            val_features_list.append(X_v_type)

            # Cleanup per iteration
            del xgb_model, X_v_type, y_v_type, y_pred
            gc.collect()

        # Convert to arrays for metric calculation
        y_true_all = np.array(val_targets)
        y_pred_all = np.array(val_preds)
        types_all = np.array(val_types)

        # Calculate Final Metric
        final_score, _ = calculate_log_mae(
            y_true_all, y_pred_all, types_all, verbose=True
        )

        # REQUIRED OUTPUT FORMAT
        print(f"Final Validation Metric: {final_score}")

        # Failure Analysis: Feature Correlations
        print("\n--- Failure Analysis ---")
        if val_features_list:
            # Reconstruct feature dataframe
            df_features_val = pd.concat(val_features_list)

            # Calculate Absolute Error
            abs_errors = np.abs(y_true_all - y_pred_all)

            # Calculate correlations between features and absolute error
            # We focus on numerical features
            numeric_cols = df_features_val.select_dtypes(include=[np.number]).columns

            # Compute correlations efficiently
            correlations = {}
            for col in numeric_cols:
                # Handle potential NaN or constant columns safely
                if df_features_val[col].std() > 1e-9:
                    # Calculate correlation
                    corr = np.corrcoef(df_features_val[col].values, abs_errors)[0, 1]
                    correlations[col] = corr
                else:
                    correlations[col] = 0.0

            corr_series = pd.Series(correlations).sort_values(ascending=False)

            print("Top 5 Features associated with HIGH Error (Positive Correlation):")
            print(corr_series.head(5))

            print("\nTop 5 Features associated with LOW Error (Negative Correlation):")
            print(corr_series.tail(5))

            del df_features_val, val_features_list, abs_errors
            gc.collect()

        # ==========================================
        # 5. Submission
        # ==========================================
        print("\n[Step 4/4] Checking Threshold for Submission...")
        THRESHOLD = -1.1285111904144287

        if final_score < THRESHOLD:
            print(
                f"Success! Metric {final_score:.6f} is better than threshold {THRESHOLD:.6f}."
            )
            print("Generating submission for Test Set...")

            # Load Test Data
            X_test, test_ids = dm.get_test_data(load_cached_data=True)

            # Generate Submission
            # This method handles stratification, prediction, and saving to CSV
            model.predict(X_test, test_ids)

        else:
            print(f"Metric {final_score:.6f} did not meet threshold {THRESHOLD:.6f}.")
            print("Submission skipped.")


if __name__ == "__main__":
    main()
