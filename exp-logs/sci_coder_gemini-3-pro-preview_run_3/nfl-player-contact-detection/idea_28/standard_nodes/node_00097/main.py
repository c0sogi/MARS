import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
from sklearn.metrics import matthews_corrcoef

# Import from provided libraries
from library.config import XGB_PARAMS_STREAM_A, XGB_PARAMS_STREAM_B, SEED
from library.utils import seed_everything
from library.data_manager import DataManager
from library.feature_builder import FeatureBuilder
from library.model_trainer import ModelTrainer


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Starting orchestration...")

    # 2. Data Loading (Train & Validation)
    # We load both train and validation data to perform training and threshold optimization
    dm_train = DataManager(mode="train")
    df_train_a, df_train_b = dm_train.load_data(load_cached=True)

    dm_val = DataManager(mode="validation")
    df_val_a, df_val_b = dm_val.load_data(load_cached=True)

    # 3. Feature Building
    # Build features for Stream A and Stream B for both train and validation sets
    fb_train = FeatureBuilder(mode="train")
    data_train_a, data_train_b = fb_train.build_features(
        df_train_a, df_train_b, load_cached=True
    )

    fb_val = FeatureBuilder(mode="validation")
    data_val_a, data_val_b = fb_val.build_features(df_val_a, df_val_b, load_cached=True)

    # 4. Model Training
    # Initialize trainer
    trainer = ModelTrainer()

    # Train models and get optimized thresholds
    # The trainer handles undersampling internally
    models, thresholds = trainer.train_and_evaluate(
        data_train_a, data_val_a, data_train_b, data_val_b
    )

    # 5. Validation & Metric Calculation
    print("\n--- Final Validation ---")

    # We need to compute the combined MCC on the validation set
    y_true_all = []
    y_pred_all = []

    # Stream A Validation Inference
    if models.get("A") and not data_val_a["X"].empty:
        dval_a = xgb.DMatrix(data_val_a["X"])
        probs_a = models["A"].predict(dval_a)
        preds_a = (probs_a >= thresholds["A"]).astype(int)

        y_true_all.append(data_val_a["y"])
        y_pred_all.append(preds_a)

        # Failure Analysis Stream A
        print("\nFailure Analysis - Stream A (Top 5 Correlated Features with Error):")
        residuals_a = np.abs(data_val_a["y"] - probs_a)
        # Handle -999 in features by replacing with NaN for correlation calc
        X_val_a_corr = data_val_a["X"].replace(-999, np.nan)
        correlations_a = X_val_a_corr.corrwith(
            pd.Series(residuals_a, index=X_val_a_corr.index)
        )
        print(correlations_a.abs().sort_values(ascending=False).head(5).to_string())

    # Stream B Validation Inference
    if models.get("B") and not data_val_b["X"].empty:
        dval_b = xgb.DMatrix(data_val_b["X"])
        probs_b = models["B"].predict(dval_b)
        preds_b = (probs_b >= thresholds["B"]).astype(int)

        y_true_all.append(data_val_b["y"])
        y_pred_all.append(preds_b)

        # Failure Analysis Stream B
        print("\nFailure Analysis - Stream B (Top 5 Correlated Features with Error):")
        residuals_b = np.abs(data_val_b["y"] - probs_b)
        correlations_b = data_val_b["X"].corrwith(
            pd.Series(residuals_b, index=data_val_b["X"].index)
        )
        print(correlations_b.abs().sort_values(ascending=False).head(5).to_string())

    # Combine and Calculate Metric
    if y_true_all:
        y_true_combined = np.concatenate(y_true_all)
        y_pred_combined = np.concatenate(y_pred_all)

        final_mcc = matthews_corrcoef(y_true_combined, y_pred_combined)
        print(f"\nFinal Validation Metric: {final_mcc}")
    else:
        final_mcc = 0.0
        print("\nFinal Validation Metric: 0.0")

    # 6. Submission
    # Generate submission if metric is good enough
    if final_mcc > 0.6968:
        print("\nMetric threshold passed. Generating submission...")

        # Load Test Data
        dm_test = DataManager(mode="test")
        df_test_a, df_test_b = dm_test.load_data(load_cached=True)

        # Build Test Features
        fb_test = FeatureBuilder(mode="test")
        data_test_a, data_test_b = fb_test.build_features(
            df_test_a, df_test_b, load_cached=True
        )

        # Generate Submission
        trainer.generate_submission(models, thresholds, data_test_a, data_test_b)
    else:
        print(f"\nMetric {final_mcc} <= 0.6968. Skipping submission generation.")


if __name__ == "__main__":
    main()
