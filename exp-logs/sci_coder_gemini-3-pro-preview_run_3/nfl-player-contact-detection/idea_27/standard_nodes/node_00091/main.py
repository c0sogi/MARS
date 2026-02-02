import sys
import os
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import matthews_corrcoef

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.pipeline_manager import PipelineManager
from library.feature_engineering import FeatureEngineer
from library.utils import calc_mcc


def main():
    # 1. Setup and Initialization
    np.random.seed(Config.SEED)

    # Initialize Pipeline Manager
    # We use debug=False to ensure we use the full dataset (with undersampling)
    # to achieve the target metric.
    pm = PipelineManager(debug=False)

    # 2. Training Pipeline
    # Trains models, optimizes thresholds, and saves artifacts.
    # Uses cached features if available to save time.
    print("Starting training pipeline...")
    thresholds = pm.run_training_pipeline(load_cached_data=True)

    # 3. Global Validation & Metric Calculation
    print("\n=== Performing Global Validation ===")
    fe_val = FeatureEngineer(mode="validation", debug=False)

    # Load Validation Data for Stream A (Interaction)
    X_val_A, y_val_A, _ = fe_val.construct_stream_a(load_cached_data=True)

    # Load Validation Data for Stream B (Impact)
    X_val_B, y_val_B, _ = fe_val.construct_stream_b(load_cached_data=True)

    # Generate Predictions
    y_pred_A = np.array([])
    y_prob_A = np.array([])
    if X_val_A.shape[0] > 0 and pm.model_wrapper.model_a:
        y_prob_A = pm.model_wrapper.predict_stream(X_val_A, stream="A")
        y_pred_A = (y_prob_A >= thresholds["A"]).astype(int)

    y_pred_B = np.array([])
    y_prob_B = np.array([])
    if X_val_B.shape[0] > 0 and pm.model_wrapper.model_b:
        y_prob_B = pm.model_wrapper.predict_stream(X_val_B, stream="B")
        y_pred_B = (y_prob_B >= thresholds["B"]).astype(int)

    # Combine predictions from both streams to evaluate global performance
    y_true_all = np.concatenate([y_val_A, y_val_B])
    y_pred_all = np.concatenate([y_pred_A, y_pred_B])

    # Calculate and Print Final Metric
    final_mcc = calc_mcc(y_true_all, y_pred_all)
    print(f"Final Validation Metric: {final_mcc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    def analyze_errors(X, y_true, y_prob, stream_name):
        if len(y_true) == 0:
            return

        # Calculate error magnitude
        errors = np.abs(y_true - y_prob)

        # Calculate correlation between features and error
        if isinstance(X, pd.DataFrame):
            # Sampling for speed if dataset is very large
            if len(X) > 100000:
                indices = np.random.choice(len(X), 100000, replace=False)
                X_sample = X.iloc[indices]
                errors_sample = errors[indices]
            else:
                X_sample = X
                errors_sample = errors

            correlations = {}
            numeric_cols = X_sample.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                try:
                    # Handle potential NaNs by filling with 0 for correlation check
                    corr = np.corrcoef(X_sample[col].fillna(0), errors_sample)[0, 1]
                    if not np.isnan(corr):
                        correlations[col] = corr
                except Exception:
                    continue

            # Sort by absolute correlation
            sorted_corr = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )
            print(f"\nTop Feature Correlations with Error (Stream {stream_name}):")
            for name, val in sorted_corr[:5]:
                print(f"  {name}: {val:.4f}")
        else:
            print(
                f"Skipping detailed feature correlation for Stream {stream_name} (X is not DataFrame)"
            )

    analyze_errors(X_val_A, y_val_A, y_prob_A, "A")
    analyze_errors(X_val_B, y_val_B, y_prob_B, "B")

    # Clean up validation memory
    del X_val_A, y_val_A, X_val_B, y_val_B, y_prob_A, y_prob_B
    gc.collect()

    # 5. Submission Generation
    # Only generate submission if the model meets the performance requirement
    if final_mcc > 0.6968:
        print("\nMetric threshold met. Generating submission...")
        pm.run_inference_pipeline(thresholds=thresholds, load_cached_data=True)
    else:
        print(f"\nMetric {final_mcc} <= 0.6968. Skipping submission generation.")


if __name__ == "__main__":
    main()
