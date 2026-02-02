import sys
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef

# Import from provided libraries
from library.config import Config
from library.pipeline import Pipeline
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Enforce fast baseline execution by limiting training samples
    # 500k samples is a good balance between speed and performance for XGBoost
    Config.DEBUG_SAMPLE_SIZE = 500000

    print("Initializing Pipeline...")
    pipeline = Pipeline()

    # 2. Run Training
    # This will compute features, train the Dual-Stream GBDT, and optimize thresholds
    pipeline.run_training(load_cached_data=True)

    # 3. Validation Assessment
    print("\n=== Performing Validation Assessment ===")

    # Retrieve validation data directly from the feature engineer
    # We use the same cache mechanism
    data_val = pipeline.feature_engineer.process_features(
        "validation", load_cached_data=True
    )

    # Generate probabilities using the trained model
    preds_proba = pipeline.model.predict_proba(data_val)

    # Prepare lists to aggregate results
    all_y_true = []
    all_y_pred = []

    # --- Stream A Evaluation ---
    X_a, y_a, _ = data_val["stream_a"]
    if len(y_a) > 0:
        thresh_a = pipeline.thresholds["stream_a"]
        pred_a_binary = (preds_proba["stream_a"] >= thresh_a).astype(int)
        all_y_true.append(y_a)
        all_y_pred.append(pred_a_binary)

    # --- Stream B Evaluation ---
    X_b, y_b, _ = data_val["stream_b"]
    if len(y_b) > 0:
        thresh_b = pipeline.thresholds["stream_b"]
        pred_b_binary = (preds_proba["stream_b"] >= thresh_b).astype(int)
        all_y_true.append(y_b)
        all_y_pred.append(pred_b_binary)

    # Calculate Final Metric
    if all_y_true:
        y_true_concat = np.concatenate(all_y_true)
        y_pred_concat = np.concatenate(all_y_pred)
        final_mcc = matthews_corrcoef(y_true_concat, y_pred_concat)
    else:
        final_mcc = 0.0

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Analyze Stream A (Interaction Model)
    if len(y_a) > 0:
        print("Stream A (Player-Player) Error Correlations:")
        # Calculate error (0 for correct, 1 for incorrect)
        errors_a = np.abs(y_a - pred_a_binary)

        # Create a dataframe for correlation calculation
        df_analysis_a = X_a.copy()
        df_analysis_a["error"] = errors_a

        # Compute correlation of features with error
        corrs_a = (
            df_analysis_a.corrwith(df_analysis_a["error"])
            .abs()
            .sort_values(ascending=False)
        )

        # Print top 5 correlated features
        print(corrs_a.drop("error").head(5).to_string())

    # Analyze Stream B (Impact Model)
    if len(y_b) > 0:
        print("\nStream B (Player-Ground) Error Correlations:")
        errors_b = np.abs(y_b - pred_b_binary)

        df_analysis_b = X_b.copy()
        df_analysis_b["error"] = errors_b

        corrs_b = (
            df_analysis_b.corrwith(df_analysis_b["error"])
            .abs()
            .sort_values(ascending=False)
        )
        print(corrs_b.drop("error").head(5).to_string())

    # 5. Submission Generation
    TARGET_SCORE = 0.6968

    if final_mcc > TARGET_SCORE:
        print(
            f"\nValidation Score ({final_mcc}) > Target ({TARGET_SCORE}). Generating Submission..."
        )
        pipeline.run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation Score ({final_mcc}) <= Target ({TARGET_SCORE}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
