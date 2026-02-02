import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calc_mcc
from library.data_loader import load_metadata
from library.feature_engineering import FeatureEngineer
from library.model import DualStreamModel
from library.train import run_training_pipeline
from library.inference import generate_predictions


def analyze_failures(X, y_true, y_prob, stream_name):
    """
    Performs failure analysis by correlating error magnitude with features.
    """
    if X.empty:
        return

    # Calculate error magnitude
    errors = np.abs(y_true - y_prob)

    # Create a temporary dataframe for correlation
    df_analysis = X.copy()
    df_analysis["error_magnitude"] = errors

    # Calculate correlations
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print(f"\n[{stream_name}] Top 5 Features correlated with Error Magnitude:")
    print(top_correlations)

    # Clean up
    del df_analysis
    import gc

    gc.collect()


def main():
    # 1. Setup and Configuration Override for Fast Baseline
    print("Setting up configuration for fast baseline...")
    seed_everything(Config.SEED)

    # Reduce estimators for speed while keeping GPU acceleration
    Config.XGB_PARAMS["n_estimators"] = 500
    Config.XGB_PARAMS["learning_rate"] = (
        0.05  # Slightly higher LR to compensate for fewer trees
    )

    # 2. Run Training Pipeline
    # This generates features, trains models, optimizes thresholds, and saves artifacts
    print("Starting Training Pipeline...")
    run_training_pipeline(debug_mode=False, load_cached_data=True)

    # 3. Validation and Failure Analysis
    # We need to manually reload validation data to get the exact metric variable
    # and perform failure analysis as requested.
    print("\n" + "=" * 40)
    print("Performing Validation and Failure Analysis")
    print("=" * 40)

    fe = FeatureEngineer()
    model_wrapper = DualStreamModel()

    # Load Validation Metadata
    df_val_meta = load_metadata("validation")

    # Load Optimized Thresholds
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {Config.STREAM_A["name"]: 0.5, Config.STREAM_B["name"]: 0.5}

    val_preds_map = {}

    # --- Stream A Analysis ---
    print("Analyzing Stream A (Interaction)...")
    X_val_a, y_val_a, ids_val_a = fe.create_features(
        metadata_df=df_val_meta,
        stream_config=Config.STREAM_A,
        dataset_type="validation",
        load_cached_data=True,
    )

    if not X_val_a.empty:
        probs_a = model_wrapper.predict_stream(X_val_a, Config.STREAM_A)
        analyze_failures(X_val_a, y_val_a, probs_a, "Stream A")

        for cid, prob in zip(ids_val_a, probs_a):
            val_preds_map[cid] = {"prob": prob, "stream": Config.STREAM_A["name"]}

    # --- Stream B Analysis ---
    print("\nAnalyzing Stream B (Impact)...")
    X_val_b, y_val_b, ids_val_b = fe.create_features(
        metadata_df=df_val_meta,
        stream_config=Config.STREAM_B,
        dataset_type="validation",
        load_cached_data=True,
    )

    if not X_val_b.empty:
        probs_b = model_wrapper.predict_stream(X_val_b, Config.STREAM_B)
        analyze_failures(X_val_b, y_val_b, probs_b, "Stream B")

        for cid, prob in zip(ids_val_b, probs_b):
            val_preds_map[cid] = {"prob": prob, "stream": Config.STREAM_B["name"]}

    # --- Global Metric Calculation ---
    y_true_global = []
    y_pred_global = []

    for _, row in df_val_meta.iterrows():
        cid = row["contact_id"]
        y_true_global.append(int(row["contact"]))

        if cid in val_preds_map:
            info = val_preds_map[cid]
            prob = info["prob"]
            thresh = thresholds.get(info["stream"], 0.5)
            y_pred_global.append(1 if prob >= thresh else 0)
        else:
            y_pred_global.append(0)

    final_mcc = calc_mcc(np.array(y_true_global), np.array(y_pred_global))

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 4. Conditional Inference
    target_score = 0.6938871601521127

    if final_mcc > target_score:
        print(
            f"\nValidation Score ({final_mcc}) > Target ({target_score}). Proceeding to Inference..."
        )
        generate_predictions(debug_mode=False, load_cached_data=True)
    else:
        print(
            f"\nValidation Score ({final_mcc}) <= Target ({target_score}). Skipping Inference."
        )


if __name__ == "__main__":
    main()
