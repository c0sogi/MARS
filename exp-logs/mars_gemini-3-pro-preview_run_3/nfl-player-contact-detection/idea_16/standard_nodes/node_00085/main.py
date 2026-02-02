import os
import sys
import json
import gc
import warnings
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import setup_seed
from library.train import TrainPipeline
from library.inference import InferencePipeline
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.model import ContactXGB

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_failure_analysis(thresholds):
    """
    Performs failure analysis on the validation set by correlating
    prediction errors with input features.
    """
    print("\n--- Failure Analysis ---")

    # Load validation metadata
    loader = DataLoader(run_mode="validation")
    try:
        # Load merged data. We expect the cache to exist from the training run.
        # Passing empty DataFrames for tracking/helmets as they shouldn't be needed if cache hits.
        df_merged = loader.merge_data(
            loader.load_metadata(),
            pd.DataFrame(),
            pd.DataFrame(),
            load_cached_data=True,
        )
    except Exception as e:
        print(f"Error loading validation data for analysis: {e}")
        return

    streams = ["stream_a", "stream_b"]

    for stream in streams:
        print(f"\nAnalyzing {stream} Failures...")

        # Generate/Load Features
        fg = FeatureGenerator(run_mode="validation")
        try:
            X, y, ids = fg.generate_features(
                df_merged, stream=stream, load_cached_data=True
            )
        except Exception as e:
            print(f"Skipping {stream} due to feature error: {e}")
            continue

        if len(X) == 0:
            print(f"No samples found for {stream}.")
            continue

        # Load IDs (fixing allow_pickle error)
        try:
            # We need to reload ids specifically if they weren't loaded correctly or just rely on X index if aligned
            # But fg.generate_features returns ids.
            pass
        except Exception:
            pass

        # Load Trained Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_{stream}.json")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        # Initialize model wrapper
        if stream == "stream_a":
            params = Config.XGB_PARAMS_STREAM_A
        else:
            params = Config.XGB_PARAMS_STREAM_B

        model = ContactXGB(params)
        model.load(model_path)

        # Predict
        probs = model.predict_proba(X)
        thresh = thresholds.get(stream, 0.5)
        preds = (probs >= thresh).astype(int)

        # Calculate Absolute Error
        errors = np.abs(y - preds)

        # Correlation Analysis
        # We create a temporary DataFrame to leverage pandas correlation
        analysis_df = X.copy()
        analysis_df["error"] = errors

        # Calculate correlation of all features with the error column
        corrs = analysis_df.corr()["error"].drop("error")

        # Identify top 5 features associated with error
        top_corrs = corrs.abs().sort_values(ascending=False).head(5)

        print(f"Top 5 Features correlated with Error in {stream}:")
        for feature, val in top_corrs.items():
            # Print the raw correlation value (sign indicates direction)
            raw_val = corrs[feature]
            print(f"  {feature}: {raw_val:.4f}")

        # Cleanup to manage memory
        del X, y, ids, probs, preds, errors, analysis_df, corrs
        gc.collect()


def main():
    # 1. Setup and Reproducibility
    setup_seed(Config.SEED)

    # 2. Configure for Fast Baseline
    # Override n_estimators to ensure the script completes within the time limit
    # Increased to 3000 to allow convergence (previous run hit 500 limit)
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 3000
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 3000

    # 3. Training Phase
    print("=== Starting Training Phase ===")
    train_pipe = TrainPipeline()
    train_pipe.run_training(load_cached_data=True)

    # 4. Validation Metric Retrieval
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")
    if not os.path.exists(thresholds_path):
        print("Critical Error: Thresholds file not found. Training may have failed.")
        sys.exit(1)

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    global_mcc = thresholds.get("global_mcc", 0.0)

    # Print Metric in Required Format
    print(f"Final Validation Metric: {global_mcc}")

    # 5. Failure Analysis
    run_failure_analysis(thresholds)

    # 6. Submission Logic
    target_metric = 0.6938871601521127

    if global_mcc > target_metric:
        print(f"\nValidation Metric ({global_mcc}) exceeds target ({target_metric}).")
        print("Proceeding to Inference and Submission Generation...")
        inference_pipe = InferencePipeline()
        inference_pipe.run_inference()
    else:
        print(
            f"\nValidation Metric ({global_mcc}) does not exceed target ({target_metric})."
        )
        print("Skipping Submission Generation.")


if __name__ == "__main__":
    main()
