import pandas as pd
import numpy as np
import os
import sys
import json
import gc

from library.config import Config
from library.utils import seed_everything, calc_mcc
from library.orchestrator import Pipeline
from library.data_factory import DataFactory
from library.feature_engine import FeatureEngine
from library.model_trainer import StreamTrainer


def main():
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # 1. Execute Training Pipeline
    # Using debug=False to ensure full training for best performance within time limits
    pipeline = Pipeline()
    pipeline.run_training(debug=False)

    # 2. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    # Initialize factories
    df_factory = DataFactory()
    feat_engine = FeatureEngine()

    # Load Validation Data
    # load_cached_data=True allows using the parquet files created during training
    meta_val, track_val, helmets_val = df_factory.load_dataset(
        mode="validation", load_cached_data=True
    )

    # Split into streams
    meta_val_a, meta_val_b = df_factory.split_contact_ids(meta_val)

    # Generate Features for Validation
    # Note: FeatureEngine handles caching internally
    df_val_a = feat_engine.build_stream_a(
        meta_val_a, track_val, helmets_val, mode="validation"
    )
    df_val_b = feat_engine.build_stream_b(
        meta_val_b, track_val, helmets_val, mode="validation"
    )

    # Clean up raw data to save memory
    del meta_val, track_val, helmets_val, meta_val_a, meta_val_b
    gc.collect()

    # Load Optimized Thresholds
    if os.path.exists(pipeline.thresholds_path):
        with open(pipeline.thresholds_path, "r") as f:
            thresholds = json.load(f)
    else:
        print("Warning: Thresholds file not found. Using defaults.")
        thresholds = {"StreamA": 0.5, "StreamB": 0.5}

    # --- Stream A Inference (Validation) ---
    trainer_a = StreamTrainer("StreamA")
    trainer_a.load_model("model_stream_a.json")
    trainer_a.best_threshold = thresholds.get("StreamA", 0.5)
    # Identify feature columns (excluding metadata/targets)
    trainer_a.feature_cols = trainer_a._get_feature_cols(df_val_a)
    preds_a = trainer_a.predict(df_val_a)

    # --- Stream B Inference (Validation) ---
    trainer_b = StreamTrainer("StreamB")
    trainer_b.load_model("model_stream_b.json")
    trainer_b.best_threshold = thresholds.get("StreamB", 0.5)
    trainer_b.feature_cols = trainer_b._get_feature_cols(df_val_b)
    preds_b = trainer_b.predict(df_val_b)

    # --- Metric Calculation ---
    # Merge predictions with ground truth
    # df_val_a/b contain the 'contact' column (true label)
    res_a = preds_a.merge(
        df_val_a[["contact_id", "contact"]],
        on="contact_id",
        suffixes=("_pred", "_true"),
    )
    res_b = preds_b.merge(
        df_val_b[["contact_id", "contact"]],
        on="contact_id",
        suffixes=("_pred", "_true"),
    )

    # Combine both streams
    full_res = pd.concat([res_a, res_b], axis=0)

    # Calculate MCC
    mcc = calc_mcc(full_res["contact_true"].values, full_res["contact_pred"].values)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {mcc}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    def analyze_errors(df_feat, df_res, stream_name):
        # Calculate absolute error
        df_res["error"] = np.abs(df_res["contact_true"] - df_res["contact_pred"])

        # Merge error back to feature dataframe
        df_analysis = df_feat.merge(df_res[["contact_id", "error"]], on="contact_id")

        # Determine numeric feature columns
        # We use the trainer's exclude list plus 'error' itself
        exclude = trainer_a.exclude_cols + ["error"]
        feat_cols = [c for c in df_analysis.columns if c not in exclude]

        corrs = {}
        for col in feat_cols:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df_analysis[col]):
                # Fill NaNs with 0 for correlation check
                series = df_analysis[col].fillna(0)
                # Avoid constant columns
                if series.std() > 1e-6:
                    corrs[col] = series.corr(df_analysis["error"])

        # Sort by absolute correlation
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"Top Error Correlations for {stream_name}:")
        for k, v in sorted_corrs[:5]:
            print(f"  {k}: {v:.4f}")

    analyze_errors(df_val_a, res_a, "Stream A")
    analyze_errors(df_val_b, res_b, "Stream B")

    # 3. Submission Generation
    TARGET_SCORE = 0.6968

    if mcc > TARGET_SCORE:
        print(
            f"\nMetric ({mcc}) > Threshold ({TARGET_SCORE}). Proceeding to submission..."
        )
        # Run inference on test set
        pipeline.run_inference(debug=False)
    else:
        print(f"\nMetric ({mcc}) <= Threshold ({TARGET_SCORE}). Submission skipped.")


if __name__ == "__main__":
    main()
