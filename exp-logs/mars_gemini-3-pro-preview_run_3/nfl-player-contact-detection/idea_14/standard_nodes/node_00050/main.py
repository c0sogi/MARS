import sys
import os
import pandas as pd
import numpy as np

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.workflow_manager import WorkflowManager
from library.utils import set_seed, compute_mcc
from library.data_manager import DataManager


def main():
    # 1. Setup and Configuration Overrides
    set_seed(Config.SEED)

    # Optimize for "Fast Baseline" and "Silent Execution"
    Config.NUM_BOOST_ROUND = 1000  # Limit training steps for speed
    Config.VERBOSE_EVAL = 0  # Suppress training logs

    # Ensure GPU usage and silence XGBoost internal logging
    for params in [Config.XGB_PARAMS_STREAM_A, Config.XGB_PARAMS_STREAM_B]:
        params["device"] = "cuda"
        params["verbosity"] = 0

    # Initialize Workflow
    wm = WorkflowManager()

    # 2. Train Stream A (Interaction Model)
    # Returns: model, threshold, train_oof (for Stream B context), val_preds
    model_a, thresh_a, train_oof_a, val_preds_a = wm.train_interaction_stream(
        load_cached_data=True
    )

    # 3. Train Stream B (Impact Model)
    # Uses train_oof_a for training context and val_preds_a for validation context
    model_b, thresh_b = wm.train_impact_stream(
        train_oof_a, val_preds_a, load_cached_data=True
    )

    # 4. Global Validation & Metric Calculation
    print("\n=== Global Validation ===")

    # Load validation ground truth
    df_val_meta = wm.data_manager.load_data("validation", load_cached_data=True)

    # We need to generate Stream B validation predictions explicitly to compute the global metric
    # (train_impact_stream optimizes internally but returns the model, not the preds df)
    X_val_b, y_val_b, ids_val_b = wm.feature_builder.build_stream_b_features(
        df_val_meta, val_preds_a, load_cached_data=True, split="validation"
    )

    # Predict Stream B
    val_probs_b = model_b.predict(X_val_b)
    val_preds_b = pd.DataFrame({"contact_id": ids_val_b, "prob": val_probs_b})

    # Apply optimized thresholds
    val_preds_a["pred"] = (val_preds_a["prob"] >= thresh_a).astype(int)
    val_preds_b["pred"] = (val_preds_b["prob"] >= thresh_b).astype(int)

    # Combine predictions from both streams
    # Stream A handles Player-Player, Stream B handles Player-Ground
    all_preds = pd.concat(
        [val_preds_a[["contact_id", "pred"]], val_preds_b[["contact_id", "pred"]]]
    )

    # Merge with Ground Truth to ensure alignment
    val_merged = pd.merge(
        df_val_meta[["contact_id", "contact"]], all_preds, on="contact_id", how="inner"
    )

    # Compute Final Metric
    final_mcc = compute_mcc(val_merged["contact"], val_merged["pred"])
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load Stream A features for correlation analysis
    X_val_a, y_val_a, ids_val_a = wm.feature_builder.build_stream_a_features(
        df_val_meta, load_cached_data=True, split="validation"
    )

    def analyze_errors(X, y_true, y_pred, stream_name):
        # Calculate binary error magnitude
        errors = np.abs(y_true - y_pred)

        # Sample if dataset is too large for correlation computation
        if len(errors) > 50000:
            idx = np.random.choice(len(errors), 50000, replace=False)
            X_sample = X.iloc[idx].copy()
            errors_sample = errors[idx]
        else:
            X_sample = X.copy()
            errors_sample = errors

        # Compute correlations
        corrs = X_sample.corrwith(pd.Series(errors_sample, index=X_sample.index))
        print(f"Top 5 Features correlated with Error ({stream_name}):")
        print(corrs.abs().sort_values(ascending=False).head(5))

    # Analyze Stream A
    analyze_errors(X_val_a, y_val_a, val_preds_a["pred"].values, "Stream A")

    # Analyze Stream B
    analyze_errors(X_val_b, y_val_b, val_preds_b["pred"].values, "Stream B")

    # 6. Submission Generation
    THRESHOLD_SCORE = 0.6938871601521127

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_mcc}) > Threshold ({THRESHOLD_SCORE}). Generating Submission..."
        )
        wm.run_inference_cascade(
            model_a, thresh_a, model_b, thresh_b, load_cached_data=True
        )
    else:
        print(
            f"\nMetric ({final_mcc}) <= Threshold ({THRESHOLD_SCORE}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
