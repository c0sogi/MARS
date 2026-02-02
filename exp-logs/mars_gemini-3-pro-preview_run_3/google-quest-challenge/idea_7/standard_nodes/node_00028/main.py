import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# ------------------------------------------------------------------------------
# 1. Configuration & Monkey Patching for Fast Baseline
# ------------------------------------------------------------------------------
from library.config import Config

# Modify Config for fast execution as requested
Config.EPOCHS = 2
Config.N_FOLDS = 2

# Import library modules after patching Config
from library.utils import set_seed, compute_spearman, load_numpy_array
from library.dapt import run_dapt
from library.fine_tuning import run_fine_tuning
from library.stacking import (
    train_meta_stacker,
    TopologyAwareRidge,
    split_features_l2,
    get_artifact_path,
)


def main():
    # Set global seed
    set_seed(Config.SEED)

    print("=== Starting End-to-End Pipeline ===")

    # --------------------------------------------------------------------------
    # 2. Domain Adaptive Pre-Training (DAPT)
    # --------------------------------------------------------------------------
    print("\n[Step 1/4] Running Domain Adaptation...")
    # This will train MLM on the combined text corpus
    run_dapt(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Supervised Fine-Tuning & Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[Step 2/4] Running Fine-Tuning and Feature Extraction...")

    # Stream 1: DeBERTa (using DAPT weights)
    print(">> Processing Stream 1: DeBERTa")
    run_fine_tuning(
        model_alias="deberta",
        base_model_name=Config.MODEL_DEBERTA,
        dapt_path=Config.DAPT_MODEL_OUTPUT_PATH,
        load_cached_data=True,
    )

    # Stream 2: MPNet (using base weights)
    print(">> Processing Stream 2: MPNet")
    run_fine_tuning(
        model_alias="mpnet",
        base_model_name=Config.MODEL_MPNET,
        dapt_path=None,  # No DAPT for MPNet in this config
        load_cached_data=True,
    )

    # --------------------------------------------------------------------------
    # 4. Stacking & Submission Generation
    # --------------------------------------------------------------------------
    print("\n[Step 3/4] Running Stacking Ensemble...")
    model_aliases = ["deberta", "mpnet"]

    # This trains L1 and L2 models and generates submission.csv
    train_meta_stacker(model_aliases, load_cached_preds=True)

    # --------------------------------------------------------------------------
    # 5. Validation Assessment & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n[Step 4/4] Validation Assessment & Failure Analysis...")

    # A. Reproduce Holdout Predictions for Analysis
    # We need to load the trained meta-model and the L1 holdout predictions
    meta_model_path = get_artifact_path("meta_stacker.joblib")
    meta_model = TopologyAwareRidge.load(meta_model_path)

    l1_holdout_list = []
    for alias in model_aliases:
        path = f"{alias}_l1_holdout_preds.npy"
        preds = load_numpy_array(path)
        if preds is None:
            raise FileNotFoundError(f"Could not find L1 holdout preds for {alias}")
        l1_holdout_list.append(preds)

    X_q_holdout, X_full_holdout = split_features_l2(l1_holdout_list)

    # Generate predictions on holdout set
    val_preds = meta_model.predict(X_q_holdout, X_full_holdout)

    # Load Ground Truth
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_targets = val_df[Config.TARGET_COLS].values

    # B. Compute Final Metric
    final_metric = compute_spearman(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # C. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per row
    row_mae = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Extract features for correlation analysis
    # We analyze correlation between Error Magnitude and Text Lengths
    val_df["q_body_len"] = val_df["question_body"].fillna("").str.len()
    val_df["q_title_len"] = val_df["question_title"].fillna("").str.len()
    val_df["a_len"] = val_df["answer"].fillna("").str.len()

    features_to_analyze = ["q_body_len", "q_title_len", "a_len"]

    print("Correlation between Error Magnitude (MAE) and Input Features:")
    for feat in features_to_analyze:
        feat_values = val_df[feat].values
        # Handle cases with constant values
        if np.std(feat_values) == 0 or np.std(row_mae) == 0:
            corr = 0.0
        else:
            corr, _ = spearmanr(feat_values, row_mae)
        print(f"  {feat}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission Threshold Check
    # --------------------------------------------------------------------------
    THRESHOLD = 0.40698660691461275

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Keeping submission."
        )
        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"Submission available at: {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Discarding submission."
        )
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
