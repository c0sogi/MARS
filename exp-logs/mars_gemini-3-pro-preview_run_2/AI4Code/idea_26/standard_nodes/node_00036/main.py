import sys
import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.train_pipeline import TrainPipeline
from library.inference_pipeline import InferencePipeline
from library.utils import compute_kendall_tau, set_seed
from library.feature_extraction import DualViewFeaturePipeline
from library.model_factory import Stage1Ridge, Stage2LGBM
from library.data_processing import NotebookProcessor


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # --------------------------------------------------------------------------
    # Reduce computational burden to ensure completion within 2 hours
    Config.LGBM_PARAMS["n_estimators"] = 800
    Config.LGBM_PARAMS["early_stopping_rounds"] = 50

    # Check for GPU and enable if available (for LightGBM)
    if torch.cuda.is_available():
        print("GPU detected. Configuring LightGBM to use GPU.")
        # Note: This assumes the installed LightGBM supports GPU.
        # If not, it might fallback or error. Given the environment constraints,
        # we'll stick to default (CPU) or minimal changes to avoid crashes,
        # as 140k rows is manageable on CPU.
        # Config.LGBM_PARAMS["device"] = "gpu"

    # --------------------------------------------------------------------------
    # 2. Training Pipeline
    # --------------------------------------------------------------------------
    print(">>> Initializing and Running Training Pipeline...")
    train_pipe = TrainPipeline()
    # Run training (loads data, extracts features, trains models, saves artifacts)
    train_pipe.run(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n>>> Performing Validation and Metric Calculation...")

    # Load Validation Data
    # We must sort by ID to ensure alignment with extract_features output (which groups by ID)
    processor = NotebookProcessor()
    df_val = processor.load_val_data(load_cached_data=True)
    df_val = df_val.sort_values("id").reset_index(drop=True)

    # Load Validation Metadata (Ground Truth)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize components to load models/features
    feat_pipeline = DualViewFeaturePipeline()
    stage1 = Stage1Ridge()
    stage2 = Stage2LGBM()

    # Load Vectorizers (TF-IDF / SVD)
    feat_pipeline._load_models()

    # Filter for Markdown cells (Targets)
    df_val_md = df_val[df_val["cell_type"] == "markdown"].reset_index(drop=True)

    # --- Stage 1 Prediction ---
    # Vectorize validation text
    val_source = df_val_md["source"].astype(str).fillna("")
    X_val_sparse = feat_pipeline.tfidf.transform(val_source)

    # Predict using Ridge
    pred_s1 = stage1.predict(X_val_sparse)

    # --- Stage 2 Prediction ---
    # Load cached features.
    # Note: extract_features saves/loads based on mode='val'.
    # Since we sorted df_val by ID, and extract_features output is sorted by ID, they align.
    # We reload features to ensure we have the exact matrix used/generated.
    df_val_feats = feat_pipeline.extract_features(
        df_val, mode="val", load_cached_data=True
    )

    # Prepare Stage 2 Input Matrix
    exclude_cols = ["id", "cell_id", "ancestor_id", "pct_rank"]
    feature_cols = [c for c in df_val_feats.columns if c not in exclude_cols]
    X_val_s2_base = df_val_feats[feature_cols].values

    # Stack Stage 1 predictions
    X_val_final = np.column_stack([X_val_s2_base, pred_s1])

    # Predict using LightGBM
    pred_s2 = stage2.predict(X_val_final)

    # Assign predictions back to MD dataframe
    df_val_md["pred_rank"] = pred_s2

    # --- Reconstruct Cell Orders ---
    # Create a lookup map: (id, cell_id) -> predicted_rank
    pred_map = dict(
        zip(zip(df_val_md["id"], df_val_md["cell_id"]), df_val_md["pred_rank"])
    )

    prediction_rows = []

    # Iterate over notebooks to sort cells
    # df_val contains both code and md cells
    for nb_id, group in df_val.groupby("id"):
        # Code cells: Fixed rank based on position (pct_rank is already computed in processor)
        code_cells = group[group["cell_type"] == "code"].copy()
        code_cells["final_rank"] = code_cells["pct_rank"]

        # Markdown cells: Use predicted rank
        md_cells = group[group["cell_type"] == "markdown"].copy()
        # Fallback to 0.5 if not found (should not happen)
        md_ranks = [pred_map.get((nb_id, cid), 0.5) for cid in md_cells["cell_id"]]
        md_cells["final_rank"] = md_ranks

        # Combine and Sort
        combined = pd.concat([code_cells, md_cells]).sort_values("final_rank")

        # Create space-delimited string
        cell_order = " ".join(combined["cell_id"].astype(str).tolist())
        prediction_rows.append({"id": nb_id, "cell_order": cell_order})

    df_preds = pd.DataFrame(prediction_rows)

    # --- Compute Kendall Tau ---
    kt_score = compute_kendall_tau(df_val_meta, df_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {kt_score}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n>>> Performing Failure Analysis...")

    # Calculate Absolute Error
    # GT is in df_val_feats['pct_rank'] (which aligns with pred_s2)
    # or df_val_md['pct_rank']
    y_true = df_val_md["pct_rank"].values
    abs_error = np.abs(y_true - pred_s2)

    # Correlate Error with Features
    # We use df_val_feats which contains the numerical features
    analysis_df = df_val_feats[feature_cols].copy()
    analysis_df["abs_error"] = abs_error

    correlations = {}
    for col in feature_cols:
        if analysis_df[col].std() > 1e-9:  # Avoid constant columns
            corr, _ = pearsonr(analysis_df[col], analysis_df["abs_error"])
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    # Sort by magnitude of correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error Magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if kt_score > THRESHOLD:
        print(f"\nMetric {kt_score} > Threshold {THRESHOLD}. Generating Submission...")
        inference = InferencePipeline()
        inference.predict_test_set(load_cached_data=True)
    else:
        print(f"\nMetric {kt_score} <= Threshold {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    main()
