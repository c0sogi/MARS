import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config, setup_reproducibility
from library.pipeline_manager import PipelineManager
from library.utils import kendall_tau_metric, format_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    setup_reproducibility(Config.SEED)
    print("Initializing Pipeline...")

    # Initialize Manager
    # Note: Config.DEBUG is False by default, ensuring we use the full dataset for best performance.
    # The training of Ridge and LGBM on ~100k samples is sufficiently fast (< 30 mins).
    manager = PipelineManager()

    # 2. Train Models (Level 1 Ridge + Level 2 LGBM)
    # We use load_cached_data=True to utilize pre-processed parquet files if they exist
    manager.train_stacking_ensemble(load_cached_data=True)

    # 3. Validation Inference & Metric Calculation
    print("\n=== Running Validation Inference ===")

    # Load Validation Data (Markdown and Notebook Context)
    df_val_md, df_val_nb = manager.loader.load_data(split="val", load_cached_data=True)

    # Load Validation Metadata to get Ground Truth Cell Orders (for Anchor logic)
    val_meta_path = Config.VAL_METADATA_PATH
    df_val_meta = pd.read_csv(val_meta_path)

    # Generate Level 1 Predictions (Ridge)
    print("Generating Validation Level 1 Predictions...")
    X_val_l1 = manager.feature_pipeline.transform_level1(df_val_md)
    l1_preds = manager.ridge_model.predict(X_val_l1)

    # Generate Level 2 Features and Predictions (LGBM)
    print("Generating Validation Level 2 Predictions...")
    df_val_l2 = manager.feature_pipeline.transform_level2(
        df_val_md, df_val_nb, level1_preds=l1_preds
    )
    final_preds = manager.lgbm_model.predict(df_val_l2)

    # Attach predictions to markdown dataframe
    df_val_md["pred_rank"] = final_preds

    # Reconstruct Cell Orders for Validation Set (Anchor-Based Sorting)
    print("Reconstructing Validation Cell Orders...")
    val_preds_map = {}

    # Group markdown predictions by notebook ID for fast access
    md_grouped = df_val_md.groupby("id")

    for _, row in df_val_meta.iterrows():
        nb_id = row["id"]
        gt_order_str = row["cell_order"]
        gt_order = gt_order_str.split()

        # Identify Markdown cells for this notebook from our dataframe
        if nb_id in md_grouped.groups:
            md_group = md_grouped.get_group(nb_id)
            md_cell_ids = set(md_group["cell_id"].values)

            # Identify Code cells (Anchors) from GT by excluding Markdown cells
            code_cells = [c for c in gt_order if c not in md_cell_ids]

            # Assign equidistant ranks to code cells
            cells_with_ranks = []
            n_code = len(code_cells)
            if n_code > 0:
                if n_code == 1:
                    cells_with_ranks.append((code_cells[0], 0.0))
                else:
                    for i, cell_id in enumerate(code_cells):
                        r = i / (n_code - 1)
                        cells_with_ranks.append((cell_id, r))

            # Add predicted markdown ranks
            for _, md_row in md_group.iterrows():
                cells_with_ranks.append((md_row["cell_id"], md_row["pred_rank"]))

            # Sort by rank
            cells_with_ranks.sort(key=lambda x: x[1])

            # Create prediction string
            pred_order = " ".join([c[0] for c in cells_with_ranks])
            val_preds_map[nb_id] = pred_order
        else:
            # Fallback if no markdown cells found (unlikely)
            val_preds_map[nb_id] = gt_order_str

    # Compute Kendall Tau
    print("Computing Kendall Tau Metric...")
    score = kendall_tau_metric(df_val_meta, val_preds_map)
    # Print the exact metric format required
    print(f"Final Validation Metric: {score}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude
    df_val_md["error"] = np.abs(df_val_md["rank"] - df_val_md["pred_rank"])

    # We want to correlate error with features in df_val_l2
    # Ensure indices match (they should, as transform_level2 preserves index)
    analysis_df = df_val_l2.copy()
    analysis_df["error"] = df_val_md["error"]

    # Select key features for correlation
    features_to_check = ["char_len", "md_ratio", "pred_ridge"]
    # Add LSA component 0 if available
    if "md_lsa_0" in analysis_df.columns:
        features_to_check.append("md_lsa_0")

    print("Correlation between Error Magnitude and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr, _ = pearsonr(analysis_df[feat], analysis_df["error"])
            print(f"  {feat}: {corr:.4f}")

    # 5. Submission Logic
    THRESHOLD = 0.7453269937267968
    if score > THRESHOLD:
        print(
            f"\nValidation score ({score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        manager.predict_and_sort(load_cached_data=True)
    else:
        print(
            f"\nValidation score ({score}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
