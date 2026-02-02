import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import warnings

# Import from provided library
from library.config import Config
from library.data_loader import load_metadata, load_notebook
from library.model import PositionRegressor
from library.utils import compute_kendall_tau

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def reconstruct_orders(df_meta, df_preds):
    """
    Reconstructs the cell order for validation notebooks based on predictions.
    Mirrors the logic in PositionRegressor.generate_submission but returns a DataFrame.
    """
    # Group predictions by notebook ID
    if not df_preds.empty:
        pred_groups = df_preds.groupby("id")
    else:
        pred_groups = None

    results = []

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        file_path = row["file_path"]

        try:
            nb_data = load_notebook(file_path)
        except Exception:
            continue

        # In validation, we use the ground truth code cells (anchors)
        # We need to know which cells are code cells.
        # load_notebook returns dicts.
        # For validation reconstruction, we assume the code cells are those present in the 'code_cells' dict.
        # Their relative order is fixed (0, 1, 2...).

        # Note: In the provided library logic, for train/val, get_ordered_cells is used to find code order.
        # However, for reconstruction based on 'pred_pos', we treat code cells as fixed anchors
        # at integer indices 0, 1, 2... and insert markdown cells based on predicted relative pos.

        # We need the correct order of code cells to assign them ranks 0, 1, 2...
        # In the provided dataset, code cells in the JSON are already in correct order.
        code_cells = list(nb_data["code_cells"].keys())
        n_code = len(code_cells)

        cells_with_ranks = []

        # Assign integer ranks to code cells
        for i, cid in enumerate(code_cells):
            cells_with_ranks.append((float(i), cid))

        # Assign predicted ranks to markdown cells
        if pred_groups is not None and nb_id in pred_groups.groups:
            md_df = pred_groups.get_group(nb_id)
            for _, r in md_df.iterrows():
                cid = r["cell_id"]
                pred_rel = r["pred_pos"]

                # Clip and scale
                pred_rel = max(0.0, min(1.0, pred_rel))
                rank = pred_rel * n_code

                cells_with_ranks.append((rank, cid))

        # Sort by rank
        cells_with_ranks.sort(key=lambda x: x[0])

        # Create string
        ordered_ids = [x[1] for x in cells_with_ranks]
        cell_order_str = " ".join(ordered_ids)

        results.append({"id": nb_id, "cell_order": cell_order_str})

    return pd.DataFrame(results)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing Notebook Cell Ordering Pipeline...")

    # Check for GPU (mostly for reporting, as LightGBM config is pre-set in library)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, running on CPU.")

    # 2. Load Metadata
    # We sample the training set to ensure the baseline runs quickly (within 2 hours).
    # 20,000 samples is robust enough for a baseline while keeping feature extraction time low.
    TRAIN_SAMPLE = 20000
    print(f"Loading metadata (Train sample: {TRAIN_SAMPLE})...")

    df_train = load_metadata("train", sample_size=TRAIN_SAMPLE)
    df_val = load_metadata("val", sample_size=None)  # Use full validation set
    df_test = load_metadata("test")

    # 3. Train Model
    print("-" * 40)
    print("Starting Model Training")
    print("-" * 40)

    # Initialize model
    model = PositionRegressor()

    # Train
    # This handles vectorizer fitting and feature generation internally
    model.train(df_train, df_val)

    # 4. Validation Inference & Metric
    print("-" * 40)
    print("Validating Model")
    print("-" * 40)

    # Predict on validation set
    val_preds = model.predict(df_val)

    # Reconstruct orders to calculate Kendall Tau
    print("Reconstructing validation cell orders...")
    df_val_pred_orders = reconstruct_orders(df_val, val_preds)

    # Compute metric
    kt_score = compute_kendall_tau(df_val_pred_orders, df_val)
    print(f"Final Validation Metric: {kt_score}")

    # 5. Failure Analysis
    print("-" * 40)
    print("Failure Analysis")
    print("-" * 40)

    # Retrieve validation features (cached) to get targets and metadata
    # We use the extractor directly to load the dataframe
    df_val_features = model.extractor.generate_dataset(
        df_val, mode="val", load_cached_data=True
    )

    if not df_val_features.empty and not val_preds.empty:
        # Merge predictions with features
        # val_preds has [id, cell_id, pred_pos]
        # df_val_features has [id, cell_id, target, n_code, md_len, sim_mean, ...]

        analysis_df = pd.merge(
            df_val_features,
            val_preds[["id", "cell_id", "pred_pos"]],
            on=["id", "cell_id"],
            how="inner",
        )

        # Calculate absolute error
        analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["pred_pos"])

        # Select features for correlation
        corr_cols = [
            "error",
            "n_code",
            "md_len",
            "sim_mean",
            "sim_max",
            "sim_std",
            "center_of_mass",
        ]
        # Filter cols that exist
        corr_cols = [c for c in corr_cols if c in analysis_df.columns]

        correlations = (
            analysis_df[corr_cols].corr()["error"].sort_values(ascending=False)
        )

        print("Correlation of features with Prediction Error:")
        print(correlations)

        print("\nTop 5 Worst Predictions (by Error):")
        worst_preds = analysis_df.sort_values("error", ascending=False).head(5)
        for _, row in worst_preds.iterrows():
            print(
                f"ID: {row['id']}, Cell: {row['cell_id']}, Target: {row['target']:.4f}, Pred: {row['pred_pos']:.4f}, Error: {row['error']:.4f}"
            )
    else:
        print("Skipping failure analysis (no data available).")

    # 6. Generate Submission
    print("-" * 40)
    print("Generating Submission")
    print("-" * 40)

    model.generate_submission(df_test)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
