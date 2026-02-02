import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, set_seed
from library.data_preprocessing import Preprocessor
from library.train import Trainer
from library.inference import run_inference
from library.utils import count_inversions
from library.dataset import CachedNotebookDataset
from torch.utils.data import DataLoader


def calculate_per_notebook_metrics(df_preds, df_gt):
    """
    Computes Kendall Tau score for each notebook individually.
    Returns a DataFrame with 'id' and 'score'.
    """
    df_merged = pd.merge(
        df_gt[["id", "cell_order"]],
        df_preds[["id", "cell_order"]],
        on="id",
        suffixes=("_gt", "_pred"),
    )

    results = []

    for _, row in df_merged.iterrows():
        nb_id = row["id"]
        gt_order = row["cell_order_gt"].split()
        pred_order = row["cell_order_pred"].split()

        n = len(gt_order)
        if n <= 1:
            results.append({"id": nb_id, "score": 1.0})
            continue

        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        pred_ranks = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                pred_ranks.append(gt_rank_map[cell_id])

        swaps = count_inversions(pred_ranks)
        total_possible = (
            n * (n - 1) // 2
        )  # Note: The metric formula uses n(n-1) in denominator but multiplies swaps by 4.
        # Metric: 1 - 4 * (Swaps / (n * (n-1)))
        # Which simplifies to 1 - 2 * Swaps / (n * (n-1)/2)

        # Using the competition formula exactly:
        term = 4 * swaps / (n * (n - 1))
        score = 1 - term
        results.append({"id": nb_id, "score": score})

    return pd.DataFrame(results)


def perform_failure_analysis(df_preds, df_gt):
    """
    Analyzes the correlation between model error and input features.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate per-notebook performance
    df_scores = calculate_per_notebook_metrics(df_preds, df_gt)

    # 2. Load features to get metadata (num_md, num_code)
    # We read the validation features parquet file
    if not os.path.exists(Config.VAL_FEATURES_PATH):
        print("Validation features not found, skipping detailed feature analysis.")
        return

    df_features = pd.read_parquet(Config.VAL_FEATURES_PATH)

    # Group by ID to get counts
    # count markdown cells
    md_counts = (
        df_features[df_features["cell_type"] == "markdown"]
        .groupby("id")
        .size()
        .reset_index(name="num_md")
    )
    # count code cells
    code_counts = (
        df_features[df_features["cell_type"] == "code"]
        .groupby("id")
        .size()
        .reset_index(name="num_code")
    )

    # Merge stats
    df_stats = pd.merge(df_scores, md_counts, on="id", how="left").fillna(0)
    df_stats = pd.merge(df_stats, code_counts, on="id", how="left").fillna(0)

    # Define Error Magnitude (1 - Score)
    # Lower score = Higher Error
    df_stats["error_magnitude"] = 1.0 - df_stats["score"]

    # Calculate Correlations
    # Correlation between Error and Num Markdown Cells
    corr_md, _ = pearsonr(df_stats["error_magnitude"], df_stats["num_md"])

    # Correlation between Error and Num Code Cells
    corr_code, _ = pearsonr(df_stats["error_magnitude"], df_stats["num_code"])

    print(f"Correlation between Error Magnitude and Num Markdown Cells: {corr_md:.4f}")
    print(f"Correlation between Error Magnitude and Num Code Cells: {corr_code:.4f}")

    if corr_md > 0.1:
        print(
            "Observation: The model tends to make more errors on notebooks with many markdown cells."
        )
    elif corr_md < -0.1:
        print(
            "Observation: The model tends to make fewer errors on notebooks with many markdown cells."
        )

    if corr_code > 0.1:
        print(
            "Observation: The model tends to make more errors on notebooks with many code cells."
        )


def main():
    # 1. Configuration & Setup
    # Limit epochs to ensure fast baseline execution within time limits
    Config.NUM_EPOCHS = 2
    set_seed(Config.SEED)

    print(f"Starting execution with Config.NUM_EPOCHS={Config.NUM_EPOCHS}")

    # 2. Preprocessing
    # Generate or load features
    preprocessor = Preprocessor()
    preprocessor.run(load_cached_data=True)

    # 3. Training
    trainer = Trainer()
    trainer.fit()

    # 4. Validation & Metric Calculation
    print("\nRunning Final Validation...")

    # Load the best model weights for validation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    # Prepare Validation Loader
    val_dataset = CachedNotebookDataset(Config.VAL_FEATURES_PATH, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=CachedNotebookDataset.collate_fn,
        pin_memory=True,
    )

    # Load Ground Truth
    df_val_gt = pd.read_csv(Config.VAL_METADATA_PATH)

    # Validate
    score, df_preds = trainer.validate(val_loader, df_val_gt)

    # Print Metric (Full Precision)
    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    perform_failure_analysis(df_preds, df_val_gt)

    # 6. Submission
    # Threshold check
    THRESHOLD = 0.8315021559000814

    if score > THRESHOLD:
        print(
            f"\nValidation score ({score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nValidation score ({score}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
