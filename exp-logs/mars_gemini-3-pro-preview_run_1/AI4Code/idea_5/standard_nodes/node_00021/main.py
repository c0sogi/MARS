import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    compute_kendall_tau,
    get_ordered_cell_ids,
    count_inversions,
)
from library.preprocess import precompute_features
from library.train import train_model
from library.inference import predict
from library.dataset import get_dataloader
from library.model import DCAN


def run_failure_analysis(model, val_loader, gt_map, device):
    """
    Runs inference on validation set, computes detailed metrics,
    and performs failure analysis.
    """
    print("Running failure analysis on validation set...")
    model.eval()

    notebook_stats = []
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            code_features = batch["code_features"].to(device)
            code_mask = batch["code_mask"].to(device)
            markdown_features = batch["markdown_features"].to(device)
            markdown_mask = batch["markdown_mask"].to(device)

            ids = batch["ids"]
            batch_code_ids = batch["code_ids"]
            batch_markdown_ids = batch["markdown_ids"]

            # Get lengths for analysis
            # code_lens is already in batch
            n_code = batch["code_lens"].cpu().numpy()
            # Calculate n_md from mask
            n_md = markdown_mask.sum(dim=1).cpu().numpy()

            # Forward Pass
            logits = model(code_features, code_mask, markdown_features, markdown_mask)

            # Soft Ranking
            probs = torch.softmax(logits, dim=-1)
            L_plus_1 = probs.size(-1)
            indices = torch.arange(L_plus_1, device=device, dtype=torch.float32)
            expected_indices = torch.sum(probs * indices, dim=-1).cpu().numpy()

            # Reconstruct and Evaluate per notebook
            for i, nb_id in enumerate(ids):
                if nb_id not in gt_map:
                    continue

                c_ids = batch_code_ids[i]
                m_ids = batch_markdown_ids[i]
                scores = expected_indices[i][: len(m_ids)]

                # Predict order
                pred_order_str = get_ordered_cell_ids(c_ids, m_ids, scores)
                pred_list = pred_order_str.split()
                true_list = gt_map[nb_id]

                all_preds.append(pred_list)
                all_gts.append(true_list)

                # Compute single instance Kendall Tau
                # We reuse the logic: K = 1 - 4 * swaps / (n*(n-1))
                # If n < 2, K=1
                n = len(true_list)
                if n < 2:
                    kt = 1.0
                else:
                    true_rank = {cid: r for r, cid in enumerate(true_list)}
                    pred_ranks = [
                        true_rank[cid] for cid in pred_list if cid in true_rank
                    ]
                    swaps = count_inversions(pred_ranks)
                    kt = 1 - 4 * (swaps / (n * (n - 1)))

                # Error is 1 - KT (higher is worse)
                error = 1.0 - kt

                notebook_stats.append(
                    {
                        "id": nb_id,
                        "error": error,
                        "num_code": n_code[i],
                        "num_md": n_md[i],
                        "total_cells": n_code[i] + n_md[i],
                    }
                )

    # Compute Global Metric
    final_metric = compute_kendall_tau(all_preds, all_gts)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    df_stats = pd.DataFrame(notebook_stats)

    if len(df_stats) > 1:
        corr_code, _ = pearsonr(df_stats["error"], df_stats["num_code"])
        corr_md, _ = pearsonr(df_stats["error"], df_stats["num_md"])

        print("\nFailure Analysis (Correlation with Error Magnitude):")
        print(f"Correlation with Num Code Cells: {corr_code:.4f}")
        print(f"Correlation with Num Markdown Cells: {corr_md:.4f}")

        # Simple insight
        if corr_code > 0.1:
            print("-> Model struggles with notebooks containing many code cells.")
        if corr_md > 0.1:
            print("-> Model struggles with notebooks containing many markdown cells.")

    return final_metric


def main():
    # 1. Configuration & Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 3  # Limit epochs for speed
    set_seed(Config.SEED)

    print("==== Starting Orchestration Pipeline ====")

    # 2. Preprocessing (Feature Computation)
    # This checks for cached files and computes if missing
    print("\n[Step 1/5] Checking/Computing Features...")
    precompute_features(load_cached_data=True)

    # 3. Training
    print("\n[Step 2/5] Training Model...")
    # train_model saves the best model to Config.MODEL_SAVE_PATH
    train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # 4. Validation & Failure Analysis
    print("\n[Step 3/5] Loading Best Model for Analysis...")
    device = Config.DEVICE
    model = DCAN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Training failed.")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    print("Loading Validation Data...")
    val_loader = get_dataloader(split="val", load_cached_data=True, shuffle=False)

    print("Loading Ground Truth...")
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    gt_map = {row["id"]: row["cell_order"].split() for _, row in val_meta.iterrows()}

    # Run analysis
    final_metric = run_failure_analysis(model, val_loader, gt_map, device)

    # 5. Submission Decision
    print("\n[Step 4/5] Submission Decision...")
    threshold = 0.8315021559000814

    if final_metric > threshold:
        print(
            f"Metric ({final_metric}) > Threshold ({threshold}). Proceeding to submission."
        )
        print("\n[Step 5/5] Generating Submission...")
        predict(load_cached_data=True)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
