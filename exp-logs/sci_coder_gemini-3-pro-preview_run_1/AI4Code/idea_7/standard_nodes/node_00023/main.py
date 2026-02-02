import os
import sys
import torch
import pandas as pd
import numpy as np
import bisect
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import set_seed, compute_kendall_tau, count_inversions
from library.preprocessor import Preprocessor
from library.dataset import NotebookDataset, custom_collate_fn
from library.model import CorrectedDCAN
from library.engine import train_one_epoch, validate
from library.inference import predict_and_rank, generate_submission_dataframe


def analyze_failures(model, val_loader, df_val_meta, device):
    """
    Performs failure analysis by computing per-notebook metrics and
    correlating them with features.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Get Predictions on Validation Set
    df_scores = predict_and_rank(model, val_loader, device)

    # 2. Reconstruct Orders and Compute Per-Notebook Kendall Tau
    # We need to do this manually here to get individual scores

    # Map predictions: (id, cell_id) -> score
    pred_scores_map = df_scores.set_index(["id", "cell_id"])["rank_score"].to_dict()

    notebook_metrics = []

    # Filter metadata to only those in the validation subset (if debug mode was used)
    valid_ids = set(df_scores["id"].unique())
    df_val_subset = df_val_meta[df_val_meta["id"].isin(valid_ids)].copy()

    for _, row in df_val_subset.iterrows():
        nb_id = row["id"]
        gt_order = row["cell_order"].split()

        # In the ground truth, we need to know which are code and which are MD to sort them
        # However, we don't have cell types in the CSV.
        # We can infer cell types: Code cells are those NOT in the prediction map?
        # No, prediction map only has MD cells.

        # Strategy:
        # 1. Identify MD cells: keys in pred_scores_map for this nb_id
        # 2. Identify Code cells: items in gt_order that are NOT in MD cells list
        # 3. Assign ranks: Code=Index in GT (relative to code only?), MD=Predicted Score

        # Actually, simpler:
        # The metric depends on the relative order of all cells.
        # Code cells are fixed anchors.
        # Let's assume the ground truth string defines the set of cells.

        # Get MD cells for this notebook from predictions
        md_cells_pred = df_scores[df_scores["id"] == nb_id]["cell_id"].tolist()
        md_set = set(md_cells_pred)

        # Identify code cells from GT (those not in MD set)
        code_cells = [c for c in gt_order if c not in md_set]

        # Construct Predicted Rank List
        cell_ranks = []

        # Code cells: Rank = integer index 0, 1, 2...
        # Note: This assumes code cells appear in GT in the correct relative order (0, 1, 2...)
        # which is true by definition of the task.
        for i, cid in enumerate(code_cells):
            cell_ranks.append((cid, float(i)))

        # MD cells: Rank = predicted score
        for cid in md_cells_pred:
            score = pred_scores_map.get((nb_id, cid), 0.0)
            cell_ranks.append((cid, score))

        # Sort to get predicted order
        cell_ranks.sort(key=lambda x: x[1])
        pred_order = [x[0] for x in cell_ranks]

        # Compute Kendall Tau for this notebook
        # 1. Map GT cell IDs to rank
        rank_map = {cid: i for i, cid in enumerate(gt_order)}

        # 2. Get predicted ranks
        # Only consider cells present in both (intersection)
        p_ranks = [rank_map[cid] for cid in pred_order if cid in rank_map]

        n = len(p_ranks)
        if n <= 1:
            kt = 1.0
        else:
            swaps = count_inversions(p_ranks)
            total_pairs = (
                n * (n - 1) // 2
            )  # The metric formula uses n*(n-1) in denominator but 4*swaps.
            # Formula: 1 - 4 * swaps / (n * (n-1))
            # This is equivalent to 1 - 2 * swaps / (n*(n-1)/2)
            kt = 1 - 4 * swaps / (n * (n - 1))

        notebook_metrics.append(
            {
                "id": nb_id,
                "kendall_tau": kt,
                "num_cells": n,
                "num_md": len(md_cells_pred),
                "num_code": len(code_cells),
            }
        )

    df_metrics = pd.DataFrame(notebook_metrics)

    # Correlation Analysis
    if not df_metrics.empty:
        # Error magnitude = 1 - Kendall Tau (higher is worse)
        df_metrics["error"] = 1.0 - df_metrics["kendall_tau"]

        corr_code = df_metrics["error"].corr(df_metrics["num_code"])
        corr_md = df_metrics["error"].corr(df_metrics["num_md"])
        corr_total = df_metrics["error"].corr(df_metrics["num_cells"])

        print(f"Correlation between Error and Num Code Cells: {corr_code:.4f}")
        print(f"Correlation between Error and Num Markdown Cells: {corr_md:.4f}")
        print(f"Correlation between Error and Total Cells: {corr_total:.4f}")

        # Insight
        if corr_md > 0.1:
            print(
                "Insight: Performance degrades as the number of markdown cells increases."
            )
        elif corr_md < -0.1:
            print(
                "Insight: Performance improves as the number of markdown cells increases."
            )
        else:
            print("Insight: Performance is relatively stable across notebook sizes.")
    else:
        print("No metrics computed (empty validation set?).")


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for Fast Baseline Execution
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 15000  # Process 15k notebooks to fit in time limit
    Config.EPOCHS = 5  # Reduced epochs for speed
    Config.BATCH_SIZE = 64

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(
        f"Running Fast Baseline with DEBUG={Config.DEBUG}, Subset={Config.DEBUG_SUBSET_SIZE}, Epochs={Config.EPOCHS}"
    )

    # ==========================================
    # 2. Preprocessing
    # ==========================================
    print("\n--- Step 2: Preprocessing ---")
    preprocessor = Preprocessor()
    preprocessor.process_all(load_cached_data=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("\n--- Step 3: Training ---")

    # Datasets
    train_ds = NotebookDataset(Config.TRAIN_FEATURES_PATH, is_test=False)
    val_ds = NotebookDataset(Config.VAL_FEATURES_PATH, is_test=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = CorrectedDCAN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Metadata for validation
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    best_score = -1.0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, df_val_meta, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Kendall Tau: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ==========================================
    # 4. Final Validation & Metrics
    # ==========================================
    print("\n--- Step 4: Final Validation ---")

    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Compute Final Metric on Full Validation Set (or subset used)
    final_val_score = validate(model, val_loader, df_val_meta, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    analyze_failures(model, val_loader, df_val_meta, device)

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.8315021559000814

    if final_val_score > THRESHOLD:
        print(
            f"\nMetric ({final_val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_ds = NotebookDataset(Config.TEST_FEATURES_PATH, is_test=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict
        df_scores = predict_and_rank(model, test_loader, device)

        # Generate Submission File
        df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        df_submission = generate_submission_dataframe(df_scores, df_test_meta)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_val_score}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    run()
