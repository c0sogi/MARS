import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from bisect import bisect_left
import warnings

# Import provided library modules
import library.config as config
from library.utils import seed_everything
from library.train import train_model
from library.inference import generate_submission, predict_ranks
from library.model import ContextAwareRanker
from library.dataset import MarkdownRankDataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def count_inversions(a):
    """
    Counts the number of inversions in a list using bisect (O(n log n)).
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        idx = bisect_left(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def calculate_kendall_tau(df_val_preds, val_metadata_path):
    """
    Calculates the Kendall Tau metric as defined in the task description.
    """
    # Load metadata to get ground truth orders
    df_meta = pd.read_csv(val_metadata_path)

    # Create a lookup for predicted ranks: {nb_id: {cell_id: rank}}
    pred_lookup = {}
    # We expect df_val_preds to have ['id', 'cell_id', 'pred_rank']
    # Grouping by ID for faster access
    grouped_preds = df_val_preds.groupby("id")

    total_swaps = 0
    total_pairs = 0

    for nb_id, row in df_meta.iterrows():
        nb_id = row["id"]
        gt_order_str = row["cell_order"]
        gt_order = gt_order_str.split()
        n = len(gt_order)

        if n <= 1:
            continue

        # Create a map from cell_id to ground truth index (0 to n-1)
        gt_rank_map = {cid: i for i, cid in enumerate(gt_order)}

        # Get predicted ranks for markdown cells
        if nb_id in grouped_preds.groups:
            # Extract the group
            group = grouped_preds.get_group(nb_id)
            md_preds = dict(zip(group["cell_id"], group["pred_rank"]))
        else:
            md_preds = {}

        # Identify code cells: those in GT but not in our markdown predictions
        # Note: This relies on the assumption that val_dataframe contains all markdown cells.
        # Given the preprocessing logic, this is a safe assumption for this dataset.
        code_cells = [cid for cid in gt_order if cid not in md_preds]

        # Assign ranks
        cells_with_ranks = []

        # 1. Code Cells: Equidistant ranks 0..1 based on original order
        if code_cells:
            if len(code_cells) == 1:
                code_ranks = [0.0]
            else:
                code_ranks = np.linspace(0, 1, len(code_cells))

            for cid, r in zip(code_cells, code_ranks):
                cells_with_ranks.append((cid, r))

        # 2. Markdown Cells: Predicted ranks
        for cid, rank in md_preds.items():
            cells_with_ranks.append((cid, rank))

        # Sort by predicted rank to get the predicted order
        cells_with_ranks.sort(key=lambda x: x[1])
        pred_order_ids = [x[0] for x in cells_with_ranks]

        # Map predicted order to ground truth indices
        # If a cell is missing from GT (shouldn't happen), we skip it or handle error.
        # Here we assume data consistency.
        pred_indices = [
            gt_rank_map[cid] for cid in pred_order_ids if cid in gt_rank_map
        ]

        # Calculate swaps (inversions) needed to sort pred_indices to 0..n-1
        swaps = count_inversions(pred_indices)

        total_swaps += swaps
        total_pairs += n * (n - 1)

    if total_pairs == 0:
        return 0.0

    k_tau = 1 - 4 * (total_swaps / total_pairs)
    return k_tau


def main():
    # 1. Setup
    seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Training
    # We use the provided training function.
    # load_cached_data=True allows skipping preprocessing if already done.
    # debug=False ensures we train on the full dataset for best performance.
    print("\n=== Starting Training Phase ===")
    train_model(load_cached_data=False, debug=False)

    # 3. Validation Inference
    print("\n=== Starting Validation Phase ===")

    # Load validation data
    val_parquet_path = os.path.join(config.WORKING_DIR, "val_dataframe.parquet")
    if not os.path.exists(val_parquet_path):
        raise FileNotFoundError(
            "Validation dataframe not found. Training might have failed."
        )

    df_val = pd.read_parquet(val_parquet_path)

    # Initialize model and load weights
    model = ContextAwareRanker(model_name=config.MODEL_NAME).to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model weights not found. Using random weights.")

    # Prepare DataLoader
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    val_dataset = MarkdownRankDataset(df_val, tokenizer, max_len=config.MAX_LEN)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Predict
    print("Running inference on validation set...")
    preds = predict_ranks(model, val_loader, device)
    df_val["pred_rank"] = preds

    # 4. Metric Calculation
    print("Calculating Kendall Tau metric...")
    metric = calculate_kendall_tau(df_val, config.VAL_METADATA_PATH)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error
    df_val["error"] = (df_val["rank"] - df_val["pred_rank"]).abs()

    # Calculate feature lengths
    df_val["text_len"] = df_val["text"].astype(str).apply(len)
    df_val["context_len"] = df_val["context"].astype(str).apply(len)

    # Correlations
    corr_text = df_val["text_len"].corr(df_val["error"])
    corr_context = df_val["context_len"].corr(df_val["error"])
    corr_rank = df_val["rank"].corr(df_val["error"])

    print("Correlation between Error Magnitude and:")
    print(f"  - Markdown Text Length: {corr_text:.4f}")
    print(f"  - Code Context Length:  {corr_context:.4f}")
    print(f"  - True Rank Position:   {corr_rank:.4f}")

    # 6. Submission
    threshold = 0.7453269937267968
    if metric > threshold:
        print(
            f"\nMetric ({metric}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(load_cached_data=True, debug=False)
    else:
        print(f"\nMetric ({metric}) <= Threshold ({threshold}). Skipping submission.")


if __name__ == "__main__":
    main()
