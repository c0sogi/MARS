import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# 1. Import Config and Override for Fast Baseline
from library.config import Config

# Override Configuration to meet the 2-hour runtime constraint
# We use a subset of data (DEBUG mode) but large enough to attempt the threshold.
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 25000  # Process 25k notebooks (~15-20 mins on A100)
Config.NUM_EPOCHS = 5  # Sufficient for MPNet adaptation
Config.BATCH_SIZE = 64
Config.WARMUP_STEPS = 0  # Ensure no warmup as per design

# 2. Import Library Modules (after Config override)
from library.data_preprocessor import DataPreprocessor
from library.train import train_model, calculate_kendall_tau_components
from library.inference import predict_and_sort
from library.dataset import NotebookEmbeddingDataset
from library.model import DualContextAnchorNetwork


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_validation_and_failure_analysis():
    """
    Reloads the best model, computes the final validation metric,
    and correlates errors with input features.
    """
    print("\n=== Starting Validation & Failure Analysis ===")
    device = Config.DEVICE

    # Load Validation Data
    val_dataset = NotebookEmbeddingDataset(
        split="val", max_size=Config.DEBUG_SAMPLE_SIZE
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NotebookEmbeddingDataset.collate_fn,
    )

    # Load Ground Truth
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    val_gt_map = {
        row["id"]: row["cell_order"].split() for _, row in val_meta.iterrows()
    }

    # Load Model
    model = DualContextAnchorNetwork().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print("Model file not found. Validation skipped.")
        return 0.0

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    total_swaps = 0
    total_max_swaps = 0

    # Store per-notebook stats for failure analysis
    notebook_errors = []  # (num_code, num_md, error_magnitude)

    with torch.no_grad():
        for batch in val_loader:
            code_embs = batch["code_embeddings"].to(device)
            md_embs = batch["markdown_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["markdown_mask"].to(device)

            ids = batch["ids"]
            code_cell_ids_batch = batch["code_cell_ids"]
            md_cell_ids_batch = batch["markdown_cell_ids"]

            # Forward pass
            logits = model(code_embs, md_embs, code_mask, md_mask)
            probs = torch.softmax(logits, dim=-1)

            # Expected Rank
            n_classes = logits.size(-1)
            indices = torch.arange(n_classes, device=device).float()
            expected_ranks = torch.sum(probs * indices, dim=-1).cpu().numpy()

            # Reconstruct and Evaluate
            batch_size = len(ids)
            for i in range(batch_size):
                nb_id = ids[i]
                if nb_id not in val_gt_map:
                    continue

                gt_order = val_gt_map[nb_id]
                code_ids = code_cell_ids_batch[i]
                md_ids = md_cell_ids_batch[i]

                num_code = len(code_ids)
                num_md = len(md_ids)

                # Get ranks
                nb_ranks = expected_ranks[i, :num_md]

                # Sort logic
                cells_with_pos = []
                for idx, cid in enumerate(code_ids):
                    cells_with_pos.append((float(idx), cid))
                for idx, cid in enumerate(md_ids):
                    rank = nb_ranks[idx]
                    cells_with_pos.append((rank - 0.5, cid))

                cells_with_pos.sort(key=lambda x: x[0])
                predicted_order = [cid for _, cid in cells_with_pos]

                # Metric
                s, max_s = calculate_kendall_tau_components(gt_order, predicted_order)
                total_swaps += s
                total_max_swaps += max_s

                # Per notebook Kendall Tau
                kt = 1 - 2 * (s / max_s) if max_s > 0 else 1.0
                error_mag = 1.0 - kt

                notebook_errors.append(
                    {"num_code": num_code, "num_md": num_md, "error": error_mag}
                )

    # Global Metric
    final_metric = (
        1 - 2 * (total_swaps / total_max_swaps) if total_max_swaps > 0 else 0.0
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    if notebook_errors:
        df_err = pd.DataFrame(notebook_errors)

        # Correlation with features
        corr_code, _ = spearmanr(df_err["num_code"], df_err["error"])
        corr_md, _ = spearmanr(df_err["num_md"], df_err["error"])

        print("\nFailure Analysis (Spearman Correlation with Error Magnitude):")
        print(f"Num Code Cells: {corr_code:.4f}")
        print(f"Num Markdown Cells: {corr_md:.4f}")

        if corr_code > 0.1 or corr_md > 0.1:
            print("-> Performance degrades on larger notebooks.")
        else:
            print("-> Performance is relatively stable across notebook sizes.")

    return final_metric


def main():
    set_seed(Config.SEED)

    # 1. Preprocessing
    print("=== Step 1: Data Preprocessing ===")
    preprocessor = DataPreprocessor()
    # load_cached_data=True allows skipping if files exist in ./working
    preprocessor.run(load_cached_data=True)

    # 2. Training
    print("\n=== Step 2: Model Training ===")
    train_model()

    # 3. Validation & Analysis
    final_metric = perform_validation_and_failure_analysis()

    # 4. Conditional Submission
    THRESHOLD = 0.8315021559000814
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        predict_and_sort()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
