import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np


# 1. Patch tqdm for silent execution
class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def silent_tqdm_constructor(iterable, *args, **kwargs):
    return SilentTqdm(iterable, *args, **kwargs)


import tqdm

tqdm.tqdm = silent_tqdm_constructor

# 2. Import Library Modules
from library.config import (
    MODEL_SAVE_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED,
    VAL_METADATA_PATH,
)
from library.utils import set_seed, collate_fn, compute_kendall_tau
from library.dataset import NotebookDataset
from library.model import CAAN
from library.engine import Engine


def perform_failure_analysis(model, dataloader, dataset, engine):
    """
    Analyzes model performance on validation set to find correlations
    between error magnitude and input features.
    """
    model.eval()

    # Ensure metadata maps are loaded in engine
    if engine.val_map is None:
        engine.val_map = engine._load_metadata_map("val")
    if engine.val_gt is None:
        df_val = pd.read_csv(VAL_METADATA_PATH)
        engine.val_gt = dict(zip(df_val["id"], df_val["cell_order"]))

    results = []
    current_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch["code_emb"].to(DEVICE)
            code_mask = batch["code_mask"].to(DEVICE)
            md_emb = batch["md_emb"].to(DEVICE)
            md_mask = batch["md_mask"].to(DEVICE)

            logits = model(code_emb, code_mask, md_emb, md_mask)
            probs = torch.softmax(logits, dim=-1)

            # Expected Rank Calculation
            rank_indices = torch.arange(logits.size(-1), device=DEVICE).float()
            expected_ranks = (probs * rank_indices).sum(dim=-1)

            batch_size = code_emb.size(0)

            for b in range(batch_size):
                # Retrieve Notebook ID
                # Note: dataset is the full validation dataset, but dataloader iterates sequentially
                nb_id = dataset.samples[current_idx + b]["id"]

                if nb_id not in engine.val_map:
                    continue

                code_ids, md_ids = engine.val_map[nb_id]
                num_md = len(md_ids)
                num_code = len(code_ids)

                # Get predicted scores for MD cells
                scores = expected_ranks[b, :num_md].cpu().numpy()

                # Reconstruct Order
                cells_with_scores = []
                for i, cid in enumerate(code_ids):
                    cells_with_scores.append((cid, i + 0.5))
                for i, cid in enumerate(md_ids):
                    cells_with_scores.append((cid, scores[i]))

                cells_with_scores.sort(key=lambda x: x[1])
                pred_order = [x[0] for x in cells_with_scores]

                # Compute Error (1 - Kendall Tau)
                gt_order = engine.val_gt[nb_id].split()
                kt = compute_kendall_tau([pred_order], [gt_order])
                error = 1.0 - kt

                results.append({"num_code": num_code, "num_md": num_md, "error": error})

            current_idx += batch_size

    # Compute Correlations
    if results:
        df_res = pd.DataFrame(results)
        corr_code = df_res["num_code"].corr(df_res["error"])
        corr_md = df_res["num_md"].corr(df_res["error"])

        print(f"Correlation between Error and Num Code Cells: {corr_code:.16f}")
        print(f"Correlation between Error and Num Markdown Cells: {corr_md:.16f}")
    else:
        print("No validation results found for analysis.")


def main():
    # Set reproducible state
    set_seed(SEED)

    # --- Data Loading ---
    # Load Train Dataset (Cached)
    # We limit training to 10,000 samples to ensure quick baseline execution within 2 hours.
    full_train_dataset = NotebookDataset(
        split="train", load_cached_data=True, debug=False
    )
    train_indices = list(range(min(len(full_train_dataset), 10000)))
    train_subset = Subset(full_train_dataset, train_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # Load Validation Dataset
    val_dataset = NotebookDataset(split="val", load_cached_data=True, debug=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # --- Model Initialization ---
    model = CAAN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Training ---
    engine = Engine(model, DEVICE, optimizer)

    # Train for 1 epoch for baseline
    engine.train_one_epoch(train_loader, epoch=1)

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- Validation ---
    # Validate and print metric
    val_loss, val_metric = engine.validate(val_loader, val_dataset)
    print(f"Final Validation Metric: {val_metric}")

    # --- Failure Analysis ---
    perform_failure_analysis(model, val_loader, val_dataset, engine)

    # --- Submission ---
    THRESHOLD = 0.6598830915782636

    if val_metric > THRESHOLD:
        # Cite debug_lesson_1: Force fresh processing to avoid loading stale/incomplete cache
        test_dataset = NotebookDataset(
            split="test", load_cached_data=False, debug=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=(DEVICE == "cuda"),
        )
        engine.generate_submission(test_loader, test_dataset)


if __name__ == "__main__":
    main()
