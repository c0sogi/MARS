import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from bisect import bisect

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.preprocess import FeatureExtractor
from library.dataset import CachedDataset, collate_fn
from library.model import DCAN
from library.engine import Engine
from library.inference import generate_submission


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for x in a:
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def calculate_notebook_kendall_tau(pred_order, gt_order):
    """
    Calculates the Kendall Tau score for a single notebook.
    """
    n = len(gt_order)
    if n <= 1:
        return 1.0

    gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}
    pred_ranks = [
        gt_rank_map[cell_id] for cell_id in pred_order if cell_id in gt_rank_map
    ]

    s = count_inversions(pred_ranks)
    total_combinations = n * (n - 1)

    if total_combinations == 0:
        return 1.0

    return 1 - 4 * (s / total_combinations)


def analyze_failures(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Correlates model performance with notebook characteristics.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    # Create lookup for notebook structure
    nb_meta = val_df.set_index("id")[
        ["code_ids", "markdown_ids", "cell_order"]
    ].to_dict("index")

    results = []

    with torch.no_grad():
        for batch in val_loader:
            code_emb = batch["code_embeddings"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            code_lens = batch["code_lens"].to(device)
            md_lens = batch["md_lens"].to(device)
            ids = batch["ids"]

            logits = model(code_emb, md_emb, code_lens, md_lens)

            # Soft Ranking
            probs = torch.softmax(logits, dim=2)
            max_cls = probs.size(2)
            indices = torch.arange(max_cls, device=device).float().view(1, 1, -1)
            expected_ranks = torch.sum(probs * indices, dim=2).cpu().numpy()

            for i, nb_id in enumerate(ids):
                if nb_id not in nb_meta:
                    continue

                curr_code_ids = nb_meta[nb_id]["code_ids"]
                curr_md_ids = nb_meta[nb_id]["markdown_ids"]
                gt_order = nb_meta[nb_id]["cell_order"].split()

                curr_md_len = md_lens[i].item()
                num_md = min(len(curr_md_ids), curr_md_len)

                ranks = expected_ranks[i, :num_md]

                # Reconstruct predicted order
                cell_rank_pairs = []
                for c_idx, c_id in enumerate(curr_code_ids):
                    cell_rank_pairs.append((c_id, c_idx + 0.5))

                for m_idx in range(num_md):
                    cell_rank_pairs.append((curr_md_ids[m_idx], ranks[m_idx]))

                # Handle truncated markdown cells
                if len(curr_md_ids) > num_md:
                    for m_idx in range(num_md, len(curr_md_ids)):
                        cell_rank_pairs.append(
                            (curr_md_ids[m_idx], len(curr_code_ids) + 100.0 + m_idx)
                        )

                cell_rank_pairs.sort(key=lambda x: x[1])
                pred_order = [cid for cid, r in cell_rank_pairs]

                # Calculate Score
                score = calculate_notebook_kendall_tau(pred_order, gt_order)

                results.append(
                    {
                        "id": nb_id,
                        "score": score,
                        "num_code": len(curr_code_ids),
                        "num_md": len(curr_md_ids),
                    }
                )

    df_results = pd.DataFrame(results)

    # Calculate correlations
    corr_code = df_results["score"].corr(df_results["num_code"])
    corr_md = df_results["score"].corr(df_results["num_md"])

    print(f"Correlation between Kendall Tau Score and Num Code Cells: {corr_code:.4f}")
    print(
        f"Correlation between Kendall Tau Score and Num Markdown Cells: {corr_md:.4f}"
    )

    if corr_code < -0.1 or corr_md < -0.1:
        print("Insight: Performance degrades as notebook size increases.")
    else:
        print("Insight: Performance is relatively stable across notebook sizes.")


def main():
    # 1. Configuration and Setup
    config = Config()
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Preprocessing
    # Check if features exist, if not run preprocessing
    if not os.path.exists(config.TRAIN_FEATS_PATH) or not os.path.exists(
        config.VAL_FEATS_PATH
    ):
        print("Feature cache not found. Running preprocessing...")
        extractor = FeatureExtractor()
        extractor.run_preprocessing()
    else:
        print("Feature cache found.")

    # 3. Data Loading
    print("Loading datasets...")
    train_dataset = CachedDataset(mode="train", load_cached_data=True)
    val_dataset = CachedDataset(mode="val", load_cached_data=True)

    # Use full dataset but limit epochs for fast baseline
    # Batch size 64 on A100 is efficient
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Model Initialization
    print("Initializing DC-AN model...")
    model = DCAN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    engine = Engine(model, device, optimizer)

    # 5. Training
    # We run for 2 epochs to ensure convergence while staying within time limits.
    # 95k samples / 64 batch ~ 1500 steps per epoch. 2 epochs ~ 3000 steps.
    # Should take < 30 mins on A100.
    print("Starting training...")
    engine.fit(
        train_loader, val_loader, train_dataset.df, val_dataset.df, epochs=2, patience=1
    )

    # 6. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    print("Evaluating on full validation set...")
    val_score = engine.evaluate(val_loader, val_dataset.df)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, val_dataset.df, device)

    # 8. Submission
    threshold = 0.8315021559000814
    if val_score > threshold:
        print(
            f"Validation score {val_score:.6f} exceeds threshold {threshold}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Validation score {val_score:.6f} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
