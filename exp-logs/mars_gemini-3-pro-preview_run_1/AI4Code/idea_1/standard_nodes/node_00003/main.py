import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, count_inversions
from library.data import EmbeddingManager, NotebookDataset, collate_fn
from library.model import SemanticAnchorClassifier
from library.train import train_one_epoch, validate
from library.inference import generate_submission


def perform_failure_analysis(model, dataloader, device, code_map, ground_truth):
    """
    Analyzes the performance of the model on the validation set to identify error patterns.
    Calculates the correlation between error magnitude (1 - Kendall Tau) and notebook properties.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    results = []

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch["code_embeddings"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            ids = batch["ids"]
            batch_md_ids = batch["markdown_ids"]

            # Forward pass
            if md_emb.size(1) > 0:
                logits = model(md_emb, code_emb)
                probs = torch.softmax(logits, dim=-1)
                num_classes = probs.size(-1)
                class_indices = torch.arange(num_classes, device=device).float()
                expected_positions = (
                    torch.sum(probs * class_indices, dim=-1).cpu().numpy()
                )
            else:
                expected_positions = np.array([])

            # Process each notebook in batch
            for i, nb_id in enumerate(ids):
                if nb_id not in ground_truth:
                    continue

                gt_order = ground_truth[nb_id]
                nb_code_ids = code_map.get(nb_id, [])
                nb_md_ids = batch_md_ids[i]

                # Reconstruct predicted order
                cells_with_pos = []
                for idx, cid in enumerate(nb_code_ids):
                    cells_with_pos.append((idx + 0.5, cid))

                if len(nb_md_ids) > 0:
                    valid_len = len(nb_md_ids)
                    if expected_positions.size > 0:
                        nb_preds = expected_positions[i][:valid_len]
                        for md_id, pos in zip(nb_md_ids, nb_preds):
                            cells_with_pos.append((pos, md_id))
                    else:
                        for md_id in nb_md_ids:
                            cells_with_pos.append((0.0, md_id))

                cells_with_pos.sort(key=lambda x: x[0])
                pred_order = [x[1] for x in cells_with_pos]

                # Calculate Kendall Tau for this specific notebook
                n = len(gt_order)
                if n <= 1:
                    score = 1.0
                else:
                    gt_rank_map = {cell_id: idx for idx, cell_id in enumerate(gt_order)}
                    ranks = [
                        gt_rank_map[cid] for cid in pred_order if cid in gt_rank_map
                    ]
                    swaps = count_inversions(ranks)
                    total_possible = (
                        n * (n - 1) // 2
                    )  # Note: The formula in task uses n(n-1) in denominator with 4*swaps
                    # Task formula: K = 1 - 4 * (swaps / (n * (n-1)))
                    # Which is equivalent to 1 - 2 * (swaps / (n * (n-1) / 2))
                    score = 1 - 4 * (swaps / (n * (n - 1)))

                # Error magnitude
                error = 1.0 - score

                results.append(
                    {
                        "id": nb_id,
                        "num_code": len(nb_code_ids),
                        "num_md": len(nb_md_ids),
                        "total_cells": n,
                        "score": score,
                        "error": error,
                    }
                )

    df_results = pd.DataFrame(results)

    # Calculate correlations
    if len(df_results) > 0:
        corr_code, _ = pearsonr(df_results["num_code"], df_results["error"])
        corr_md, _ = pearsonr(df_results["num_md"], df_results["error"])

        print(f"Correlation between Error and Num Code Cells: {corr_code:.4f}")
        print(f"Correlation between Error and Num Markdown Cells: {corr_md:.4f}")

        # Insight
        if abs(corr_md) > 0.1:
            direction = "increases" if corr_md > 0 else "decreases"
            print(
                f"-> Error tends to {direction} as the number of markdown cells increases."
            )
    else:
        print("Insufficient data for failure analysis.")


def main():
    # 1. Configuration
    config = Config()

    # Optimized Settings for Full Training
    config.epochs = 15
    config.train_batch_size = 64
    # Ensure we use GPU if available
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(config.seed)
    print(f"Running Optimized Training on {config.device}")

    # 2. Data Loading & Preprocessing
    manager = EmbeddingManager(config)

    # Load Train Data
    print("Loading Training Data...")
    df_train = manager.process_data("train", load_cached_data=True)
    # Cite solution_lesson_node_00002: Using full dataset to improve Macro-Ordering
    # by exposing the model to more anchor-markdown relationships.

    train_dataset = NotebookDataset(df_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load Validation Data
    print("Loading Validation Data...")
    df_val = manager.process_data("val", load_cached_data=True)

    # We use the full validation set for the final metric,
    # but we can use the same loader for the training loop.
    val_dataset = NotebookDataset(df_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Prepare maps for validation
    val_code_map = {
        row["id"]: json.loads(row["code_ids"]) for _, row in df_val.iterrows()
    }
    df_val_meta = pd.read_csv(config.val_metadata_path)
    val_ground_truth = dict(zip(df_val_meta.id, df_val_meta.cell_order.str.split()))

    # 3. Model Initialization
    model = SemanticAnchorClassifier(config)
    model.to(config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Ignore padding index -100
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    best_score = -float("inf")

    print(f"Starting training for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )

        # Validate (using full val set here as it's reasonably sized ~23k)
        val_loss, val_score = validate(
            model, val_loader, criterion, config.device, val_code_map, val_ground_truth
        )

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Kendall Tau: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )
    model.eval()

    _, final_metric = validate(
        model, val_loader, criterion, config.device, val_code_map, val_ground_truth
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(
        model, val_loader, config.device, val_code_map, val_ground_truth
    )

    # 7. Submission Generation
    # Only generate submission if we improved over the baseline
    if final_metric > 0.5325106444396335:
        print(
            f"\nMetric {final_metric:.6f} > 0.5325. Generating Submission for Test Set..."
        )

        # Remove stale test cache if it exists to ensure full test set is processed
        if os.path.exists(config.test_cache_path):
            os.remove(config.test_cache_path)

        # generate_submission handles loading the model from config.model_save_path
        generate_submission(config)
        print("Done.")
    else:
        print(
            f"\nMetric {final_metric:.6f} did not beat baseline 0.5325. Skipping submission."
        )


if __name__ == "__main__":
    main()
