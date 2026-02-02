import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from bisect import bisect_left

from library.config import Config
from library.feature_extraction import extract_features
from library.dataset import HAPSDataset, haps_collate_fn
from library.model import HAPSModel
from library.loss import HAPSLoss
from library.inference_utils import (
    compute_global_sort,
    refine_order,
    generate_predictions,
)


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_inversions(a):
    """Counts the number of inversions in a list."""
    inversions = 0
    sorted_so_far = []
    for x in a:
        idx = bisect_left(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(ground_truth, predicted):
    """
    Computes the number of swaps and the max possible swaps for Kendall Tau.
    K = 1 - 4 * (S / (n * (n - 1)))
    """
    n = len(ground_truth)
    if n <= 1:
        return 0, n  # S=0, n=0/1 results in denom 0, handled by caller

    # Map cell_id to rank in ground truth
    rank_map = {cid: i for i, cid in enumerate(ground_truth)}

    # Convert predicted list to ranks
    # Filter out any ids not in ground truth (safety check)
    predicted_ranks = [rank_map[cid] for cid in predicted if cid in rank_map]

    # Count inversions (swaps)
    s = count_inversions(predicted_ranks)

    return s, n


def validate(model, dataset, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    total_swaps = 0
    total_denominator = 0

    # Access the list of notebook IDs
    notebook_ids = dataset.notebooks

    with torch.no_grad():
        for i in range(len(notebook_ids)):
            sample = dataset[i]
            nb_id = sample["notebook_id"]

            # Prepare inputs
            code_emb = sample["code_embeddings"].unsqueeze(0).to(device)
            md_emb = sample["md_embeddings"].unsqueeze(0).to(device)

            n_code = code_emb.size(1)
            n_md = md_emb.size(1)

            code_mask = torch.ones((1, n_code), dtype=torch.bool, device=device)
            md_mask = torch.ones((1, n_md), dtype=torch.bool, device=device)

            # Get Ground Truth Order and Cell IDs
            # dataset.grouped is a pandas GroupBy object
            nb_df = dataset.grouped.get_group(nb_id)

            # Ground Truth: Sort by 'rank'
            gt_df = nb_df.sort_values("rank")
            gt_order = gt_df["cell_id"].tolist()

            # Input IDs
            code_ids = nb_df[nb_df["cell_type"] == "code"]["cell_id"].tolist()
            md_ids = nb_df[nb_df["cell_type"] == "markdown"]["cell_id"].tolist()

            # Forward Pass
            outputs = model(code_emb, code_mask, md_emb, md_mask, pairwise_indices=None)

            # Global Sort (Anchor Head)
            all_cells = compute_global_sort(outputs["anchor_logits"], code_ids, md_ids)

            # Refinement (Pairwise Head)
            if n_md > 1:
                pred_order = refine_order(model, all_cells, md_emb, device, passes=2)
            else:
                pred_order = [c["cell_id"] for c in all_cells]

            # Metric Calculation
            s, n = compute_kendall_tau(gt_order, pred_order)

            total_swaps += s
            total_denominator += n * (n - 1)

    if total_denominator == 0:
        return 0.0

    kt = 1 - 4 * (total_swaps / total_denominator)
    return kt


def train():
    set_seed(Config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Feature Extraction
    # Checks cache first, computes if missing
    print("Extracting/Loading features...")
    extract_features(load_cached_data=True)

    # 2. Datasets & Loaders
    print("Initializing datasets...")
    train_dataset = HAPSDataset(Config.train_features_path, mode="train")
    val_dataset = HAPSDataset(Config.val_features_path, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=haps_collate_fn,
        pin_memory=True,
    )

    # 3. Model & Optimization
    model = HAPSModel().to(device)
    criterion = HAPSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    best_kt = -float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0
        running_anchor = 0.0
        running_pairwise = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            code_emb = batch["code_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_emb = batch["md_embeddings"].to(device)
            md_mask = batch["md_mask"].to(device)

            # Labels
            anchor_labels = batch.get("anchor_labels")
            if anchor_labels is not None:
                anchor_labels = anchor_labels.to(device)

            pairwise_indices = batch.get("pairwise_indices")
            if pairwise_indices is not None:
                pairwise_indices = pairwise_indices.to(device)

            pairwise_labels = batch.get("pairwise_labels")
            if pairwise_labels is not None:
                pairwise_labels = pairwise_labels.to(device)

            # Construct batch dict for loss
            batch_data = {
                "anchor_labels": anchor_labels,
                "pairwise_labels": pairwise_labels,
            }

            optimizer.zero_grad()

            outputs = model(code_emb, code_mask, md_emb, md_mask, pairwise_indices)

            loss_dict = criterion(outputs, batch_data)
            loss = loss_dict["loss"]

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_anchor += loss_dict["anchor_loss"].item()
            running_pairwise += loss_dict["pairwise_loss"].item()

        # Epoch Stats
        avg_loss = running_loss / len(train_loader)
        avg_anchor = running_anchor / len(train_loader)
        avg_pairwise = running_pairwise / len(train_loader)

        print(
            f"Epoch {epoch+1}/{Config.num_epochs} - Loss: {avg_loss:.6f} (Anchor: {avg_anchor:.6f}, Pairwise: {avg_pairwise:.6f})"
        )

        # Validation
        print("Validating...")
        val_kt = validate(model, val_dataset, device)
        print(f"Validation Kendall Tau: {val_kt}")

        # Checkpoint & Early Stopping
        if val_kt > best_kt:
            best_kt = val_kt
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved with KT: {val_kt}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.patience}")

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation KT: {best_kt}")

    # 4. Inference on Test Set
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.model_save_path))

    print("Generating submission...")
    test_dataset = HAPSDataset(Config.test_features_path, mode="test")
    submission_df = generate_predictions(model, test_dataset, device)

    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
