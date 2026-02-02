import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from bisect import bisect_left
from library.config import Config
from library.dataset import NotebookEmbeddingDataset
from library.model import DualContextAnchorNetwork


def count_inversions(a):
    """
    Counts the number of inversions in a list using a binary search approach.
    Complexity: O(N log N)
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        idx = bisect_left(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def calculate_kendall_tau_components(ground_truth, predicted):
    """
    Calculates the number of swaps (S) and max possible swaps (M) for a single notebook.
    """
    # Map ground truth cell IDs to their rank (0 to N-1)
    rank_map = {cell_id: i for i, cell_id in enumerate(ground_truth)}

    # Convert predicted sequence to ranks based on ground truth
    # Filter ensures we only consider cells present in both (safety check)
    predicted_ranks = [rank_map[cid] for cid in predicted if cid in rank_map]

    n = len(predicted_ranks)
    if n <= 1:
        return 0, 0

    swaps = count_inversions(predicted_ranks)
    max_swaps = n * (n - 1) // 2

    return swaps, max_swaps


def train_model():
    # 1. Setup
    device = Config.DEVICE
    print(f"Training on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Handle debug mode sizing
    max_size = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None

    train_dataset = NotebookEmbeddingDataset(split="train", max_size=max_size)
    val_dataset = NotebookEmbeddingDataset(split="val", max_size=max_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NotebookEmbeddingDataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NotebookEmbeddingDataset.collate_fn,
    )

    # Load Validation Ground Truth for Metric Calculation
    # We need the full cell_order to compute Kendall Tau
    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA_PATH}"
        )

    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    val_gt_map = {
        row["id"]: row["cell_order"].split() for _, row in val_meta.iterrows()
    }

    # 3. Model & Optimization
    model = DualContextAnchorNetwork().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # No label smoothing as per design
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        train_loss_sum = 0
        train_batches = 0

        for batch in train_loader:
            # Move data to device
            code_embs = batch["code_embeddings"].to(device)
            md_embs = batch["markdown_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["markdown_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            # Logits shape: (Batch, N_md, N_code + 1)
            logits = model(code_embs, md_embs, code_mask, md_mask)

            # Flatten for CrossEntropyLoss
            # Logits: (B * N_md, N_code + 1)
            # Labels: (B * N_md)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()

            if Config.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0

        # ==========================
        # Validation Phase
        # ==========================
        model.eval()
        val_loss_sum = 0
        val_batches = 0

        total_swaps = 0
        total_max_swaps = 0

        with torch.no_grad():
            for batch in val_loader:
                code_embs = batch["code_embeddings"].to(device)
                md_embs = batch["markdown_embeddings"].to(device)
                code_mask = batch["code_mask"].to(device)
                md_mask = batch["markdown_mask"].to(device)
                labels = batch["labels"].to(device)

                ids = batch["ids"]
                code_cell_ids_batch = batch["code_cell_ids"]
                md_cell_ids_batch = batch["markdown_cell_ids"]

                logits = model(code_embs, md_embs, code_mask, md_mask)

                # Validation Loss
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(logits_flat, labels_flat)
                val_loss_sum += loss.item()
                val_batches += 1

                # --- Metric Calculation (Kendall Tau) ---
                # 1. Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)  # (B, N_md, N_code + 1)

                # 2. Compute Expected Index (Soft Ranking)
                # We use the index as the value (0, 1, 2, ...)
                n_classes = logits.size(-1)
                indices = torch.arange(n_classes, device=device).float()

                # Expected rank: (B, N_md)
                # This value represents "before which code cell" the markdown cell should be placed.
                expected_ranks = torch.sum(probs * indices, dim=-1)
                expected_ranks_np = expected_ranks.cpu().numpy()

                # 3. Reconstruct Order per notebook
                for i, nb_id in enumerate(ids):
                    if nb_id not in val_gt_map:
                        continue

                    gt_order = val_gt_map[nb_id]
                    code_ids = code_cell_ids_batch[i]
                    md_ids = md_cell_ids_batch[i]

                    # Slice expected ranks to actual number of markdown cells (ignoring padding)
                    num_md = len(md_ids)
                    nb_ranks = expected_ranks_np[i, :num_md]

                    # Construct list of (position_score, cell_id)
                    cells_with_pos = []

                    # Code cells are anchors at positions 0.0, 1.0, 2.0, ...
                    for idx, cid in enumerate(code_ids):
                        cells_with_pos.append((float(idx), cid))

                    # Markdown cells are placed relative to anchors.
                    # If rank is k, it means before code cell k.
                    # To place it correctly in a sort, we assign position k - 0.5.
                    for idx, cid in enumerate(md_ids):
                        rank = nb_ranks[idx]
                        cells_with_pos.append((rank - 0.5, cid))

                    # Sort by position score
                    cells_with_pos.sort(key=lambda x: x[0])
                    predicted_order = [cid for _, cid in cells_with_pos]

                    # Accumulate swaps
                    s, max_s = calculate_kendall_tau_components(
                        gt_order, predicted_order
                    )
                    total_swaps += s
                    total_max_swaps += max_s

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0

        # Compute Global Kendall Tau
        # K = 1 - 4 * (Sum_Swaps / Sum_Ni(Ni-1))
        # Sum_Ni(Ni-1) = 2 * Sum_Max_Swaps
        # K = 1 - 2 * (Sum_Swaps / Sum_Max_Swaps)
        if total_max_swaps > 0:
            kendall_tau = 1 - 2 * (total_swaps / total_max_swaps)
        else:
            kendall_tau = 0.0

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Kendall Tau: {kendall_tau:.6f}"
        )

        # ==========================
        # Checkpointing & Early Stopping
        # ==========================
        if kendall_tau > best_score:
            best_score = kendall_tau
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with score: {best_score:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")
