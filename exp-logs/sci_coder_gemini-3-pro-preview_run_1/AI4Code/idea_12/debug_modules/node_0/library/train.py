import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys

# Import from library
from library.config import Config
from library.dataset import CachedNotebookDataset, collate_fn
from library.model import DC_AN
from library.preprocess import generate_embeddings


def count_inversions(prediction, ground_truth):
    """
    Counts the number of swaps (inversions) needed to sort the prediction
    into the ground truth order.
    """
    gt_rank = {cid: i for i, cid in enumerate(ground_truth)}
    # Filter prediction to ensure we only consider cells present in GT
    pred_ranks = [gt_rank[cid] for cid in prediction if cid in gt_rank]

    inversions = 0
    n = len(pred_ranks)
    for i in range(n):
        for j in range(i + 1, n):
            if pred_ranks[i] > pred_ranks[j]:
                inversions += 1
    return inversions


def get_ordered_cells(code_ids, md_ids, md_pred_pos):
    """
    Reconstructs the cell order by interleaving code and markdown cells.
    Code cells are treated as fixed anchors at positions 0.5, 1.5, 2.5, etc.
    Markdown cells are placed based on their predicted continuous position.
    """
    cells = []
    # Code cells are anchors
    for i, cid in enumerate(code_ids):
        cells.append({"id": cid, "pos": i + 0.5})

    # Markdown cells
    for i, cid in enumerate(md_ids):
        cells.append({"id": cid, "pos": md_pred_pos[i]})

    # Sort by position
    cells.sort(key=lambda x: x["pos"])
    return [x["id"] for x in cells]


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move inputs to device
        code_emb = batch["code_emb"].to(device)
        code_lens = batch["code_lens"].to(device)
        md_emb = batch["md_emb"].to(device)
        md_mask = batch["md_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # logits: [B, M, L+1]
        logits = model(code_emb, code_lens, md_emb, md_mask)

        # Flatten for loss calculation
        # logits: [B*M, L+1]
        # labels: [B*M]
        # Labels are padded with -100, which CrossEntropyLoss ignores.
        B, M, L_plus_1 = logits.shape
        logits_flat = logits.view(-1, L_plus_1)
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / count if count > 0 else 0.0


def validate(model, dataloader, df_metadata, device):
    """
    Evaluates the model on the validation set using the Kendall Tau metric.
    """
    model.eval()

    # Create a map for quick ground truth lookup
    gt_map = {}
    for _, row in df_metadata.iterrows():
        if isinstance(row["cell_order"], str):
            gt_map[row["id"]] = row["cell_order"].split()

    total_swaps = 0
    total_max_swaps = 0

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch["code_emb"].to(device)
            code_lens = batch["code_lens"].to(device)
            md_emb = batch["md_emb"].to(device)
            md_mask = batch["md_mask"].to(device)

            ids = batch["ids"]
            batch_code_ids = batch["code_ids"]
            batch_md_ids = batch["md_ids"]

            # Forward pass
            logits = model(code_emb, code_lens, md_emb, md_mask)

            # Calculate Expected Position (Soft Ranking)
            probs = torch.softmax(logits, dim=-1)  # [B, M, L+1]
            L_plus_1 = probs.size(-1)
            indices = torch.arange(L_plus_1, device=device).float()

            # expected_pos: [B, M]
            expected_pos = torch.sum(probs * indices, dim=-1)
            expected_pos = expected_pos.cpu().numpy()

            for i, nb_id in enumerate(ids):
                if nb_id not in gt_map:
                    continue

                gt_order = gt_map[nb_id]
                c_ids = batch_code_ids[i]
                m_ids = batch_md_ids[i]

                num_md = len(m_ids)
                if num_md == 0:
                    pred_order = c_ids
                else:
                    m_scores = expected_pos[i, :num_md]
                    pred_order = get_ordered_cells(c_ids, m_ids, m_scores)

                # Calculate metric components
                n = len(gt_order)
                if n > 1:
                    swaps = count_inversions(pred_order, gt_order)
                    max_s = n * (n - 1) // 2

                    total_swaps += swaps
                    total_max_swaps += max_s

    if total_max_swaps == 0:
        return 0.0

    kendall_tau = 1 - 4 * (total_swaps / total_max_swaps)
    return kendall_tau


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []

    print(f"Generating submission for {len(dataloader.dataset)} notebooks...")

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch["code_emb"].to(device)
            code_lens = batch["code_lens"].to(device)
            md_emb = batch["md_emb"].to(device)
            md_mask = batch["md_mask"].to(device)

            ids = batch["ids"]
            batch_code_ids = batch["code_ids"]
            batch_md_ids = batch["md_ids"]

            logits = model(code_emb, code_lens, md_emb, md_mask)
            probs = torch.softmax(logits, dim=-1)
            L_plus_1 = probs.size(-1)
            indices = torch.arange(L_plus_1, device=device).float()
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

            for i, nb_id in enumerate(ids):
                c_ids = batch_code_ids[i]
                m_ids = batch_md_ids[i]
                num_md = len(m_ids)

                if num_md == 0:
                    pred_order = c_ids
                else:
                    m_scores = expected_pos[i, :num_md]
                    pred_order = get_ordered_cells(c_ids, m_ids, m_scores)

                predictions.append({"id": nb_id, "cell_order": " ".join(pred_order)})

    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function.
    """
    # 1. Ensure Data Embeddings
    print("Checking/Generating Embeddings...")
    generate_embeddings(load_cached_data=True)

    # 2. Setup Datasets & Loaders
    train_ds = CachedNotebookDataset(Config.TRAIN_FEATURES_PATH, Config)
    val_ds = CachedNotebookDataset(Config.VAL_FEATURES_PATH, Config)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DC_AN(Config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    best_score = -float("inf")
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, df_val_meta, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Loss: {avg_loss:.4f} - Val Kendall: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved! ({val_score:.4f})")

    print(f"Training complete. Best Validation Score: {best_score:.4f}")

    # 5. Generate Final Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    test_ds = CachedNotebookDataset(Config.TEST_FEATURES_PATH, Config)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
