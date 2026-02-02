import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.model import SemanticAnchorClassifier
from library.data import EmbeddingManager, NotebookDataset, collate_fn


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move batch to device
        code_emb = batch["code_embeddings"].to(device)
        md_emb = batch["markdown_embeddings"].to(device)
        labels = batch["labels"].to(device)

        # Skip batches with no markdown cells if any (though collate handles padding)
        if md_emb.size(1) == 0:
            continue

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, Num_MD, Num_Code + 1)
        logits = model(md_emb, code_emb)

        # Flatten for CrossEntropyLoss
        # Logits: (Batch * Num_MD, Num_Classes)
        # Labels: (Batch * Num_MD)
        # Note: labels contain -100 for padded positions, which CrossEntropyLoss ignores.
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1

    return running_loss / count if count > 0 else 0.0


def validate(model, dataloader, criterion, device, code_map, ground_truth):
    """
    Evaluates the model on the validation set and computes Kendall Tau.

    Args:
        code_map: Dict[nb_id, List[code_cell_ids]] to reconstruct order.
        ground_truth: Dict[nb_id, List[all_cell_ids]] for metric calculation.
    """
    model.eval()
    running_loss = 0.0
    count = 0
    predictions = {}

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch["code_embeddings"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            labels = batch["labels"].to(device)
            ids = batch["ids"]
            batch_md_ids = batch["markdown_ids"]

            # Forward pass
            if md_emb.size(1) > 0:
                logits = model(md_emb, code_emb)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                running_loss += loss.item()

                # Calculate probabilities
                probs = torch.softmax(logits, dim=-1)

                # Calculate Expected Position (Soft Rank)
                # Shape: (Batch, Num_MD)
                num_classes = probs.size(-1)
                class_indices = torch.arange(num_classes, device=device).float()
                expected_positions = torch.sum(probs * class_indices, dim=-1)

                expected_positions = expected_positions.cpu().numpy()
            else:
                # Handle case with no markdown cells
                loss = 0.0  # No loss contribution
                expected_positions = np.array([])

            count += 1

            # Reconstruct Order for each notebook in batch
            for i, nb_id in enumerate(ids):
                # Get Code Cells
                nb_code_ids = code_map.get(nb_id, [])

                # Get Markdown Cells and their predicted positions
                nb_md_ids = batch_md_ids[i]

                # Combine into a list of (position, cell_id)
                cells_with_pos = []

                # Code cells are anchors at positions 0.5, 1.5, 2.5, ...
                for idx, cid in enumerate(nb_code_ids):
                    cells_with_pos.append((idx + 0.5, cid))

                # Markdown cells are at their expected index
                if len(nb_md_ids) > 0:
                    # Extract valid predictions (remove padding)
                    # The batch is padded, but nb_md_ids has the true length
                    valid_len = len(nb_md_ids)
                    nb_preds = expected_positions[i][:valid_len]

                    for md_id, pos in zip(nb_md_ids, nb_preds):
                        cells_with_pos.append((pos, md_id))

                # Sort by position
                cells_with_pos.sort(key=lambda x: x[0])

                # Extract ordered IDs
                pred_order = [x[1] for x in cells_with_pos]
                predictions[nb_id] = pred_order

    avg_loss = running_loss / count if count > 0 else 0.0

    # Compute Metric
    score = compute_kendall_tau(ground_truth, predictions)

    return avg_loss, score


def run_training(config: Config = None):
    """
    Main execution function for training.
    """
    if config is None:
        config = Config()

    set_seed(config.seed)

    # ---------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # ---------------------------------------------------------
    manager = EmbeddingManager(config)

    # Load or Compute Train Features
    df_train = manager.process_data("train", load_cached_data=True)
    train_dataset = NotebookDataset(df_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load or Compute Val Features
    df_val = manager.process_data("val", load_cached_data=True)
    val_dataset = NotebookDataset(df_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Prepare Validation Maps (Code IDs and Ground Truth)
    # Code IDs are needed to reconstruct the notebook.
    # Ground Truth is needed for Kendall Tau.
    val_code_map = {
        row["id"]: json.loads(row["code_ids"]) for _, row in df_val.iterrows()
    }

    # Load ground truth from metadata file
    df_val_meta = pd.read_csv(config.val_metadata_path)
    val_ground_truth = dict(zip(df_val_meta.id, df_val_meta.cell_order.str.split()))

    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    model = SemanticAnchorClassifier(config)
    model.to(config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Ignore index -100 (used for padding in collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training on {config.device} for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )

        # Validate
        val_loss, val_score = validate(
            model, val_loader, criterion, config.device, val_code_map, val_ground_truth
        )

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Kendall Tau: {val_score:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  -> New best model saved! (Score: {best_score:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")
