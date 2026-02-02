import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_dataloader
from library.model import DCAN
from library.utils import set_seed, compute_kendall_tau, get_ordered_cell_ids


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move data to device
        code_features = batch["code_features"].to(device)
        code_mask = batch["code_mask"].to(device)
        markdown_features = batch["markdown_features"].to(device)
        markdown_mask = batch["markdown_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward Pass
        # logits shape: (Batch, Num_Markdown, Num_Code + 1)
        logits = model(code_features, code_mask, markdown_features, markdown_mask)

        # Flatten outputs and labels for CrossEntropyLoss
        # logits: (B * M, L+1)
        # labels: (B * M)
        B, M, L_plus_1 = logits.shape
        logits_flat = logits.view(-1, L_plus_1)
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        count += B

    avg_loss = total_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device, ground_truth_map):
    """
    Evaluates the model on the validation set using Kendall Tau.
    """
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in dataloader:
            code_features = batch["code_features"].to(device)
            code_mask = batch["code_mask"].to(device)
            markdown_features = batch["markdown_features"].to(device)
            markdown_mask = batch["markdown_mask"].to(device)

            ids = batch["ids"]
            batch_code_ids = batch["code_ids"]
            batch_markdown_ids = batch["markdown_ids"]

            # Forward Pass
            logits = model(code_features, code_mask, markdown_features, markdown_mask)

            # Compute Soft Ranking (Expected Index)
            # Probabilities: (B, M, L+1)
            probs = torch.softmax(logits, dim=-1)

            # Range [0, 1, ..., L]
            L_plus_1 = probs.size(-1)
            indices = torch.arange(L_plus_1, device=device, dtype=torch.float32)

            # Expected Index: sum(p * i) -> (B, M)
            expected_indices = torch.sum(probs * indices, dim=-1)
            expected_indices = expected_indices.cpu().numpy()

            # Reconstruct Order
            for i, nb_id in enumerate(ids):
                if nb_id not in ground_truth_map:
                    continue

                c_ids = batch_code_ids[i]
                m_ids = batch_markdown_ids[i]
                scores = expected_indices[i]

                # Filter scores to valid markdown cells (remove padding)
                valid_scores = scores[: len(m_ids)]

                # Get predicted order string
                pred_order_str = get_ordered_cell_ids(c_ids, m_ids, valid_scores)
                pred_list = pred_order_str.split()

                # Get ground truth list
                true_list = ground_truth_map[nb_id]

                predictions.append(pred_list)
                ground_truths.append(true_list)

    score = compute_kendall_tau(predictions, ground_truths)
    return score


def load_ground_truth(metadata_path):
    """
    Loads ground truth cell orders from metadata CSV into a dictionary.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    gt_map = {}
    for _, row in df.iterrows():
        gt_map[row["id"]] = row["cell_order"].split()
    return gt_map


def train_model(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, patience=Config.PATIENCE
):
    """
    Main function to train the model.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading DataLoaders...")
    train_loader = get_dataloader(
        split="train", batch_size=batch_size, load_cached_data=True
    )
    val_loader = get_dataloader(
        split="val", batch_size=batch_size, load_cached_data=True
    )

    # 2. Load Ground Truth for Validation
    print("Loading validation ground truth...")
    val_gt_map = load_ground_truth(Config.VAL_METADATA_PATH)

    # 3. Initialize Model
    print("Initializing Model...")
    model = DCAN().to(device)

    # 4. Optimizer & Loss
    # No warmup, constant LR as per instructions
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Ignore padding index (-100)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Training Loop
    best_score = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device, val_gt_map)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Kendall Tau: {val_score:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")
