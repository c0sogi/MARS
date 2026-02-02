import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from library.config import Config
from library.model import IMCN
from library.dataset import get_dataloaders
from library.embeddings import get_embedding_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion_la, criterion_sa, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move inputs to device
        q_indices = batch["q_indices"].to(device)
        c_indices = batch["c_indices"].to(device)

        # Move targets to device
        label_long = batch["label_long"].to(device)
        short_start = batch["short_start"].to(device)
        short_end = batch["short_end"].to(device)

        optimizer.zero_grad()

        # Forward pass
        la_logits, start_logits, end_logits = model(q_indices, c_indices)

        # Squeeze la_logits to match label shape (Batch,)
        la_logits = la_logits.squeeze(-1)

        # Compute Losses
        loss_la = criterion_la(la_logits, label_long)
        loss_start = criterion_sa(start_logits, short_start)
        loss_end = criterion_sa(end_logits, short_end)

        # Combined Loss
        loss = (Config.WEIGHT_LONG * loss_la) + (
            Config.WEIGHT_SHORT * (loss_start + loss_end)
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion_la, criterion_sa, device):
    model.eval()
    total_loss = 0.0
    correct_la = 0
    correct_sa = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            q_indices = batch["q_indices"].to(device)
            c_indices = batch["c_indices"].to(device)
            label_long = batch["label_long"].to(device)
            short_start = batch["short_start"].to(device)
            short_end = batch["short_end"].to(device)

            la_logits, start_logits, end_logits = model(q_indices, c_indices)
            la_logits = la_logits.squeeze(-1)

            # Loss
            loss_la = criterion_la(la_logits, label_long)
            loss_start = criterion_sa(start_logits, short_start)
            loss_end = criterion_sa(end_logits, short_end)

            loss = (Config.WEIGHT_LONG * loss_la) + (
                Config.WEIGHT_SHORT * (loss_start + loss_end)
            )
            total_loss += loss.item()

            # Metrics
            # Long Answer Accuracy (Threshold 0.5 for binary)
            preds_la = (torch.sigmoid(la_logits) > 0.5).float()
            correct_la += (preds_la == label_long).sum().item()

            # Short Answer Exact Match
            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)

            # Exact match requires both start and end to be correct
            match = (pred_start == short_start) & (pred_end == short_end)
            correct_sa += match.sum().item()

            total_samples += label_long.size(0)

    avg_loss = total_loss / len(loader)
    acc_la = correct_la / total_samples
    acc_sa = correct_sa / total_samples

    return avg_loss, acc_la, acc_sa


def train_model(
    num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    """
    Main function to train the IMCN model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Preparing DataLoaders...")
    train_loader, val_loader, _, word2idx = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # 2. Prepare Embeddings
    embedding_matrix = get_embedding_matrix(word2idx, load_cached_data=load_cached_data)

    # 3. Initialize Model
    print("Initializing Model...")
    model = IMCN(embedding_matrix)
    model.to(device)

    # 4. Setup Training
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion_la = nn.BCEWithLogitsLoss()
    criterion_sa = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion_la, criterion_sa, device
        )
        val_loss, val_acc_la, val_acc_sa = validate(
            model, val_loader, criterion_la, criterion_sa, device
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Long Answer Accuracy: {val_acc_la}")
        print(f"Val Short Answer Exact Match: {val_acc_sa}")

        # Early Stopping and Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Config.ensure_directories()
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Validation loss improved. Model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
