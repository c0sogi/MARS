import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config

# -----------------------------------------------------------------------------
# Reproducibility Setup
# -----------------------------------------------------------------------------
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


def train_ranker_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the ranker for one epoch.

    Args:
        model: The KMaxInteractionRanker model.
        dataloader: DataLoader providing (q_indices, pos_indices, neg_indices).
        criterion: Loss function (MarginRankingLoss).
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_pairs = 0
    total_pairs = 0

    for batch in dataloader:
        q_indices, pos_indices, neg_indices = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass
        pos_scores = model(q_indices, pos_indices)
        neg_scores = model(q_indices, neg_indices)

        # Target is 1, meaning pos_scores should be ranked higher than neg_scores
        target = torch.ones_like(pos_scores).to(device)
        loss = criterion(pos_scores, neg_scores, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * q_indices.size(0)
        correct_pairs += (pos_scores > neg_scores).sum().item()
        total_pairs += q_indices.size(0)

    epoch_loss = running_loss / total_pairs if total_pairs > 0 else 0.0
    epoch_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_ranker(model, dataloader, criterion, device):
    """
    Validates the ranker.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_pairs = 0
    total_pairs = 0

    with torch.no_grad():
        for batch in dataloader:
            q_indices, pos_indices, neg_indices = [b.to(device) for b in batch]

            pos_scores = model(q_indices, pos_indices)
            neg_scores = model(q_indices, neg_indices)

            target = torch.ones_like(pos_scores).to(device)
            loss = criterion(pos_scores, neg_scores, target)

            running_loss += loss.item() * q_indices.size(0)
            correct_pairs += (pos_scores > neg_scores).sum().item()
            total_pairs += q_indices.size(0)

    epoch_loss = running_loss / total_pairs if total_pairs > 0 else 0.0
    epoch_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    return epoch_loss, epoch_acc


def train_ranker(
    model,
    train_loader,
    val_loader,
    device,
    epochs=config.NUM_EPOCHS,
    lr=config.LEARNING_RATE,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    """
    Full training loop for the Ranker model with Early Stopping.
    """
    print("Starting Ranker Training...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # MarginRankingLoss: loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    # We want pos > neg by at least margin 1.0
    criterion = nn.MarginRankingLoss(margin=1.0)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_ranker_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_ranker(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.10f}")
        print(f"Val Loss: {val_loss:.10f} | Val Acc: {val_acc:.10f}")

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(config.RANKER_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.RANKER_MODEL_PATH)
            print(f"New best model saved to {config.RANKER_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def train_reader_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the reader for one epoch.

    Args:
        model: The HighwayCoAttentionReader model.
        dataloader: DataLoader providing (q_indices, para_indices, start_target, end_target).
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        tuple: (average_loss, start_accuracy, end_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_start = 0
    correct_end = 0
    total_samples = 0

    for batch in dataloader:
        q_indices, para_indices, start_target, end_target = [
            b.to(device) for b in batch
        ]

        optimizer.zero_grad()

        start_logits, end_logits = model(q_indices, para_indices)

        loss_start = criterion(start_logits, start_target)
        loss_end = criterion(end_logits, end_target)
        loss = loss_start + loss_end

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * q_indices.size(0)

        # Calculate accuracies
        start_preds = torch.argmax(start_logits, dim=1)
        end_preds = torch.argmax(end_logits, dim=1)

        correct_start += (start_preds == start_target).sum().item()
        correct_end += (end_preds == end_target).sum().item()
        total_samples += q_indices.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc_start = correct_start / total_samples if total_samples > 0 else 0.0
    acc_end = correct_end / total_samples if total_samples > 0 else 0.0
    return epoch_loss, acc_start, acc_end


def validate_reader(model, dataloader, criterion, device):
    """
    Validates the reader.

    Returns:
        tuple: (average_loss, start_accuracy, end_accuracy, exact_match_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_start = 0
    correct_end = 0
    exact_match = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            q_indices, para_indices, start_target, end_target = [
                b.to(device) for b in batch
            ]

            start_logits, end_logits = model(q_indices, para_indices)

            loss_start = criterion(start_logits, start_target)
            loss_end = criterion(end_logits, end_target)
            loss = loss_start + loss_end

            running_loss += loss.item() * q_indices.size(0)

            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)

            correct_start += (start_preds == start_target).sum().item()
            correct_end += (end_preds == end_target).sum().item()

            # Exact Match: both start and end indices must match ground truth
            match_mask = (start_preds == start_target) & (end_preds == end_target)
            exact_match += match_mask.sum().item()

            total_samples += q_indices.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc_start = correct_start / total_samples if total_samples > 0 else 0.0
    acc_end = correct_end / total_samples if total_samples > 0 else 0.0
    acc_em = exact_match / total_samples if total_samples > 0 else 0.0
    return epoch_loss, acc_start, acc_end, acc_em


def train_reader(
    model,
    train_loader,
    val_loader,
    device,
    epochs=config.NUM_EPOCHS,
    lr=config.LEARNING_RATE,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    """
    Full training loop for the Reader model with Early Stopping.
    """
    print("Starting Reader Training...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_s_acc, train_e_acc = train_reader_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_s_acc, val_e_acc, val_em = validate_reader(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{epochs} | Time: {time.time() - start_time:.2f}s")
        print(
            f"Train Loss: {train_loss:.10f} | Start Acc: {train_s_acc:.10f} | End Acc: {train_e_acc:.10f}"
        )
        print(
            f"Val Loss: {val_loss:.10f} | Start Acc: {val_s_acc:.10f} | End Acc: {val_e_acc:.10f} | EM: {val_em:.10f}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(config.READER_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.READER_MODEL_PATH)
            print(f"New best model saved to {config.READER_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
