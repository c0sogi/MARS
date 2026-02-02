import torch
import torch.nn as nn
import numpy as np
import os
import time
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, device, epoch_index):
    """
    Trains the model for one epoch.
    Handles both standard batches and MixUp batches dynamically.
    """
    model.train()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    # Use reduction='none' to handle per-sample MixUp weights correctly
    criterion_none = nn.CrossEntropyLoss(
        label_smoothing=Config.LABEL_SMOOTHING, reduction="none"
    )
    # Use reduction='mean' for standard batch processing if needed,
    # but we can use 'none' and .mean() for consistency.

    for batch_idx, batch_data in enumerate(dataloader):
        # Move data to device
        # Check if MixUp is active based on number of unpacked elements
        # MixUp returns: (features, l1, l2, l3, l1_b, l2_b, l3_b, lam) -> 8 elements
        # Standard returns: (features, l1, l2, l3) -> 4 elements

        if len(batch_data) == 8:
            features, l1, l2, l3, l1_b, l2_b, l3_b, lam = batch_data
            features = features.to(device)
            l1, l2, l3 = l1.to(device), l2.to(device), l3.to(device)
            l1_b, l2_b, l3_b = l1_b.to(device), l2_b.to(device), l3_b.to(device)
            lam = lam.to(device)

            # Forward pass
            out_l1, out_l2, out_l3 = model(features)

            # Calculate Mixed Loss
            # lam is shape (Batch,), losses are shape (Batch,)
            loss_l1 = criterion_none(out_l1, l1) * lam + criterion_none(
                out_l1, l1_b
            ) * (1 - lam)
            loss_l2 = criterion_none(out_l2, l2) * lam + criterion_none(
                out_l2, l2_b
            ) * (1 - lam)
            loss_l3 = criterion_none(out_l3, l3) * lam + criterion_none(
                out_l3, l3_b
            ) * (1 - lam)

            # Sum tasks and average over batch
            batch_loss = (loss_l1 + loss_l2 + loss_l3).mean()

            # For accuracy tracking during MixUp, we compare against the primary target (l3)
            # This is an approximation but sufficient for progress monitoring
            preds = torch.argmax(out_l3, dim=1)
            correct_l3 += (preds == l3).sum().item()

        else:
            features, l1, l2, l3 = batch_data
            features = features.to(device)
            l1, l2, l3 = l1.to(device), l2.to(device), l3.to(device)

            # Forward pass
            out_l1, out_l2, out_l3 = model(features)

            # Calculate Standard Loss
            loss_l1 = criterion_none(out_l1, l1)
            loss_l2 = criterion_none(out_l2, l2)
            loss_l3 = criterion_none(out_l3, l3)

            batch_loss = (loss_l1 + loss_l2 + loss_l3).mean()

            preds = torch.argmax(out_l3, dim=1)
            correct_l3 += (preds == l3).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Metrics
        batch_size = features.size(0)
        running_loss += batch_loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_l3 / total_samples

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss(
        label_smoothing=Config.LABEL_SMOOTHING, reduction="sum"
    )

    with torch.no_grad():
        for batch_data in dataloader:
            features, l1, l2, l3 = batch_data
            features = features.to(device)
            l1, l2, l3 = l1.to(device), l2.to(device), l3.to(device)

            out_l1, out_l2, out_l3 = model(features)

            loss_l1 = criterion(out_l1, l1)
            loss_l2 = criterion(out_l2, l2)
            loss_l3 = criterion(out_l3, l3)

            # Total summed loss for the batch
            batch_loss = loss_l1 + loss_l2 + loss_l3
            running_loss += batch_loss.item()

            # Accuracy (Level 3 Target)
            preds = torch.argmax(out_l3, dim=1)
            correct_l3 += (preds == l3).sum().item()
            total_samples += features.size(0)

    avg_loss = running_loss / total_samples
    accuracy = correct_l3 / total_samples

    return avg_loss, accuracy


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    save_path="best_model.pth",
    patience=5,
):
    """
    Main training loop with Early Stopping.
    """
    best_val_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! (Acc: {val_acc:.6f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.6f}")
    return best_val_acc


def predict_test(model, test_loader, device):
    """
    Generates predictions for the test set.
    Returns:
        ids (numpy array): Product IDs
        preds (numpy array): Predicted Category IDs (Level 3)
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Generating predictions on test set...")

    with torch.no_grad():
        for features, product_ids in test_loader:
            features = features.to(device)

            # Forward pass
            _, _, out_l3 = model(features)

            # Get predicted class indices
            probs = torch.softmax(out_l3, dim=1)
            pred_indices = torch.argmax(probs, dim=1).cpu().numpy()

            all_preds.append(pred_indices)
            all_ids.append(product_ids.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    return all_ids, all_preds
