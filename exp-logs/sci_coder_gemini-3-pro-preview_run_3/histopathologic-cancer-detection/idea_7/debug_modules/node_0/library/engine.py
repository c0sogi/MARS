import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.utils import save_checkpoint, calculate_metric


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function.
        device (str): Computing device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Ensure labels are (B, 1) to match model output
        labels = labels.view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation dataloader.
        criterion (Loss): Loss function.
        device (str): Computing device.

    Returns:
        tuple: (avg_loss, auc_score, probabilities, targets)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            labels_reshaped = labels.view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels_reshaped)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate results
    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Calculate AUC
    auc_score = calculate_metric(all_targets, all_probs)

    return epoch_loss, auc_score, all_probs, all_targets


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, H+V Flip.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Inference dataloader.
        device (str): Computing device.

    Returns:
        np.ndarray: Aggregated probability predictions.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Original
            out1 = model(images)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip (dim 3 is width)
            img_h = torch.flip(images, [3])
            out2 = model(img_h)
            prob2 = torch.sigmoid(out2)

            # 3. Vertical Flip (dim 2 is height)
            img_v = torch.flip(images, [2])
            out3 = model(img_v)
            prob3 = torch.sigmoid(out3)

            # 4. H+V Flip
            img_hv = torch.flip(images, [2, 3])
            out4 = model(img_hv)
            prob4 = torch.sigmoid(out4)

            # Average probabilities
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            all_probs.append(avg_prob.cpu().numpy())

    return np.concatenate(all_probs).flatten()


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    save_path,
    epochs=Config.NUM_EPOCHS,
):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (str): Device.
        save_path (str): Path to save the best model.
        epochs (int): Max epochs.

    Returns:
        tuple: (best_auc, best_loss)
    """
    best_loss = float("inf")
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Loss: {val_loss:.8f} - "
            f"Val AUC: {val_auc:.8f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping Logic based on Validation Loss (or AUC)
        # Strategy: Minimize Val Loss. If Val Loss improves, save model.
        if val_loss < best_loss:
            best_loss = val_loss
            best_auc = val_auc
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "best_auc": best_auc,
            }
            save_checkpoint(checkpoint, save_path)
            # print(f"  Saved best model to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(
        f"Training complete. Best Val Loss: {best_loss:.8f}, Best Val AUC: {best_auc:.8f}"
    )
    return best_auc, best_loss
