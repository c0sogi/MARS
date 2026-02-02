import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from library import config


def train_one_epoch(
    model, loader, optimizer, criterion, device, epoch, label_smoothing=0.0
):
    """
    Trains the model for one epoch using the provided loader and optimizer.

    Args:
        model: The PyTorch model.
        loader: DataLoader yielding (images, angles, labels).
        optimizer: The optimizer.
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        device: Torch device.
        epoch: Current epoch number (for logging).
        label_smoothing: Float value for label smoothing (0.0 to 1.0).

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    dataset_size = len(loader.dataset)

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Apply label smoothing for loss calculation
        if label_smoothing > 0.0:
            targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets = labels

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # Store predictions for accuracy calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    preds = (all_probs > 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds)

    print(
        f"Epoch {epoch} | Train Loss: {epoch_loss:.16f} | Train Acc: {epoch_acc:.16f}"
    )

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, use_tta=False):
    """
    Evaluates the model on a validation set. Supports Test Time Augmentation (TTA).

    Args:
        model: The PyTorch model.
        loader: DataLoader yielding (images, angles, labels).
        criterion: The loss function.
        device: Torch device.
        use_tta: If True, averages predictions from Original, H-Flip, and V-Flip.

    Returns:
        tuple: (average_loss, accuracy, all_probs, all_targets)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # 1. Original View
            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            if use_tta:
                # 2. Horizontal Flip (dim 3 is width)
                images_h = torch.flip(images, dims=[3])
                logits_h = model(images_h, angles)
                probs_h = torch.sigmoid(logits_h)

                # 3. Vertical Flip (dim 2 is height)
                images_v = torch.flip(images, dims=[2])
                logits_v = model(images_v, angles)
                probs_v = torch.sigmoid(logits_v)

                # 4. Horizontal + Vertical Flip (Rot 180) - Cite solution_lesson_node_00031
                images_hv = torch.flip(images, dims=[2, 3])
                logits_hv = model(images_hv, angles)
                probs_hv = torch.sigmoid(logits_hv)

                # Average Probabilities
                probs = (probs + probs_h + probs_v + probs_hv) / 4.0

                # Compute Loss on Averaged Probabilities manually
                # BCEWithLogitsLoss expects logits, but we have averaged probs.
                # Formula: -(y * log(p) + (1-y) * log(1-p))
                p_clamped = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
                batch_loss = -(
                    labels * torch.log(p_clamped)
                    + (1 - labels) * torch.log(1 - p_clamped)
                )
                batch_loss = batch_loss.mean()
            else:
                # Standard Loss using logits
                batch_loss = criterion(logits, labels)

            running_loss += batch_loss.item() * images.size(0)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    total_loss = running_loss / dataset_size
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    preds = (all_probs > 0.5).astype(int)
    total_acc = accuracy_score(all_targets, preds)

    print(f"Eval (TTA={use_tta}) | Loss: {total_loss:.16f} | Acc: {total_acc:.16f}")

    return total_loss, total_acc, all_probs, all_targets


def predict(model, loader, device, use_tta=True):
    """
    Generates predictions for the test set (no labels). Supports TTA.

    Args:
        model: The PyTorch model.
        loader: DataLoader yielding (images, angles, ids).
        device: Torch device.
        use_tta: If True, uses Original + H-Flip + V-Flip.

    Returns:
        tuple: (predictions_array, ids_list)
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            if use_tta:
                # 2. Horizontal Flip
                images_h = torch.flip(images, dims=[3])
                logits_h = model(images_h, angles)
                probs_h = torch.sigmoid(logits_h)

                # 3. Vertical Flip
                images_v = torch.flip(images, dims=[2])
                logits_v = model(images_v, angles)
                probs_v = torch.sigmoid(logits_v)

                # 4. H + V Flip
                images_hv = torch.flip(images, dims=[2, 3])
                logits_hv = model(images_hv, angles)
                probs_hv = torch.sigmoid(logits_hv)

                # Average
                probs = (probs + probs_h + probs_v + probs_hv) / 4.0

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    all_probs = np.concatenate(all_probs)
    return all_probs, all_ids


def swa_step(model, swa_model):
    """
    Updates the SWA model parameters with the current model parameters.
    """
    swa_model.update_parameters(model)


def update_swa_batch_norm(swa_model, loader, device):
    """
    Custom function to update Batch Normalization statistics for the SWA model.
    Necessary because the standard torch.optim.swa_utils.update_bn does not
    support models taking multiple inputs (image + angle).
    """
    swa_model.train()
    with torch.no_grad():
        for batch in loader:
            # Loader yields (images, angles, labels/ids)
            images = batch[0].to(device)
            angles = batch[1].to(device)

            # Forward pass updates running_mean and running_var
            _ = swa_model(images, angles)


def save_checkpoint(model, optimizer, epoch, best_loss, filename):
    """
    Saves the model checkpoint.
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_loss": best_loss,
    }
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(state, path)


def load_checkpoint(model, filename, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.
    """
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"Loaded checkpoint from {path} (Epoch {epoch}, Loss {best_loss:.6f})")
    return epoch, best_loss


def write_submission(ids, probs, output_path):
    """
    Writes the predictions to a CSV file in the required format.

    Args:
        ids: List or array of image IDs.
        probs: List or array of predicted probabilities (is_iceberg).
        output_path: Full path to save the CSV.
    """
    # Flatten probs if shape is (N, 1)
    if len(probs.shape) > 1:
        probs = probs.flatten()

    df = pd.DataFrame({"id": ids, "is_iceberg": probs})

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
