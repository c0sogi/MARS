import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        device (str): 'cuda' or 'cpu'.
        epoch (int): Current epoch index.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, data in enumerate(dataloader):
        # Unpack data: Dataset returns (img, angle, label)
        images, angles, labels = data

        images = images.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)

        # Loss calculation
        # BCEWithLogitsLoss expects logits and float targets of shape (B, 1)
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    # Print metrics with full precision
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for data in dataloader:
            # Dataset returns (img, angle, label)
            images, angles, labels = data

            images = images.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            batch_size = images.size(0)

            outputs = model(images, angles)
            loss = criterion(outputs, labels.view(-1, 1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    print(f"Validation Loss: {avg_loss}")

    return avg_loss


def cyclic_swa_step(
    model,
    swa_model,
    optimizer,
    epoch,
    swa_start_epoch,
    swa_lr_max,
    swa_lr_min,
    swa_cycle_len,
):
    """
    Executes a step of the Low-Energy Cyclic SWA schedule.
    Updates the learning rate based on a cosine schedule within the cycle.
    Updates the SWA model at the end of each cycle.

    Args:
        model (nn.Module): Current model.
        swa_model (AveragedModel): SWA model wrapper.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch.
        swa_start_epoch (int): Epoch to start SWA.
        swa_lr_max (float): Max LR (start of cycle).
        swa_lr_min (float): Min LR (end of cycle).
        swa_cycle_len (int): Length of one cycle in epochs.
    """
    if epoch < swa_start_epoch:
        return

    # Calculate position in cycle
    idx = epoch - swa_start_epoch
    t = idx % swa_cycle_len

    # Cosine Annealing: Max -> Min
    # We want LR to be Max at t=0 and Min at t=cycle_len-1
    denom = max(1, swa_cycle_len - 1)
    fraction = t / denom

    # lr = min + 0.5 * (max - min) * (1 + cos(fraction * pi))
    # fraction=0 -> cos(0)=1 -> lr = max
    # fraction=1 -> cos(pi)=-1 -> lr = min
    lr = swa_lr_min + 0.5 * (swa_lr_max - swa_lr_min) * (
        1 + math.cos(fraction * math.pi)
    )

    # Update optimizer LR
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Check if end of cycle
    if t == swa_cycle_len - 1:
        swa_model.update_parameters(model)
        print(f"Epoch {epoch}: SWA Model Updated (End of Cycle). LR was {lr}")
    else:
        print(f"Epoch {epoch}: Cyclic SWA LR set to {lr}")


def update_bn(loader, swa_model, device):
    """
    Updates Batch Normalization statistics for the SWA model.
    Custom implementation to handle multi-input (image, angle) model.

    Args:
        loader (DataLoader): Training loader.
        swa_model (AveragedModel): SWA model.
        device (str): Device.
    """
    swa_model.train()

    # 1. Reset stats
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None  # Use simple average
            module.num_batches_tracked = torch.tensor(
                0, dtype=torch.long, device=device
            )

    # 2. Accumulate
    print("Updating SWA Batch Normalization statistics...")
    with torch.no_grad():
        for data in loader:
            images, angles, _ = data
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass updates BN stats
            swa_model(images, angles)

    print("SWA Batch Normalization statistics updated.")


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set using TTA and saves to CSV.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device.
        output_path (str): Path to save CSV.
    """
    model.eval()

    # Load test IDs from metadata
    # We assume the loader iterates in the same order as the metadata CSV
    df_test = pd.read_csv(Config.TEST_META)
    ids = df_test["id"].values

    probs = []

    with torch.no_grad():
        for data in test_loader:
            # Test loader yields (img, angle) - no label
            images, angles = data

            images = images.to(device)
            angles = angles.to(device)

            # TTA: Klein Four-Group
            # 1. Original
            out1 = torch.sigmoid(model(images, angles))

            # 2. Horizontal Flip (Flip W, dim 3)
            images_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_h, angles))

            # 3. Vertical Flip (Flip H, dim 2)
            images_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(images_v, angles))

            # 4. Rotate 180 (H + V)
            images_r180 = torch.flip(images, [2, 3])
            out4 = torch.sigmoid(model(images_r180, angles))

            # Average
            avg_out = (out1 + out2 + out3 + out4) / 4.0

            # Flatten and append
            probs.extend(avg_out.view(-1).cpu().numpy())

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": probs})

    # Save
    df_sub.to_csv(output_path, index=False, float_format="%.15f")
    print(f"Submission saved to {output_path}")
