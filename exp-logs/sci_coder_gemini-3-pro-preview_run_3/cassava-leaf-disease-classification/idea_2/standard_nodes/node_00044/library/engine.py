import time
import numpy as np
import torch
from library.config import CFG
from library.utils import AverageMeter
from library.mixup import mixup_data, cutmix_data


def train_one_epoch(
    epoch, model, train_loader, optimizer, device, criterion, scaler=None
):
    """
    Performs one training epoch with MixUp and CutMix augmentation.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run on.
        criterion (loss): The loss function.
        scaler (GradScaler, optional): Scaler for AMP.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    summary_loss = AverageMeter()
    start_time = time.time()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Randomly decide which augmentation to apply based on CFG probabilities
        r = np.random.rand()

        # Cite solution_lesson_node_00038: Automatic Mixed Precision
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            if r < CFG.mixup_prob:
                # Apply MixUp
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=1.0)
                outputs = model(images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                    outputs, y_b
                )
            elif r < (CFG.mixup_prob + CFG.cutmix_prob):
                # Apply CutMix
                images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
                outputs = model(images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                    outputs, y_b
                )
            else:
                # Normal training
                outputs = model(images)
                loss = criterion(outputs, labels)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        summary_loss.update(loss.item(), batch_size)

        if CFG.print_freq > 0 and step % CFG.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{step}/{len(train_loader)}] "
                f"Loss: {summary_loss.val:.4f} ({summary_loss.avg:.4f}) "
                f"Time: {time.time() - start_time:.2f}s"
            )

    return summary_loss.avg


def valid_one_epoch(epoch, model, val_loader, device, criterion):
    """
    Performs one validation epoch.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): The device to run on.
        criterion (loss): The loss function.

    Returns:
        tuple: (Average Loss, Accuracy)
    """
    model.eval()
    summary_loss = AverageMeter()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            summary_loss.update(loss.item(), batch_size)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            if CFG.print_freq > 0 and step % CFG.print_freq == 0:
                print(
                    f"EVAL: [{epoch}][{step}/{len(val_loader)}] "
                    f"Loss: {summary_loss.val:.4f} ({summary_loss.avg:.4f})"
                )

    accuracy = correct / total

    # Print full precision metrics as requested
    print(f"Epoch {epoch} Validation - Loss: {summary_loss.avg}, Accuracy: {accuracy}")

    return summary_loss.avg, accuracy
