import time
import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, time_since


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, config):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Torch device.
        epoch: Current epoch number.
        config: Configuration object.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    start_time = time.time()

    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=config.FP16)
    criterion = nn.CrossEntropyLoss()

    num_batches = len(dataloader)

    for step, data in enumerate(dataloader):
        # Move data to device
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        scalar_features = data["scalar_features"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # Zero gradients (if not accumulating, though usually done after step)
        # Here we zero at start of accumulation cycle if we were strictly following that pattern,
        # but standard optimizer.zero_grad() location is fine provided we handle accumulation correctly.
        if step % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()

        # Forward pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=config.FP16):
            logits = model(input_ids, attention_mask, scalar_features=scalar_features)
            loss = criterion(logits, labels)

            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

        # Update metrics (scale loss back up for reporting)
        loss_meter.update(loss.item() * config.GRADIENT_ACCUMULATION_STEPS, batch_size)

    print(
        f"Epoch {epoch+1} Train Loss: {loss_meter.avg:.6f} Time: {time_since(start_time, 1.0)}"
    )
    return loss_meter.avg


def validate(model, dataloader, device, config):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: Torch device.
        config: Configuration object.

    Returns:
        float: Average validation loss.
    """
    model.eval()

    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            scalar_features = data["scalar_features"].to(device)
            labels = data["labels"].to(device)

            batch_size = input_ids.size(0)

            with torch.cuda.amp.autocast(enabled=config.FP16):
                logits = model(
                    input_ids, attention_mask, scalar_features=scalar_features
                )
                loss = criterion(logits, labels)

            loss_meter.update(loss.item(), batch_size)

    # Print full precision as requested
    print(f"Val Loss: {loss_meter.avg:.15f} Time: {time_since(start_time, 1.0)}")
    return loss_meter.avg


def inference_fn(model, dataloader, device, config):
    """
    Generates predictions for the test set, optionally using Test-Time Augmentation (TTA).

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for test data.
        device: Torch device.
        config: Configuration object.

    Returns:
        np.ndarray: Array of probability predictions (N, 3).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            scalar_features = data["scalar_features"].to(device)

            # 1. Standard Forward Pass
            with torch.cuda.amp.autocast(enabled=config.FP16):
                logits = model(
                    input_ids, attention_mask, scalar_features=scalar_features
                )

            probs = torch.softmax(logits, dim=1)

            # 2. Test-Time Augmentation (TTA)
            if config.USE_TTA:
                # Swap Branch A and Branch B
                # input_ids shape: (batch, 2, seq_len) -> Flip dim 1
                input_ids_flip = torch.flip(input_ids, dims=[1])
                attention_mask_flip = torch.flip(attention_mask, dims=[1])

                # Swap Scalar Features
                # scalar_features shape: (batch, 3) -> [len_prompt, len_a, len_b]
                # We need [len_prompt, len_b, len_a]
                scalar_features_flip = scalar_features.clone()
                scalar_features_flip[:, 1] = scalar_features[:, 2]
                scalar_features_flip[:, 2] = scalar_features[:, 1]

                with torch.cuda.amp.autocast(enabled=config.FP16):
                    logits_flip = model(
                        input_ids_flip,
                        attention_mask_flip,
                        scalar_features=scalar_features_flip,
                    )

                probs_flip = torch.softmax(logits_flip, dim=1)

                # Realign Probabilities
                # Original Output: [Win A, Win B, Tie]
                # Flipped Output: [Win Input0, Win Input1, Tie] where Input0=B, Input1=A
                # So Flipped Output is effectively [Win B, Win A, Tie]
                # We swap columns 0 and 1 to match the original order
                probs_flip_aligned = probs_flip[:, [1, 0, 2]]

                # Average predictions
                probs = (probs + probs_flip_aligned) / 2.0

            preds.append(probs.cpu().numpy())

    predictions = np.concatenate(preds)
    return predictions
