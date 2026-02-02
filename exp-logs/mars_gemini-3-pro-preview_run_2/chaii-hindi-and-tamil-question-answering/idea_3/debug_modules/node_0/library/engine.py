import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: The training dataloader.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to train on.
        epoch: The current epoch number.

    Returns:
        float: Average training loss.
    """
    model.train()
    scaler = GradScaler(enabled=Config.FP16)
    losses = AverageMeter()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=Config.FP16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss

        # Track loss
        losses.update(loss.item(), input_ids.size(0))

        # Backward pass
        scaler.scale(loss).backward()

        # Unscale and clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

    print(f"Epoch {epoch+1} Training Loss: {losses.avg}")
    return losses.avg


def validate_one_epoch(model, dataloader, device):
    """
    Validates the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        device: The device to validate on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            with autocast(enabled=Config.FP16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                loss = outputs.loss

            losses.update(loss.item(), input_ids.size(0))

    # Print full precision as requested
    print(f"Validation Loss: {losses.avg}")
    return losses.avg


def get_predictions(model, dataloader, device):
    """
    Generates predictions (logits) for the given dataloader.

    Args:
        model: The PyTorch model.
        dataloader: The dataloader (test/val).
        device: The device.

    Returns:
        tuple: (start_logits, end_logits) as numpy arrays.
    """
    model.eval()
    start_logits_list = []
    end_logits_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(enabled=Config.FP16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Move to CPU and numpy
            start_logits_list.append(outputs.start_logits.cpu().numpy())
            end_logits_list.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits_list, axis=0)
    end_logits = np.concatenate(end_logits_list, axis=0)

    return start_logits, end_logits
