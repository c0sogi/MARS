import torch
import numpy as np
from library.config import Config


def train_fn(dataloader, model, optimizer, scheduler, device, epoch):
    """
    Performs one epoch of training with mixed precision and gradient accumulation.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The neural network model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Device to run training on.
        epoch: Current epoch number (0-indexed).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    running_loss = 0.0
    dataset_size = 0

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        labels = data["label"].to(device)

        batch_size = input_ids.size(0)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask, token_type_ids, labels)
            loss = outputs["loss"]
            loss = loss / Config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and Config.batch_scheduler:
                scheduler.step()

        # Recover the unscaled loss for reporting
        running_loss += (loss.item() * Config.gradient_accumulation_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.6f}")

    return epoch_loss


def valid_fn(dataloader, model, device):
    """
    Performs inference on the validation or test set.

    Args:
        dataloader: PyTorch DataLoader for validation/test data.
        model: The neural network model.
        device: Device to run inference on.

    Returns:
        tuple: (predictions, ground_truth)
               predictions: numpy array of predicted scores.
               ground_truth: numpy array of true scores (or None if not available).
    """
    model.eval()

    preds = []
    labels_list = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            if "label" in data:
                labels = data["label"].to(device)
                labels_list.append(labels.cpu().numpy())

            # Use autocast for inference as well for potential speedup/consistency
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, token_type_ids)

            logits = outputs["logits"]
            preds.append(logits.cpu().numpy())

    predictions = np.concatenate(preds)

    if len(labels_list) > 0:
        ground_truth = np.concatenate(labels_list)
    else:
        ground_truth = None

    return predictions, ground_truth
