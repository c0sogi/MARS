import torch
import torch.nn as nn
import numpy as np
import sys
from library.config import Config
from library.utils import compute_spearmanr, save_checkpoint


def train_fn(dataloader, model, optimizer, device, scheduler=None, grad_accum_steps=1):
    """
    Performs one epoch of training.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        device: The device to run on.
        scheduler: (Optional) Learning rate scheduler.
        grad_accum_steps: Number of steps to accumulate gradients.

    Returns:
        float: Average training loss.
    """
    model.train()
    final_loss = 0
    count = 0

    optimizer.zero_grad()

    for i, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        q_mask = data["q_mask"].to(device, dtype=torch.long)
        a_mask = data["a_mask"].to(device, dtype=torch.long)
        labels = data["labels"].to(device, dtype=torch.float)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            q_mask=q_mask,
            a_mask=a_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        final_loss += loss.item()

        loss = loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        count += 1

    return final_loss / count


def eval_fn(dataloader, model, device):
    """
    Performs evaluation on the validation set.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The neural network model.
        device: The device to run on.

    Returns:
        tuple: (Average Validation Loss, Spearman Correlation Score)
    """
    model.eval()
    final_loss = 0
    count = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            q_mask = data["q_mask"].to(device, dtype=torch.long)
            a_mask = data["a_mask"].to(device, dtype=torch.long)
            labels = data["labels"].to(device, dtype=torch.float)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                q_mask=q_mask,
                a_mask=a_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            final_loss += loss.item()
            count += 1

            logits = outputs["logits"]
            preds = torch.sigmoid(logits)

            preds_list.append(preds.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = final_loss / count

    predictions = np.vstack(preds_list)
    targets = np.vstack(targets_list)

    score = compute_spearmanr(targets, predictions)

    return avg_loss, score


def extract_features(dataloader, model, device):
    """
    Extracts features and predictions from the model.
    Used for OOF generation and Test set inference in the stacking pipeline.

    Args:
        dataloader: PyTorch DataLoader.
        model: The neural network model.
        device: The device to run on.

    Returns:
        tuple: (Features numpy array, Predictions numpy array)
    """
    model.eval()
    features_list = []
    preds_list = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            q_mask = data["q_mask"].to(device, dtype=torch.long)
            a_mask = data["a_mask"].to(device, dtype=torch.long)

            # Forward pass (labels=None for inference)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                q_mask=q_mask,
                a_mask=a_mask,
                labels=None,
            )

            features = outputs["features"]
            logits = outputs["logits"]
            preds = torch.sigmoid(logits)

            features_list.append(features.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

    features_arr = np.vstack(features_list)
    preds_arr = np.vstack(preds_list)

    return features_arr, preds_arr


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    scheduler,
    num_epochs,
    save_filename,
    grad_accum_steps=1,
):
    """
    Orchestrates the training process for a model including early stopping.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: Device.
        scheduler: Scheduler.
        num_epochs: Maximum epochs.
        save_filename: Filename to save the best checkpoint (relative to Config.WORKING_DIR).
        grad_accum_steps: Gradient accumulation steps.

    Returns:
        float: Best validation score achieved.
    """
    best_score = -np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_fn(
            train_loader, model, optimizer, device, scheduler, grad_accum_steps
        )
        val_loss, val_score = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
        )

        if val_score > best_score:
            best_score = val_score
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_score, save_filename
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_score
