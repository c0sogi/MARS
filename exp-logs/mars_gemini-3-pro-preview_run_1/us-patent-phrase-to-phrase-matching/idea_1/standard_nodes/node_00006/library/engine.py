import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_metrics


def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer instance.
        device (torch.device): The device to run training on.
        scheduler (LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    loss_fn = nn.MSELoss()

    for batch in dataloader:
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Pearson correlation score)
    """
    model.eval()
    running_loss = 0.0
    loss_fn = nn.MSELoss()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Collect predictions and labels
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    pearson_score = compute_metrics(all_preds, all_labels)

    return avg_loss, pearson_score


def fit(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    patience,
    save_path,
    scheduler=None,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): Training data.
        val_dataloader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
        scheduler (LRScheduler, optional): Learning rate scheduler.

    Returns:
        nn.Module: The model with the best weights loaded.
    """
    best_score = -float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, device, scheduler
        )
        val_loss, val_score = evaluate(model, val_dataloader, device)

        # Print full precision metrics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Pearson: {val_score}")

        # Early Stopping Check
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Validation score improved. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model
    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Data to predict on.
        device (torch.device): Device.

    Returns:
        tuple: (List of IDs, List of Scores)
    """
    model.eval()
    all_ids = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Clamp scores to valid range [0, 1]
            scores = torch.clamp(outputs, 0.0, 1.0)

            all_scores.extend(scores.cpu().numpy())
            if "id" in batch:
                all_ids.extend(batch["id"])

    return all_ids, all_scores


def generate_submission(model, test_dataloader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model (nn.Module): The trained model.
        test_dataloader (DataLoader): Test data.
        device (torch.device): Device.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating submission...")
    ids, scores = predict(model, test_dataloader, device)

    submission_df = pd.DataFrame({"id": ids, "score": scores})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
