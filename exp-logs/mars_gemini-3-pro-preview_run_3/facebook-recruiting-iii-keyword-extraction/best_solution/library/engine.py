import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    calculate_f1_score,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
)
from library.model import TextCNN


def train_fn(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (torch.device): The computation device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(tokens)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def eval_fn(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (torch.device): The computation device.

    Returns:
        tuple: (Average validation loss, Micro F1 Score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    # Accumulators for Micro F1
    total_tp = 0
    total_pred_pos = 0
    total_actual_pos = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)

            logits = model(tokens)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            # Threshold probabilities at 0.5 to get binary predictions
            preds = (probs > 0.5).float()

            # Update accumulators
            total_tp += (preds * labels).sum().item()
            total_pred_pos += preds.sum().item()
            total_actual_pos += labels.sum().item()

    avg_loss = running_loss / num_batches

    # Calculate Micro F1
    denominator = total_pred_pos + total_actual_pos
    f1 = (2 * total_tp) / denominator if denominator > 0 else 0.0

    return avg_loss, f1


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Test data loader.
        device (torch.device): The computation device.

    Returns:
        tuple: (Predicted probabilities [numpy array], Question IDs [numpy array])
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            ids = batch["id"]

            logits = model(tokens)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids.numpy())

    return np.vstack(all_probs), np.array(all_ids)


def run_training(train_loader, val_loader, test_loader, mlb):
    """
    Orchestrates the training, evaluation, and inference pipeline.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        mlb (MultiLabelBinarizer): Fitted binarizer to convert predictions back to tags.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    # We use len(mlb.classes_) to ensure the output layer matches the fitted label binarizer
    model = TextCNN(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=len(mlb.classes_),
        kernel_sizes=Config.KERNEL_SIZES,
        num_filters=Config.NUM_FILTERS,
        dropout=Config.DROPOUT,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = eval_fn(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, Config.MODEL_SAVE_PATH)
            print("Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model for inference
    print("Loading best model for inference...")
    load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)

    # Generate Predictions
    print("Generating predictions on test set...")
    probs, ids = inference_fn(model, test_loader, device)

    # Convert probabilities to tags
    # Threshold at 0.5
    preds_binary = (probs > 0.5).astype(int)

    # Inverse transform to get tag strings
    # mlb.inverse_transform returns a list of tuples
    pred_tags_tuples = mlb.inverse_transform(preds_binary)
    pred_tags_list = [" ".join(tags) for tags in pred_tags_tuples]

    # Create Submission DataFrame
    df_submission = pd.DataFrame({"Id": ids, "Tags": pred_tags_list})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
