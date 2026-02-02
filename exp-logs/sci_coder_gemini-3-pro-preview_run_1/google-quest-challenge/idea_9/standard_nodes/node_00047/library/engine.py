import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_metric


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (Scheduler): Learning rate scheduler.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        # Move inputs to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update scheduler (step per batch for warmup schedules)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} Training Loss: {avg_loss}")
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, spearman_score)
    """
    model.eval()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)

    # Concatenate predictions and labels
    all_preds = np.concatenate(preds_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Compute metric
    score = compute_metric(all_labels, all_preds)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Score: {score}")

    return avg_loss, score


def generate_submission(model, dataloader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    preds_list = []
    qa_ids_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            qa_ids = batch["qa_ids"]

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            qa_ids_list.extend(qa_ids.numpy())

    all_preds = np.concatenate(preds_list, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)
    df_sub.insert(0, "qa_id", qa_ids_list)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
