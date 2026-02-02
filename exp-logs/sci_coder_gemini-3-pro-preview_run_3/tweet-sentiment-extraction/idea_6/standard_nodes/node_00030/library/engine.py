import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import jaccard
from library.model import loss_fn, decode_prediction


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Training loop for one epoch.

    Args:
        data_loader: DataLoader for training data.
        model: The neural network model.
        optimizer: Optimizer instance.
        device: Device to run training on.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_labels"].to(device)
        end_positions = batch["end_labels"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        if scheduler:
            scheduler.step()

        losses.append(loss.item())

    return np.mean(losses)


def predict_fn(data_loader, model, device):
    """
    Runs inference on the data_loader and returns raw logits.

    Args:
        data_loader: DataLoader for inference data.
        model: The neural network model.
        device: Device to run inference on.

    Returns:
        tuple: (start_logits, end_logits) as numpy arrays.
    """
    model.eval()
    start_logits_list = []
    end_logits_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_logits_list.append(start_logits.cpu().numpy())
            end_logits_list.append(end_logits.cpu().numpy())

    return np.concatenate(start_logits_list), np.concatenate(end_logits_list)


def eval_fn(data_loader, model, device, df, offsets):
    """
    Evaluation loop calculating loss and Jaccard score.

    Args:
        data_loader: DataLoader for validation data.
        model: The neural network model.
        device: Device to run evaluation on.
        df: DataFrame containing the validation data (must be aligned with data_loader).
        offsets: List/Array of token offsets for the validation data.

    Returns:
        tuple: (average_loss, average_jaccard_score)
    """
    model.eval()
    losses = []

    # 1. Calculate Loss
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_labels"].to(device)
            end_positions = batch["end_labels"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
            losses.append(loss.item())

    avg_loss = np.mean(losses)

    # 2. Calculate Jaccard Score
    # Get raw logits from the model
    s_logits, e_logits = predict_fn(data_loader, model, device)

    # Apply Softmax to convert logits to probabilities
    s_probs = np.exp(s_logits) / np.sum(np.exp(s_logits), axis=-1, keepdims=True)
    e_probs = np.exp(e_logits) / np.sum(np.exp(e_logits), axis=-1, keepdims=True)

    jaccards = []

    # Iterate through the DataFrame to decode predictions and compare with ground truth
    # Note: df and offsets must be perfectly aligned with the samples in data_loader
    for i in range(len(df)):
        text = str(df.iloc[i]["text"])
        sentiment = df.iloc[i]["sentiment"]
        selected_text = str(df.iloc[i]["selected_text"])
        off = offsets[i]

        # Decode the prediction using the probabilities and offsets
        pred_str = decode_prediction(s_probs[i], e_probs[i], text, off, sentiment)

        # Calculate Jaccard score
        score = jaccard(selected_text, pred_str)
        jaccards.append(score)

    avg_jaccard = np.mean(jaccards)

    return avg_loss, avg_jaccard
