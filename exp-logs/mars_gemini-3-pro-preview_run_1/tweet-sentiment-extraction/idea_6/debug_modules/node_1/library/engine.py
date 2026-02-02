import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import os
from tqdm import tqdm
from library.utils import jaccard
from library.config import Config


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Calculates the KL Divergence loss for start and end logits against smoothed targets.

    Args:
        start_logits: Model output for start position (batch, seq_len)
        end_logits: Model output for end position (batch, seq_len)
        start_targets: Gaussian smoothed targets for start position (batch, seq_len)
        end_targets: Gaussian smoothed targets for end position (batch, seq_len)

    Returns:
        torch.Tensor: The scalar loss value.
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    # Apply LogSoftmax to logits as required by KLDivLoss (expects log-probabilities)
    start_log_probs = torch.log_softmax(start_logits, dim=1)
    end_log_probs = torch.log_softmax(end_logits, dim=1)

    start_loss = loss_fct(start_log_probs, start_targets)
    end_loss = loss_fct(end_log_probs, end_targets)

    return start_loss + end_loss


def decode_prediction(
    start_logits, end_logits, text, offsets, sentiment, max_len=Config.max_len
):
    """
    Decodes the model output to extract the selected text substring.

    Args:
        start_logits: Tensor of start logits.
        end_logits: Tensor of end logits.
        text: The normalized input text string.
        offsets: Token offsets mapping.
        sentiment: The sentiment label.
        max_len: Maximum sequence length.

    Returns:
        str: The extracted substring.
    """
    # Heuristic: For neutral tweets, the selected text is usually the full text.
    if sentiment == "neutral":
        return text

    start_probs = torch.softmax(start_logits, dim=0).cpu().detach().numpy()
    end_probs = torch.softmax(end_logits, dim=0).cpu().detach().numpy()

    # Ensure we don't go out of bounds if logits are larger than max_len (unlikely with fixed size)
    start_probs = start_probs[:max_len]
    end_probs = end_probs[:max_len]

    # Compute the score matrix: score[i, j] = P_start[i] + P_end[j]
    # Shape: (seq_len, seq_len)
    score_mat = np.expand_dims(start_probs, 1) + np.expand_dims(end_probs, 0)

    # Enforce condition i <= j by keeping only the upper triangle
    score_mat = np.triu(score_mat)

    # Find indices of the maximum score
    best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
    idx_start, idx_end = best_idx

    # Extract text using offsets
    # Offsets are relative to the normalized text provided in data.py
    start_char = offsets[idx_start][0]
    end_char = offsets[idx_end][1]

    # Handle edge case where model predicts special tokens (0,0) or invalid span
    if start_char == 0 and end_char == 0:
        return text

    return text[start_char:end_char]


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set using Jaccard score.
    """
    model.eval()
    jaccard_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Metadata needed for evaluation
            texts = batch["text"]
            selected_texts = batch["selected_text"]
            sentiments = batch["sentiment"]
            offsets_batch = batch["offsets"].numpy()

            start_logits_batch, end_logits_batch = model(input_ids, attention_mask)

            for i in range(len(texts)):
                pred_text = decode_prediction(
                    start_logits_batch[i],
                    end_logits_batch[i],
                    texts[i],
                    offsets_batch[i],
                    sentiments[i],
                )

                score = jaccard(selected_texts[i], pred_text)
                jaccard_scores.append(score)

    return np.mean(jaccard_scores)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    model_save_path,
):
    """
    Main training loop with Early Stopping.
    """
    best_jaccard = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_jaccard = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Model saved! New best Jaccard: {best_jaccard:.5f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model for future use
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model


def predict(data_loader, model, device, output_path):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            texts = batch["text"]
            text_ids = batch["textID"]
            sentiments = batch["sentiment"]
            offsets_batch = batch["offsets"].numpy()

            start_logits_batch, end_logits_batch = model(input_ids, attention_mask)

            for i in range(len(texts)):
                pred_text = decode_prediction(
                    start_logits_batch[i],
                    end_logits_batch[i],
                    texts[i],
                    offsets_batch[i],
                    sentiments[i],
                )

                predictions.append({"textID": text_ids[i], "selected_text": pred_text})

    df_sub = pd.DataFrame(predictions)
    # Ensure quoting is handled correctly (csv.QUOTE_NONNUMERIC quotes all non-numeric fields)
    df_sub.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Submission saved to {output_path}")
