import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE,
    NUM_EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    MIN_DELTA,
    BATCH_SIZE,
)
from library.utils import jaccard
from library.model import TweetModel
from library.data_loader import get_loaders


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the sum of CrossEntropyLoss for start and end indices.
    """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return start_loss + end_loss


def decode_span(start_logits, end_logits, attention_mask):
    """
    Decodes the best span (start, end) for each sample in the batch.
    Maximizes P(start) * P(end) subject to start <= end.
    """
    batch_size, seq_len = start_logits.size()
    mask = attention_mask.bool()

    # Mask padding tokens by setting logits to a very small number
    start_logits = start_logits.masked_fill(~mask, -1e9)
    end_logits = end_logits.masked_fill(~mask, -1e9)

    # Convert logits to probabilities
    start_probs = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
    end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

    start_preds = []
    end_preds = []

    for k in range(batch_size):
        s_p = start_probs[k]
        e_p = end_probs[k]

        # Compute joint probability matrix: P(start=i, end=j) = P(start=i) * P(end=j)
        score_mat = np.outer(s_p, e_p)

        # Enforce start <= end by masking the lower triangle
        score_mat = np.triu(score_mat)

        # Find the indices maximizing the score
        flat_idx = np.argmax(score_mat)
        i = flat_idx // seq_len
        j = flat_idx % seq_len

        start_preds.append(i)
        end_preds.append(j)

    return start_preds, end_preds


def train_fn(data_loader, model, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_ids = batch["sentiment_id"].to(device)
        start_positions = batch["start_idx"].to(device)
        end_positions = batch["end_idx"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, sentiment_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device, tokenizer):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score.
    """
    model.eval()
    total_loss = 0
    total_jaccard = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_ids = batch["sentiment_id"].to(device)
            start_positions = batch["start_idx"].to(device)
            end_positions = batch["end_idx"].to(device)
            texts = batch["text"]

            start_logits, end_logits = model(input_ids, sentiment_ids, attention_mask)

            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
            total_loss += loss.item()

            # Decode predictions
            start_preds, end_preds = decode_span(
                start_logits, end_logits, attention_mask
            )

            # Calculate Jaccard
            for k in range(len(texts)):
                # Reconstruct Predicted Text using Tokenizer
                i_p, j_p = start_preds[k], end_preds[k]
                ids = input_ids[k].cpu().numpy()

                # Decode the predicted span
                pred_str = tokenizer.decode(
                    ids[i_p : j_p + 1], skip_special_tokens=True
                )

                # Reconstruct Ground Truth Text from labels
                i_t, j_t = start_positions[k].item(), end_positions[k].item()
                gt_str = tokenizer.decode(ids[i_t : j_t + 1], skip_special_tokens=True)

                score = jaccard(pred_str, gt_str)
                total_jaccard += score
                total_samples += 1

    return total_loss / len(data_loader), total_jaccard / total_samples


def predict_fn(data_loader, model, device, tokenizer):
    """
    Runs inference on the test set and saves the submission file.
    """
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_ids = batch["sentiment_id"].to(device)
            text_ids = batch["textID"]

            start_logits, end_logits = model(input_ids, sentiment_ids, attention_mask)
            start_preds, end_preds = decode_span(
                start_logits, end_logits, attention_mask
            )

            for k in range(len(text_ids)):
                i, j = start_preds[k], end_preds[k]
                ids = input_ids[k].cpu().numpy()

                # Decode the predicted span
                pred_str = tokenizer.decode(ids[i : j + 1], skip_special_tokens=True)

                # Clean up spaces (RoBERTa decoding sometimes leaves leading spaces)
                pred_str = pred_str.strip()

                predictions.append({"textID": text_ids[k], "selected_text": pred_str})

    df_sub = pd.DataFrame(predictions)
    # Ensure correct column order
    df_sub = df_sub[["textID", "selected_text"]]
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")


def run():
    """
    Main execution function:
    1. Loads data
    2. Initializes model
    3. Trains with Early Stopping
    4. Predicts on Test Set
    """
    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_loaders(load_cached_data=True)

    # 2. Initialize Model
    print("Initializing model...")
    model = TweetModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    best_jaccard = 0.0
    patience_counter = 0

    print(f"Starting training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, DEVICE)
        val_loss, val_jaccard = eval_fn(val_loader, model, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Jaccard: {val_jaccard}")

        # Early Stopping Logic
        if val_jaccard > best_jaccard + MIN_DELTA:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # 4. Prediction
    print("Loading best model for inference...")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Warning: No best model saved. Using current model.")

    predict_fn(test_loader, model, DEVICE)
