import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import AverageMeter, jaccard, normalize_text


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence loss for start and end logits against soft targets.
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    # Apply log_softmax to logits for KLDivLoss
    start_log_probs = torch.log_softmax(start_logits, dim=1)
    end_log_probs = torch.log_softmax(end_logits, dim=1)

    start_loss = loss_fct(start_log_probs, start_targets)
    end_loss = loss_fct(end_log_probs, end_targets)

    return 0.5 * (start_loss + end_loss)


def train_fn(dataloader, model, optimizer, device, scheduler=None):
    """
    Training loop for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for data in dataloader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_targets = data["start_targets"].to(device)
        end_targets = data["end_targets"].to(device)
        sentiment = data["sentiment"]

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        # Compute loss
        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(dataloader, model, device):
    """
    Evaluation loop for validation set. Computes Loss and Jaccard score.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            sentiment = data["sentiment"]
            texts = data["text"]
            selected_texts = data["selected_text"]
            offsets = data["offsets"].cpu().numpy()

            # Validation set has targets
            start_targets = data["start_targets"].to(device)
            end_targets = data["end_targets"].to(device)

            logits = model(input_ids, attention_mask)
            start_logits = logits[:, :, 0]
            end_logits = logits[:, :, 1]

            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), input_ids.size(0))

            # Move logits to CPU for decoding
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            # Decode batch
            for i in range(len(texts)):
                orig_text = texts[i]
                current_sentiment = sentiment[i]

                # Neutral Strategy: Predict full text
                if current_sentiment == "neutral":
                    pred_text = orig_text
                else:
                    # Pos/Neg Strategy: Span Extraction
                    start_pred = start_logits[i]
                    end_pred = end_logits[i]

                    # Score matrix: S[k, m] = start[k] + end[m]
                    scores = start_pred[:, None] + end_pred[None, :]

                    # Mask invalid spans where end < start (upper triangle is valid)
                    upper_tri_mask = np.triu(np.ones_like(scores))
                    scores = np.where(upper_tri_mask == 1, scores, -np.inf)

                    # Find max score
                    max_idx = np.argmax(scores)
                    start_idx, end_idx = np.unravel_index(max_idx, scores.shape)

                    # Extract substring using offsets on NORMALIZED text
                    norm_text = normalize_text(orig_text)
                    current_offsets = offsets[i]

                    # Ensure indices are within bounds
                    if start_idx < len(current_offsets) and end_idx < len(
                        current_offsets
                    ):
                        start_char = current_offsets[start_idx][0]
                        end_char = current_offsets[end_idx][1]

                        # Handle edge cases where offsets might be 0,0 for special tokens
                        if end_char > start_char and end_char <= len(norm_text):
                            pred_text = norm_text[start_char:end_char]
                        else:
                            # Fallback if extraction fails or is empty
                            pred_text = norm_text
                    else:
                        pred_text = norm_text

                # Compute Jaccard
                score = jaccard(pred_text, selected_texts[i])
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def generate_submission(dataloader, model, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating predictions for test set...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            sentiment = data["sentiment"]
            texts = data["text"]
            offsets = data["offsets"].cpu().numpy()

            logits = model(input_ids, attention_mask)
            start_logits = logits[:, :, 0].cpu().numpy()
            end_logits = logits[:, :, 1].cpu().numpy()

            for i in range(len(texts)):
                orig_text = texts[i]
                current_sentiment = sentiment[i]

                if current_sentiment == "neutral":
                    pred_text = orig_text
                else:
                    start_pred = start_logits[i]
                    end_pred = end_logits[i]

                    scores = start_pred[:, None] + end_pred[None, :]
                    upper_tri_mask = np.triu(np.ones_like(scores))
                    scores = np.where(upper_tri_mask == 1, scores, -np.inf)

                    max_idx = np.argmax(scores)
                    start_idx, end_idx = np.unravel_index(max_idx, scores.shape)

                    norm_text = normalize_text(orig_text)
                    current_offsets = offsets[i]

                    if start_idx < len(current_offsets) and end_idx < len(
                        current_offsets
                    ):
                        start_char = current_offsets[start_idx][0]
                        end_char = current_offsets[end_idx][1]

                        if end_char > start_char and end_char <= len(norm_text):
                            pred_text = norm_text[start_char:end_char]
                        else:
                            pred_text = norm_text
                    else:
                        pred_text = norm_text

                predictions.append(pred_text)

    # Load test metadata to get textIDs
    test_df = pd.read_csv(Config.TEST_META)

    # In DEBUG mode, we only process a subset of the data
    if Config.DEBUG:
        test_df = test_df.iloc[: len(predictions)]

    test_df["selected_text"] = predictions

    # Create submission dataframe
    submission = test_df[["textID", "selected_text"]]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(
    model, train_loader, val_loader, test_loader, optimizer, scheduler, device, epochs
):
    """
    Main execution function: Training -> Validation -> Saving -> Submission.
    """
    best_jaccard = 0.0
    model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training for {epochs} epochs on device {device}...")

    for epoch in range(epochs):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with Jaccard: {best_jaccard:.5f}")

    print(f"Training complete. Best Jaccard: {best_jaccard:.5f}")

    # Load best model for submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate submission
    generate_submission(test_loader, model, device)
