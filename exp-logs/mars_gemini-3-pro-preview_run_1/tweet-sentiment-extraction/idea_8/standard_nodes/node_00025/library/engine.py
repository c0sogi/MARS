import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.utils import AverageMeter, jaccard
from library.config import Config


def loss_fn(
    start_logits, end_logits, mask_logits, start_targets, end_targets, mask_targets
):
    """
    Computes the combined loss for the Mask-Guided DeBERTa model.

    Args:
        start_logits: Logits for start index (Batch, Seq)
        end_logits: Logits for end index (Batch, Seq)
        mask_logits: Logits for mask (Batch, Seq)
        start_targets: Gaussian smoothed probabilities (Batch, Seq)
        end_targets: Gaussian smoothed probabilities (Batch, Seq)
        mask_targets: Binary mask targets (Batch, Seq)

    Returns:
        torch.Tensor: The weighted total loss.
    """
    # Span Loss: KL Divergence expects LogSoftmax input and Probability targets
    loss_fct_span = nn.KLDivLoss(reduction="batchmean")
    start_loss = loss_fct_span(torch.log_softmax(start_logits, dim=1), start_targets)
    end_loss = loss_fct_span(torch.log_softmax(end_logits, dim=1), end_targets)

    # Average start and end loss for the span component
    span_loss = (start_loss + end_loss) / 2

    # Mask Loss: Binary Cross Entropy
    loss_fct_mask = nn.BCEWithLogitsLoss()
    mask_loss = loss_fct_mask(mask_logits, mask_targets)

    # Combined Loss
    total_loss = (Config.SPAN_LOSS_WEIGHT * span_loss) + (
        Config.MASK_LOSS_WEIGHT * mask_loss
    )

    return total_loss


def decode_batch(start_logits_np, end_logits_np, offsets_np, texts, sentiments):
    """
    Decodes the logits into text spans using Joint Logit Decoding.
    """
    preds = []

    for i in range(len(texts)):
        text = texts[i]
        sentiment = sentiments[i]
        offsets = offsets_np[i]

        # Heuristic: Neutral tweets usually return the full text
        if sentiment == "neutral":
            preds.append(text)
            continue

        s_logits = start_logits_np[i]
        e_logits = end_logits_np[i]

        # Joint Logit Decoding: Maximize (start_logit + end_logit) where start <= end
        # Create sum matrix: (Seq, Seq)
        sum_matrix = s_logits[:, None] + e_logits[None, :]

        # Mask out the lower triangle (where start > end)
        # np.triu returns the upper triangle (including diagonal)
        mask = np.triu(np.ones_like(sum_matrix), k=0).astype(bool)

        # Apply mask: set invalid positions to -inf
        valid_scores = np.where(mask, sum_matrix, -np.inf)

        # Find the indices of the maximum score
        flat_idx = np.argmax(valid_scores)
        best_start = flat_idx // len(e_logits)
        best_end = flat_idx % len(e_logits)

        # Map token indices to character indices using offsets
        start_char = offsets[best_start][0]
        end_char = offsets[best_end][1]

        # Extract substring
        if start_char < end_char:
            pred_string = text[start_char:end_char]
        else:
            # Fallback for degenerate cases (rare)
            pred_string = text

        preds.append(pred_string)

    return preds


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        start_targets = d["start_targets"].to(device)
        end_targets = d["end_targets"].to(device)
        mask_targets = d["mask_targets"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits, mask_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = loss_fn(
            start_logits,
            end_logits,
            mask_logits,
            start_targets,
            end_targets,
            mask_targets,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    print(f"Training Loss: {losses.avg}")
    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            # Targets are available in validation
            start_targets = d["start_targets"].to(device)
            end_targets = d["end_targets"].to(device)
            mask_targets = d["mask_targets"].to(device)

            start_logits, end_logits, mask_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # Compute Validation Loss
            loss = loss_fn(
                start_logits,
                end_logits,
                mask_logits,
                start_targets,
                end_targets,
                mask_targets,
            )
            losses.update(loss.item(), input_ids.size(0))

            # Decode predictions for Jaccard calculation
            start_logits_np = start_logits.cpu().numpy()
            end_logits_np = end_logits.cpu().numpy()
            offsets_np = d["offsets"].numpy()
            texts = d["text"]
            sentiments = d["sentiment"]
            selected_texts = d["selected_text"]

            preds = decode_batch(
                start_logits_np, end_logits_np, offsets_np, texts, sentiments
            )

            # Compute Jaccard
            for pred, target in zip(preds, selected_texts):
                score = jaccard(pred, target)
                jaccards.update(score, 1)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Jaccard: {jaccards.avg}")

    return losses.avg, jaccards.avg


def predict_test_set(test_loader, model, device):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for d in test_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            start_logits, end_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_logits_np = start_logits.cpu().numpy()
            end_logits_np = end_logits.cpu().numpy()
            offsets_np = d["offsets"].numpy()
            texts = d["text"]
            sentiments = d["sentiment"]
            ids = d["textID"]

            preds = decode_batch(
                start_logits_np, end_logits_np, offsets_np, texts, sentiments
            )

            all_ids.extend(ids)
            all_preds.extend(preds)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"textID": all_ids, "selected_text": all_preds})

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
