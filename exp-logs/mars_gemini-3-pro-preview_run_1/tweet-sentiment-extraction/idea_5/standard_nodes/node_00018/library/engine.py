import torch
import torch.nn as nn
import numpy as np
import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

from library.config import Config
from library.utils import AverageMeter, jaccard


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence for Start/End Span (Soft Targets)
    """
    # KL Divergence expects log-probabilities as input
    start_log_probs = torch.log_softmax(start_logits, dim=1)
    end_log_probs = torch.log_softmax(end_logits, dim=1)

    # nn.KLDivLoss(reduction='batchmean') is standard for probability distributions
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    loss_start = kl_loss_fn(start_log_probs, start_targets)
    loss_end = kl_loss_fn(end_log_probs, end_targets)

    span_loss = (loss_start + loss_end) / 2.0
    return span_loss


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Standard Optimizer Parameters with Weight Decay handling.
    Separates Backbone (Encoder) and Head (Decoder).
    """
    # Parameters for the custom heads (Decoder)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # Group 1: Custom Heads (Pooling, Conv, Linear layers)
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer_parameters.append(
        {
            "params": [
                p
                for p in head_params
                if not any(nd in getattr(p, "_name", "") for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": decoder_lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": [
                p
                for p in head_params
                if any(nd in getattr(p, "_name", "") for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": decoder_lr,
        }
    )

    # Group 2: Backbone Layers
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    optimizer_parameters.append(
        {
            "params": [
                p
                for p in backbone_params
                if not any(nd in getattr(p, "_name", "") for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": encoder_lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": [
                p
                for p in backbone_params
                if any(nd in getattr(p, "_name", "") for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": encoder_lr,
        }
    )

    return optimizer_parameters


def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    model.train()
    losses = AverageMeter()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)
        # span_masks are unused

        batch_size = input_ids.size(0)

        # Forward Pass
        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        # Backward Pass
        loss.backward()

        # Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.update(loss.item(), batch_size)

        if (step + 1) % Config.print_freq == 0:
            print(f"Epoch {epoch+1} | Step {step+1} | Loss: {losses.avg:.5f}")

    return losses.avg


def eval_fn(data_loader, model, device):
    model.eval()
    jaccard_scores = AverageMeter()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets = batch["offsets"].cpu().numpy()
            raw_texts = batch["text"]
            sentiments = batch["sentiment"]

            # For validation, we have selected_text. For test, we might not.
            # The logic handles both if we just need predictions, but for metric calc we need targets.
            # Assuming this is used for validation where targets exist.
            target_texts = batch.get("selected_text", None)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(input_ids)):
                text = raw_texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                # Neutral Strategy: Always predict full text
                if sentiment == "neutral" or len(text.split()) < 2:
                    pred_text = text
                else:
                    s_prob = start_probs[i]
                    e_prob = end_probs[i]

                    # Compute joint probability matrix
                    # (Seq, 1) + (1, Seq) -> (Seq, Seq)
                    score_mat = s_prob[:, None] + e_prob[None, :]

                    # Enforce start <= end constraint by masking lower triangle
                    # We only care about valid tokens (not padding), but offsets handle mapping back
                    # Using numpy triu to set lower triangle to 0 (or very small number)
                    # However, since we sum probs, 0 is fine if probs are positive.
                    # But probs are small, so let's just use loop or mask.
                    # Simple efficient way:
                    seq_len = len(s_prob)
                    # Create upper triangular mask
                    mask = np.triu(np.ones((seq_len, seq_len)))
                    score_mat = score_mat * mask

                    # Find max
                    max_idx = np.argmax(score_mat)
                    start_idx, end_idx = np.unravel_index(max_idx, score_mat.shape)

                    # Decode to string
                    if start_idx > end_idx:
                        # Fallback (should be covered by mask, but safety check)
                        pred_text = text
                    else:
                        # Map tokens to characters
                        # offset[start_idx] is (start_char, end_char)
                        # offset[end_idx] is (start_char, end_char)

                        # Handle special tokens (offset is (0,0))
                        if offset[start_idx][0] == 0 and offset[start_idx][1] == 0:
                            # Try to find first valid token
                            pass

                        char_start = offset[start_idx][0]
                        char_end = offset[end_idx][1]

                        pred_text = text[char_start:char_end]

                # Calculate Score if targets are available
                if target_texts is not None:
                    score = jaccard(pred_text, target_texts[i])
                    jaccard_scores.update(score, 1)

    return jaccard_scores.avg


def predict_fn(data_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets = batch["offsets"].cpu().numpy()
            raw_texts = batch["text"]
            sentiments = batch["sentiment"]

            # Assuming dataset returns textID if available, but TweetDataset currently doesn't return ID directly.
            # We need to rely on the order being preserved or modify Dataset.
            # Based on library/data.py, TweetDataset returns text, sentiment, etc.
            # We will assume the caller handles ID mapping or we rely on sequential order.
            # The task description says "For each ID in the test set...".
            # We will return a list of strings, and the caller (main script) maps them to IDs.

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(input_ids)):
                text = raw_texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                if sentiment == "neutral" or len(text.split()) < 2:
                    pred_text = text
                else:
                    s_prob = start_probs[i]
                    e_prob = end_probs[i]

                    score_mat = s_prob[:, None] + e_prob[None, :]
                    mask = np.triu(np.ones((len(s_prob), len(s_prob))))
                    score_mat = score_mat * mask

                    max_idx = np.argmax(score_mat)
                    start_idx, end_idx = np.unravel_index(max_idx, score_mat.shape)

                    char_start = offset[start_idx][0]
                    char_end = offset[end_idx][1]

                    pred_text = text[char_start:char_end]

                predictions.append(pred_text)

    return predictions
