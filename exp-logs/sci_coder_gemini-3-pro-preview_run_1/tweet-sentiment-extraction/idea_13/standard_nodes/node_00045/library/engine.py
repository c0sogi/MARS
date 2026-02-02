import torch
import numpy as np
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, jaccard
from library.loss import compute_loss


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures parameters for the optimizer with Layer-wise Learning Rate Decay (LLRD).
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # DeBERTa-v3-large has 24 layers
    num_layers = 24

    # 1. Backbone Parameters (LLRD)
    for layer_i in range(num_layers):
        # Determine decay factor: Top layers get higher LR, bottom layers get lower LR
        # Layer 23 (top) -> decay^0 = 1.0
        # Layer 0 (bottom) -> decay^23
        decay_factor = Config.LLRD_DECAY ** (num_layers - 1 - layer_i)

        layer_params = []
        for n, p in model.backbone.encoder.layer[layer_i].named_parameters():
            layer_params.append((n, p))

        group_decay = [
            {
                "params": [
                    p for n, p in layer_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": encoder_lr * decay_factor,
            },
            {
                "params": [
                    p for n, p in layer_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": encoder_lr * decay_factor,
            },
        ]
        optimizer_parameters.extend(group_decay)

    # 2. Embeddings (Lowest LR)
    embedding_decay_factor = Config.LLRD_DECAY**num_layers
    embeddings_params = list(model.backbone.embeddings.named_parameters())
    optimizer_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in embeddings_params
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": encoder_lr * embedding_decay_factor,
            },
            {
                "params": [
                    p for n, p in embeddings_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": encoder_lr * embedding_decay_factor,
            },
        ]
    )

    # 3. Head Parameters (Decoder LR - usually highest)
    # Includes pooler, cnn, dropout, fc, and any other non-backbone modules
    head_params = []
    for n, p in model.named_parameters():
        if "backbone" not in n:
            head_params.append((n, p))

    optimizer_parameters.extend(
        [
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": decoder_lr,
            },
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": decoder_lr,
            },
        ]
    )

    return optimizer_parameters


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, data in enumerate(data_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_targets = data["start_targets"].to(device)
        end_targets = data["end_targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(input_ids, attention_mask)

        # Compute Loss
        loss = compute_loss(
            start_logits, end_logits, start_targets, end_targets, attention_mask
        )

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

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
        for batch_idx, data in enumerate(data_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start_targets = data["start_targets"].to(device)
            end_targets = data["end_targets"].to(device)
            offsets = data["offsets"].cpu().numpy()

            # Metadata for metric calculation
            texts = data["text"]
            selected_texts = data["selected_text"]
            sentiments = data["sentiment"]

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Compute Loss
            loss = compute_loss(
                start_logits, end_logits, start_targets, end_targets, attention_mask
            )
            losses.update(loss.item(), input_ids.size(0))

            # Decoding Logic (Joint Logit Decoding)
            # Move logits to CPU for decoding
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            pred_strings = []

            for i in range(len(input_ids)):
                text = texts[i]
                sentiment = sentiments[i]

                # Although validation loader filters out neutrals, we handle the logic for robustness
                if sentiment == "neutral" or len(text.strip()) == 0:
                    pred_strings.append(text)
                    continue

                start_logit = start_logits[i]
                end_logit = end_logits[i]
                offset = offsets[i]

                # Mask padding in logits to prevent selecting pad tokens
                # attention_mask[i] == 0 means padding
                mask = attention_mask[i].cpu().numpy()
                start_logit[mask == 0] = -1e9
                end_logit[mask == 0] = -1e9

                # Joint decoding: maximize start_logit[s] + end_logit[e] subject to s <= e
                # We can compute this efficiently by adding outer product, but simple loop is fine for batch size
                # Create a matrix of sums
                sum_matrix = start_logit[:, None] + end_logit[None, :]

                # Mask out invalid positions (where start > end)
                # np.tril gives lower triangle (including diagonal), we want upper triangle for s <= e
                # However, indices are (start, end). row is start, col is end.
                # We want row <= col. This is the upper triangle.
                upper_tri_mask = np.triu(np.ones_like(sum_matrix))

                # Apply mask (set invalid to -inf)
                sum_matrix = np.where(upper_tri_mask, sum_matrix, -1e9)

                # Find max
                max_idx = np.argmax(sum_matrix)
                start_idx, end_idx = np.unravel_index(max_idx, sum_matrix.shape)

                # Extract text using offsets
                # offsets contain (start_char, end_char) for each token
                if start_idx < len(offset) and end_idx < len(offset):
                    char_start = offset[start_idx][0]
                    char_end = offset[end_idx][1]
                    pred_text = text[char_start:char_end]
                else:
                    pred_text = text

                pred_strings.append(pred_text)

            # Calculate Jaccard Score
            score = np.mean(
                [jaccard(s, t) for s, t in zip(selected_texts, pred_strings)]
            )
            jaccards.update(score, input_ids.size(0))

    return losses.avg, jaccards.avg
