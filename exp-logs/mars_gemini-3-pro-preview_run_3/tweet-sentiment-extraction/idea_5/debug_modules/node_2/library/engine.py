import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard


def loss_fn(
    start_logits, end_logits, content_logits, start_labels, end_labels, content_masks
):
    """
    Computes the Compound Loss for the Dual-Head architecture.

    Args:
        start_logits: (Batch, Seq_Len)
        end_logits: (Batch, Seq_Len)
        content_logits: (Batch, Seq_Len)
        start_labels: (Batch,)
        end_labels: (Batch,)
        content_masks: (Batch, Seq_Len)

    Returns:
        torch.Tensor: The scalar loss value.
    """
    # 1. Pointer Loss (Boundary Detection)
    # Using Label Smoothing to handle fuzzy boundaries
    loss_fct_pointer = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct_pointer(start_logits, start_labels)
    end_loss = loss_fct_pointer(end_logits, end_labels)
    pointer_loss = start_loss + end_loss

    # 2. Content Loss (Semantic Segmentation)
    # Binary Cross Entropy for the token-level mask
    loss_fct_content = nn.BCEWithLogitsLoss()
    content_loss = loss_fct_content(content_logits, content_masks)

    # 3. Total Loss
    total_loss = pointer_loss + (Config.CONTENT_LOSS_WEIGHT * content_loss)
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch.

    Args:
        data_loader: PyTorch DataLoader
        model: The TweetModel
        optimizer: PyTorch Optimizer
        device: torch.device
        scheduler: Learning rate scheduler (optional)

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in data_loader:
        # Move inputs and targets to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_labels = batch["start_labels"].to(device)
        end_labels = batch["end_labels"].to(device)
        content_masks = batch["content_masks"].to(device)

        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits, content_logits = model(input_ids, attention_mask)

        # Compute Loss
        loss = loss_fn(
            start_logits,
            end_logits,
            content_logits,
            start_labels,
            end_labels,
            content_masks,
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device, tokenizer):
    """
    Evaluates the model on the validation set using Hybrid Decoding.

    Args:
        data_loader: PyTorch DataLoader
        model: The TweetModel
        device: torch.device
        tokenizer: Transformers tokenizer for decoding IDs to text

    Returns:
        tuple: (Average Loss, Average Jaccard Score)
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)
            content_masks = batch["content_masks"].to(device)

            # Forward pass
            start_logits, end_logits, content_logits = model(input_ids, attention_mask)

            # Compute Loss (for monitoring)
            loss = loss_fn(
                start_logits,
                end_logits,
                content_logits,
                start_labels,
                end_labels,
                content_masks,
            )
            losses.update(loss.item(), input_ids.size(0))

            # --- Hybrid Decoding Strategy ---

            # Get probabilities
            p_start = torch.softmax(start_logits, dim=1).cpu().numpy()
            p_end = torch.softmax(end_logits, dim=1).cpu().numpy()
            p_content = torch.sigmoid(content_logits).cpu().numpy()

            # Prepare data for decoding
            ids = input_ids.cpu().numpy()
            s_labels = start_labels.cpu().numpy()
            e_labels = end_labels.cpu().numpy()

            for i in range(len(ids)):
                # Per-sample probabilities
                ps = p_start[i]
                pe = p_end[i]
                pc = p_content[i]

                best_score = -float("inf")
                best_start = 0
                best_end = 0

                seq_len = len(ps)

                # Search for best span (i, j) maximizing:
                # Score = P_start[i] + P_end[j] + Mean(P_content[i:j])
                # Optimization: Skip very low probability start tokens to speed up loop
                for s_idx in range(seq_len):
                    if ps[s_idx] < 0.001:
                        continue

                    for e_idx in range(s_idx, seq_len):
                        if pe[e_idx] < 0.001:
                            continue

                        # Calculate mean content relevance for this span
                        # Slice includes e_idx
                        content_score = np.mean(pc[s_idx : e_idx + 1])

                        score = ps[s_idx] + pe[e_idx] + content_score

                        if score > best_score:
                            best_score = score
                            best_start = s_idx
                            best_end = e_idx

                # Decode Prediction
                pred_ids = ids[i][best_start : best_end + 1]
                pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)

                # Decode Ground Truth
                target_ids = ids[i][s_labels[i] : e_labels[i] + 1]
                target_str = tokenizer.decode(target_ids, skip_special_tokens=True)

                # Calculate Jaccard
                score = jaccard(target_str, pred_str)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg
