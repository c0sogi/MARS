import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard, get_selected_text


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the CrossEntropyLoss with Label Smoothing for start and end indices.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return start_loss + end_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None, scaler=None):
    """
    Executes one training epoch.

    Args:
        data_loader: PyTorch DataLoader for training data.
        model: The TweetModel instance.
        optimizer: The optimizer instance.
        device: The torch device (CPU/GPU).
        scheduler: Optional learning rate scheduler.
        scaler: Optional GradScaler for AMP.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)

        # Handle token_type_ids safely (conditional in dataset)
        token_type_ids = d.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        start_labels = d["start_labels"].to(device)
        end_labels = d["end_labels"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Context
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, start_labels, end_labels)

        if use_amp:
            scaler.scale(loss).backward()
            if Config.CLIP_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), Config.CLIP_GRAD_NORM
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if Config.CLIP_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), Config.CLIP_GRAD_NORM
                )
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        data_loader: PyTorch DataLoader for validation data.
        model: The TweetModel instance.
        device: The torch device.

    Returns:
        tuple: (Average Loss, Average Jaccard Score)
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            token_type_ids = d.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Labels are optional during inference, but expected for eval_fn
            start_labels = d.get("start_labels")
            end_labels = d.get("end_labels")

            if start_labels is not None:
                start_labels = start_labels.to(device)
                end_labels = end_labels.to(device)

            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            if start_labels is not None:
                loss = loss_fn(start_logits, end_logits, start_labels, end_labels)
                losses.update(loss.item(), input_ids.size(0))

            # Decoding for Jaccard Score
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            orig_texts = d["orig_text"]
            sentiments = d["sentiment"]
            offsets = d["offsets"].numpy()
            selected_texts = d.get("selected_text", [])

            for i in range(len(orig_texts)):
                text = orig_texts[i]
                sp = start_probs[i]
                ep = end_probs[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                pred_text = get_selected_text(text, sp, ep, sentiment, offset)

                # Calculate Jaccard if ground truth is available
                if len(selected_texts) > 0:
                    score = jaccard(pred_text, selected_texts[i])
                    jaccards.update(score, 1)

    return losses.avg, jaccards.avg
