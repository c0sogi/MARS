import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the sum of Cross Entropy Loss for start and end token predictions
    with label smoothing.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_targets)
    end_loss = loss_fct(end_logits, end_targets)
    return start_loss + end_loss


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

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(dataloader, model, device):
    """
    Evaluation loop for validation set.
    Computes Loss and Jaccard Score.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start_targets = data["start_targets"].to(device)
            end_targets = data["end_targets"].to(device)
            offsets = data["offsets"].cpu().numpy()
            orig_texts = data["orig_text"]
            sentiments = data["sentiment"]

            start_logits, end_logits = model(input_ids, attention_mask)

            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), input_ids.size(0))

            # Apply Softmax to get probabilities
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                text = orig_texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                # --- Ground Truth Reconstruction ---
                # Reconstruct the true selected text from targets for Jaccard calculation
                s_idx_gt = start_targets[i].item()
                e_idx_gt = end_targets[i].item()

                # Check bounds and validity
                if (
                    s_idx_gt < len(offset)
                    and e_idx_gt < len(offset)
                    and offset[s_idx_gt][0] <= offset[e_idx_gt][1]
                ):
                    gt_text = text[offset[s_idx_gt][0] : offset[e_idx_gt][1]]
                else:
                    # Fallback to full text if indices are invalid (e.g. 0,0 for CLS usually implies empty or full)
                    # In this dataset, if not found, it's usually full text or handled by neutral check
                    gt_text = text

                # --- Prediction Decoding ---
                if sentiment == "neutral":
                    pred_text = text
                else:
                    idx_start = np.argmax(start_probs[i])
                    idx_end = np.argmax(end_probs[i])

                    if idx_start > idx_end:
                        idx_end = idx_start

                    # Extract text using offsets
                    pred_text = text[offset[idx_start][0] : offset[idx_end][1]]

                score = jaccard(pred_text, gt_text)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def infer_fn(dataloader, model, device):
    """
    Inference loop for test set.
    Returns a list of predictions.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            offsets = data["offsets"].cpu().numpy()
            orig_texts = data["orig_text"]
            sentiments = data["sentiment"]
            text_ids = data["text_id"]

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                text = orig_texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]
                id_ = text_ids[i]

                if sentiment == "neutral":
                    pred_text = text
                else:
                    idx_start = np.argmax(start_probs[i])
                    idx_end = np.argmax(end_probs[i])

                    if idx_start > idx_end:
                        idx_end = idx_start

                    pred_text = text[offset[idx_start][0] : offset[idx_end][1]]

                predictions.append({"textID": id_, "selected_text": pred_text})

    return predictions
