import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard
from library.loss import LabelSmoothingLoss, DistillationLoss


def decode_prediction(start_idx, end_idx, text, offsets, sentiment):
    """
    Decodes the predicted start and end indices into the selected text string.
    Applies the neutral sentiment heuristic and handles offset mapping.
    """
    # Neutral Heuristic: Predict entire text if sentiment is neutral
    # Also fallback for very short texts
    if sentiment == "neutral" or len(text.split()) < 2:
        return text

    # Ensure valid range: if start > end, force end = start
    if start_idx > end_idx:
        end_idx = start_idx

    # Clamp indices to the valid range of offsets
    max_idx = len(offsets) - 1
    start_idx = max(0, min(start_idx, max_idx))
    end_idx = max(0, min(end_idx, max_idx))

    # Extract character start and end positions from offsets
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]

    # Handle special tokens (CLS/SEP) which might have (0,0) offsets
    if char_start == 0 and char_end == 0:
        return text

    return text[char_start:char_end]


def train_fn(
    data_loader, model, optimizer, device, scheduler=None, soft_labels_cache=None
):
    """
    Performs one epoch of training.

    Args:
        data_loader: PyTorch DataLoader.
        model: The neural network model.
        optimizer: The optimizer.
        device: Torch device (CPU/GPU).
        scheduler: Learning rate scheduler (optional).
        soft_labels_cache (dict, optional): A dictionary mapping 'text' -> (start_logits, end_logits).
                                            Used for Stage 2 Distillation.
    """
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    # Select Loss Function based on stage
    if soft_labels_cache is not None:
        criterion = DistillationLoss()
    else:
        criterion = LabelSmoothingLoss()

    # Initialize Scaler for Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    for data in data_loader:
        # Move inputs to device
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        start_targets = data["start_targets"].to(device, dtype=torch.long)
        end_targets = data["end_targets"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass with Automatic Mixed Precision (AMP)
        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            if soft_labels_cache is not None:
                # Stage 2: Distillation
                # Retrieve teacher logits for the current batch using text as key
                texts = data["text"]
                t_start_list = []
                t_end_list = []

                for t in texts:
                    if t in soft_labels_cache:
                        ts, te = soft_labels_cache[t]
                        t_start_list.append(torch.tensor(ts))
                        t_end_list.append(torch.tensor(te))
                    else:
                        # Fallback if text not found (should be rare)
                        # Use current logits detached to result in 0 KL loss, relying only on CE
                        t_start_list.append(start_logits[0].detach().cpu())
                        t_end_list.append(end_logits[0].detach().cpu())

                teacher_start_logits = torch.stack(t_start_list).to(device)
                teacher_end_logits = torch.stack(t_end_list).to(device)

                loss = criterion(
                    start_logits,
                    end_logits,
                    teacher_start_logits,
                    teacher_end_logits,
                    start_targets,
                    end_targets,
                )
            else:
                # Stage 1: Standard Training with Label Smoothing
                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                loss = loss_start + loss_end

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Calculate Training Metrics (Jaccard)
        # Detach and move to CPU for metric calculation
        start_preds = torch.argmax(start_logits, dim=1).detach().cpu().numpy()
        end_preds = torch.argmax(end_logits, dim=1).detach().cpu().numpy()

        texts = data["text"]
        offsets = data["offsets"].numpy()
        sentiments = data["sentiment"]
        selected_texts = data["selected_text"]

        batch_jaccard_scores = []
        for i in range(len(texts)):
            pred_text = decode_prediction(
                start_preds[i], end_preds[i], texts[i], offsets[i], sentiments[i]
            )
            score = jaccard(selected_texts[i], pred_text)
            batch_jaccard_scores.append(score)

        losses.update(loss.item(), len(texts))
        jaccards.update(np.mean(batch_jaccard_scores), len(texts))

    print(f"Train Loss: {losses.avg} | Train Jaccard: {jaccards.avg}")


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on validation or test set.

    Returns:
        avg_jaccard (float): Average Jaccard score.
        final_predictions (list): List of predicted strings.
        logits_dict (dict): Dictionary mapping text -> (start_logits, end_logits) for OOF.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    final_predictions = []
    logits_dict = {}  # text -> (start_logits, end_logits)

    # Use standard CrossEntropy for validation loss monitoring
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

            # Check if targets exist (Validation vs Test)
            has_targets = "start_targets" in data
            if has_targets:
                start_targets = data["start_targets"].to(device, dtype=torch.long)
                end_targets = data["end_targets"].to(device, dtype=torch.long)

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            if has_targets:
                loss = criterion(start_logits, start_targets) + criterion(
                    end_logits, end_targets
                )
                losses.update(loss.item(), len(input_ids))

            # Decode predictions
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            start_preds = np.argmax(start_probs, axis=1)
            end_preds = np.argmax(end_probs, axis=1)

            texts = data["text"]
            offsets = data["offsets"].numpy()
            sentiments = data["sentiment"]

            # If validation/train, we have selected_text to compute Jaccard
            if "selected_text" in data:
                selected_texts = data["selected_text"]
                batch_jaccard_scores = []

                for i in range(len(texts)):
                    pred_text = decode_prediction(
                        start_preds[i],
                        end_preds[i],
                        texts[i],
                        offsets[i],
                        sentiments[i],
                    )
                    score = jaccard(selected_texts[i], pred_text)
                    batch_jaccard_scores.append(score)

                    final_predictions.append(pred_text)
                    # Store logits for OOF generation
                    logits_dict[texts[i]] = (
                        start_logits[i].cpu().numpy(),
                        end_logits[i].cpu().numpy(),
                    )

                jaccards.update(np.mean(batch_jaccard_scores), len(texts))
            else:
                # Test set (no selected_text available)
                for i in range(len(texts)):
                    pred_text = decode_prediction(
                        start_preds[i],
                        end_preds[i],
                        texts[i],
                        offsets[i],
                        sentiments[i],
                    )
                    final_predictions.append(pred_text)

    print(f"Eval Loss: {losses.avg} | Eval Jaccard: {jaccards.avg}")

    return jaccards.avg, final_predictions, logits_dict
