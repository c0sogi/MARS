import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the sum of Cross Entropy Loss for start and end indices with label smoothing.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()

    for step, data in enumerate(data_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        start_positions = data["start_idx"].to(device)
        end_positions = data["end_idx"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set using Jaccard score.
    """
    model.eval()
    jaccards = AverageMeter()

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            # Metadata for decoding and evaluation
            offsets = data["offsets"].cpu().numpy()
            texts = data["text"]
            sentiments = data["sentiment"]
            selected_texts = data["selected_text"]

            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # Move logits to CPU as numpy arrays for efficient decoding
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            for i in range(len(texts)):
                text = texts[i]
                sentiment = sentiments[i]
                target_text = selected_texts[i]
                offset = offsets[i]

                # Neutral Heuristic: Predict full text for neutral sentiment
                if sentiment == "neutral":
                    pred_text = text
                else:
                    # Decoding: Maximize sum of logits subject to start <= end
                    start_l = start_logits[i]
                    end_l = end_logits[i]

                    # Create score matrix: (SeqLen, SeqLen)
                    score_mat = start_l[:, None] + end_l[None, :]

                    # Mask out the lower triangle to enforce start <= end
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    score_mat = np.where(upper_tri_mask == 1, score_mat, -np.inf)

                    # Find indices (start, end) with maximum score
                    best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                    idx_start, idx_end = best_idx

                    # Extract text using offsets
                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    # Handle edge cases (e.g. prediction on special tokens)
                    if char_start == 0 and char_end == 0:
                        pred_text = text
                    else:
                        pred_text = text[char_start:char_end]

                score = jaccard(pred_text, target_text)
                jaccards.update(score, 1)

    return jaccards.avg
