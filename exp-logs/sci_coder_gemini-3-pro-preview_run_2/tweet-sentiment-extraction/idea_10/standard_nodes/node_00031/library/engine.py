import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from library.utils import AverageMeter, jaccard, AWP


def loss_fn(start_logits, end_logits, start_labels, end_labels, config):
    """
    Computes the sum of CrossEntropyLoss for start and end indices
    with label smoothing.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    start_loss = loss_fct(start_logits, start_labels)
    end_loss = loss_fct(end_logits, end_labels)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler, config, epoch):
    """
    Executes the training loop for a single epoch with Mixed Precision and AWP.
    """
    model.train()
    losses = AverageMeter()

    # Initialize scaler for Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    # Initialize AWP
    awp = AWP(model, optimizer, adv_lr=config.awp_lr, adv_eps=config.awp_eps)

    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Train Epoch {epoch+1}")

    for step, batch in enumerate(pbar):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        start_labels = batch["start_labels"].to(device)
        end_labels = batch["end_labels"].to(device)

        # -------------------------------------------------------
        # 1. Standard Forward & Backward
        # -------------------------------------------------------
        with torch.amp.autocast("cuda"):
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, start_labels, end_labels, config)

        # Scale loss and backward
        scaler.scale(loss).backward()

        # -------------------------------------------------------
        # 2. Adversarial Weight Perturbation (AWP)
        # -------------------------------------------------------
        if config.use_awp and epoch >= config.awp_start_epoch:
            # Perturb weights based on gradients from the standard step
            awp.attack()

            # Forward pass with perturbed weights
            with torch.amp.autocast("cuda"):
                start_logits_adv, end_logits_adv = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss_adv = loss_fn(
                    start_logits_adv, end_logits_adv, start_labels, end_labels, config
                )

            # Backward pass to accumulate gradients from robust loss
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # -------------------------------------------------------
        # 3. Optimization Step
        # -------------------------------------------------------
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        # Update logs
        losses.update(loss.item(), input_ids.size(0))
        pbar.set_postfix(loss=losses.avg)

    return losses.avg


def eval_fn(data_loader, model, device, config):
    """
    Evaluates the model on the validation set, computing Loss and Jaccard Score.
    Handles decoding logic including the neutral sentiment heuristic.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    pbar = tqdm(data_loader, total=len(data_loader), desc="Eval")

    with torch.no_grad():
        for batch in pbar:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)

            # Meta-data for decoding
            offsets = batch["offsets"].numpy()
            orig_texts = batch["orig_text"]
            sentiments = batch["sentiment"]
            selected_texts = batch["selected_text"]

            # Forward pass
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = loss_fn(start_logits, end_logits, start_labels, end_labels, config)
            losses.update(loss.item(), input_ids.size(0))

            # Convert logits to probabilities
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            # Decoding Loop
            for i in range(len(orig_texts)):
                orig_text = orig_texts[i]
                sentiment = sentiments[i]
                target_text = selected_texts[i]

                # Neutral Heuristic: Predict entire text
                if sentiment == "neutral":
                    pred_text = orig_text
                else:
                    # Find best (start, end) pair where start <= end
                    start_p = start_probs[i]
                    end_p = end_probs[i]

                    # Compute joint probability matrix: (seq_len, seq_len)
                    score_matrix = np.outer(start_p, end_p)

                    # Mask out invalid positions (end < start) by setting upper triangle to 0 (keep lower? no, keep upper)
                    # We want start <= end. Matrix rows are start, cols are end.
                    # So we want col >= row. This is the upper triangle.
                    # np.triu returns upper triangle.
                    score_matrix = np.triu(score_matrix)

                    # Find indices of maximum score
                    best_idx = np.argmax(score_matrix)
                    best_start_idx, best_end_idx = np.unravel_index(
                        best_idx, score_matrix.shape
                    )

                    # Map tokens to character offsets
                    # offsets[i] is shape (seq_len, 2) -> (start_char, end_char)
                    char_start = offsets[i][best_start_idx][0]
                    char_end = offsets[i][best_end_idx][1]

                    pred_text = orig_text[char_start:char_end]

                # Calculate Jaccard
                score = jaccard(target_text, pred_text)
                jaccards.update(score, 1)

            pbar.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Jaccard: {jaccards.avg}")

    return losses.avg, jaccards.avg
