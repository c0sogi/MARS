import torch
import torch.nn as nn
import numpy as np
import os
from transformers import get_linear_schedule_with_warmup
from library.utils import AverageMeter, jaccard
from library.model import TweetModel
from library.data import get_fold_dls


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for d in data_loader:
        input_ids = d["ids"].to(device)
        attention_mask = d["mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        target_start = d["target_start"].to(device)
        target_end = d["target_end"].to(device)

        optimizer.zero_grad()

        loss, _, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=target_start,
            end_positions=target_end,
        )

        loss.backward()
        optimizer.step()
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
        for d in data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            target_start = d["target_start"].to(device)
            target_end = d["target_end"].to(device)
            offsets = d["offsets"].cpu().numpy()
            orig_texts = d["orig_text"]
            sentiments = d["sentiment"]

            # Forward pass
            loss, start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=target_start,
                end_positions=target_end,
            )
            losses.update(loss.item(), input_ids.size(0))

            # Get probabilities
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            # Move auxiliary data to CPU for decoding
            target_start = target_start.cpu().numpy()
            target_end = target_end.cpu().numpy()
            mask_np = attention_mask.cpu().numpy()

            # Decode batch
            for i in range(len(orig_texts)):
                # Re-process text to match the logic used during tokenization (in library/data.py)
                # This ensures offsets align correctly with the string we slice
                text = " " + " ".join(orig_texts[i].split())
                sentiment = sentiments[i]
                offset = offsets[i]

                # --- 1. Reconstruct Ground Truth ---
                t_s = target_start[i]
                t_e = target_end[i]

                # Ensure indices are within valid range and start <= end
                if t_s < len(offset) and t_e < len(offset) and t_s <= t_e:
                    true_str = text[offset[t_s][0] : offset[t_e][1]]
                else:
                    true_str = text

                # --- 2. Generate Prediction ---
                if sentiment == "neutral":
                    # Strategy: For neutral tweets, prediction is the full text
                    pred_str = text
                else:
                    # Strategy: Maximize joint probability P(start) * P(end) subject to start <= end
                    s_p = start_probs[i] * mask_np[i]  # Mask padding tokens
                    e_p = end_probs[i] * mask_np[i]

                    # Compute outer product to get matrix of all start-end combinations
                    scores = np.outer(s_p, e_p)

                    # Mask invalid combinations where end < start (lower triangle)
                    scores = np.triu(scores)

                    # Find indices of maximum score
                    best_idx = np.argmax(scores)
                    best_s, best_e = np.unravel_index(best_idx, scores.shape)

                    if best_s < len(offset) and best_e < len(offset):
                        pred_str = text[offset[best_s][0] : offset[best_e][1]]
                    else:
                        pred_str = text

                # --- 3. Compute Metric ---
                score = jaccard(pred_str, true_str)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def train_fold(fold, config):
    """
    Orchestrates the training for a single fold.
    Includes optimizer setup, scheduler, training loop, and early stopping.
    """
    print(f"--- Training Fold {fold} ---")
    device = config.device

    # Load Data
    train_dl, val_dl = get_fold_dls(fold, config)

    # Initialize Model
    model = TweetModel(config.model_name, config.dropout)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Scheduler
    num_train_steps = int(len(train_dl) * config.epochs)
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Training State
    best_jaccard = 0.0
    patience = 3
    counter = 0

    for epoch in range(config.epochs):
        train_loss = train_fn(train_dl, model, optimizer, device, scheduler)
        val_loss, val_jaccard = eval_fn(val_dl, model, device)

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        # Checkpoint & Early Stopping
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), config.get_model_path(fold))
            counter = 0
            print(f"  -> Saved Best Model (Jaccard: {best_jaccard:.5f})")
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"  -> Early Stopping triggered after {patience} epochs of no improvement."
                )
                break

    # Cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_jaccard
