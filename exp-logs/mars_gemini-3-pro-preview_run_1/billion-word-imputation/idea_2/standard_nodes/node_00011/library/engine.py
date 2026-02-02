import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, calculate_levenshtein
from library.model import GapTransformer

logger = setup_logger("engine")


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler, device, epoch
):
    """
    Trains the model for one epoch using Automatic Mixed Precision.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
            logits = model(input_ids)
            # Flatten logits and targets for CrossEntropyLoss
            # logits: (B, L, V) -> (B*L, V)
            # targets: (B, L) -> (B*L)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if (i + 1) % 2000 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {i+1}/{num_batches} | Loss: {loss.item():.6f}"
            )

    return total_loss / num_batches


def validate(model, dataloader, vocab, device):
    """
    Evaluates the model by reconstructing sentences and computing
    the Levenshtein distance between the predicted and ground truth sentences.
    """
    model.eval()
    total_distance = 0.0
    total_samples = 0

    pad_idx = vocab.get_pad_index()
    unk_idx = vocab.get_unk_index()
    no_insert_idx = vocab.get_no_insert_index()
    vocab_size = len(vocab)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(input_ids)

            # Mask invalid tokens to ensure we pick a real word
            logits[:, :, pad_idx] = float("-inf")
            logits[:, :, unk_idx] = float("-inf")
            logits[:, :, no_insert_idx] = float("-inf")

            # Find global maximum probability for insertion (Position, Word) per sentence
            B, L, V = logits.shape
            flat_logits = logits.view(B, -1)
            _, flat_indices = torch.max(flat_logits, dim=1)

            pred_pos_indices = (flat_indices // V).cpu().numpy()
            pred_word_indices = (flat_indices % V).cpu().numpy()

            input_ids_np = input_ids.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for b in range(B):
                # --- 1. Reconstruct Ground Truth Sentence ---
                curr_input = input_ids_np[b]
                curr_target = targets_np[b]

                # Extract valid input tokens (ignoring padding)
                valid_len = 0
                input_tokens = []
                for tid in curr_input:
                    if tid == pad_idx:
                        break
                    input_tokens.append(vocab.itos[tid])
                    valid_len += 1

                # Find where the target word was removed (marked by non-NO_INSERT in targets)
                # In validation set, there is exactly one insertion per sentence.
                t_indices = np.where(curr_target[:valid_len] != no_insert_idx)[0]

                if len(t_indices) > 0:
                    t_pos = t_indices[0]
                    t_word_idx = curr_target[t_pos]
                    t_word = vocab.itos[t_word_idx]

                    # Insert target word back
                    ref_tokens = list(input_tokens)
                    ref_tokens.insert(t_pos + 1, t_word)
                    ref_sentence = " ".join(ref_tokens)
                else:
                    # Fallback (should not happen with valid data)
                    ref_sentence = " ".join(input_tokens)

                # --- 2. Reconstruct Predicted Sentence ---
                p_pos = pred_pos_indices[b]
                p_word_idx = pred_word_indices[b]
                p_word = vocab.itos[p_word_idx]

                # Clamp position to valid length
                if p_pos >= valid_len:
                    p_pos = valid_len - 1

                hyp_tokens = list(input_tokens)
                hyp_tokens.insert(p_pos + 1, p_word)
                hyp_sentence = " ".join(hyp_tokens)

                # --- 3. Compute Metric ---
                dist = calculate_levenshtein(ref_sentence, hyp_sentence)
                total_distance += dist
                total_samples += 1

    return total_distance / total_samples if total_samples > 0 else 0.0


def run_training(model, train_loader, val_loader, vocab):
    """
    Orchestrates the training process, including optimization, scheduling,
    and early stopping based on Levenshtein distance.
    """
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Loss Configuration: Down-weight the majority [NO_INSERT] class
    weights = torch.ones(len(vocab)).to(device)
    weights[vocab.get_no_insert_index()] = Config.NO_INSERT_WEIGHT
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=vocab.get_pad_index())

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps, pct_start=0.1
    )

    # Gradient Scaler for AMP
    scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_AMP)

    best_metric = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    logger.info(f"Starting training on {device}...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        val_metric = validate(model, val_loader, vocab, device)

        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Levenshtein: {val_metric:.6f}"
        )

        # Early Stopping Logic based on Levenshtein Distance
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(
                f"New best model saved to {save_path} with Levenshtein: {val_metric:.6f}"
            )
        else:
            patience_counter += 1
            logger.info(f"Patience {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # Load best model for final return
    if os.path.exists(save_path):
        logger.info(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def generate_submission(model, test_loader, vocab):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    device = torch.device(Config.DEVICE)
    model.eval()
    model.to(device)

    predictions = []
    ids = []

    pad_idx = vocab.get_pad_index()
    unk_idx = vocab.get_unk_index()
    no_insert_idx = vocab.get_no_insert_index()
    vocab_size = len(vocab)

    logger.info("Generating submission predictions...")

    with torch.no_grad():
        for batch in test_loader:
            batch_ids = batch["id"].numpy()
            input_ids = batch["input_ids"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(input_ids)

            # Mask invalid tokens
            logits[:, :, pad_idx] = float("-inf")
            logits[:, :, unk_idx] = float("-inf")
            logits[:, :, no_insert_idx] = float("-inf")

            # Find global max
            B, L, V = logits.shape
            flat_logits = logits.view(B, -1)
            _, flat_indices = torch.max(flat_logits, dim=1)

            pred_pos_indices = (flat_indices // V).cpu().numpy()
            pred_word_indices = (flat_indices % V).cpu().numpy()

            input_ids_np = input_ids.cpu().numpy()

            for i in range(len(batch_ids)):
                row_id = batch_ids[i]
                p_pos = pred_pos_indices[i]
                p_word_idx = pred_word_indices[i]

                # Reconstruct sentence
                curr_input = input_ids_np[i]
                valid_tokens = []
                for tid in curr_input:
                    if tid == pad_idx:
                        break
                    valid_tokens.append(vocab.itos[tid])

                # Clamp position
                if p_pos >= len(valid_tokens):
                    p_pos = len(valid_tokens) - 1

                word_to_insert = vocab.itos[p_word_idx]
                valid_tokens.insert(p_pos + 1, word_to_insert)

                pred_sentence = " ".join(valid_tokens)

                ids.append(row_id)
                predictions.append(pred_sentence)

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df = pd.DataFrame({"id": ids, "sentence": predictions})
    df.to_csv(Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
