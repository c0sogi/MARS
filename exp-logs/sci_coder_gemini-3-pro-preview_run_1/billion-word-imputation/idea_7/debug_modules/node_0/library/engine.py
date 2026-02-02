import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from library.config import Config
from library.utils import MetricMonitor, get_device, set_seed
from library.model import SyntaxAwareTransformer
from library.loss import MultiTaskLoss
from library.vocab import load_or_build_artifacts


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch in dataloader:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gap_mask = batch["gap_mask"].to(device)
        loc_targets = batch["loc_targets"].to(device)
        syntax_targets = batch["syntax_targets"].to(device)
        word_targets = batch["word_targets"].to(device)

        # Prepare batch dict for loss function
        batch_data = {
            "gap_mask": gap_mask,
            "loc_targets": loc_targets,
            "syntax_targets": syntax_targets,
            "word_targets": word_targets,
        }

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)

        # Compute loss
        loss_dict = criterion(outputs, batch_data)
        loss = loss_dict["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        # Convert tensor metrics to float for monitoring
        metrics_to_log = {k: v.item() for k, v in loss_dict.items()}
        metric_monitor.update(metrics_to_log)

    return metric_monitor.get_metrics()


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Accuracy metrics.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gap_mask = batch["gap_mask"].to(device)
            loc_targets = batch["loc_targets"].to(device)
            syntax_targets = batch["syntax_targets"].to(device)
            word_targets = batch["word_targets"].to(device)

            batch_data = {
                "gap_mask": gap_mask,
                "loc_targets": loc_targets,
                "syntax_targets": syntax_targets,
                "word_targets": word_targets,
            }

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Compute loss
            loss_dict = criterion(outputs, batch_data)

            # --- Compute Accuracies ---

            # 1. Localization Accuracy (Binary, only at GAP tokens)
            loc_logits = outputs["loc_logits"]  # (B, S)
            loc_preds = (torch.sigmoid(loc_logits) > 0.5).float()

            # Filter by gap mask
            gap_mask_bool = gap_mask.bool()
            valid_loc_preds = loc_preds[gap_mask_bool]
            valid_loc_targets = loc_targets[gap_mask_bool]

            if valid_loc_targets.numel() > 0:
                loc_acc = (valid_loc_preds == valid_loc_targets).float().mean()
            else:
                loc_acc = torch.tensor(0.0, device=device)

            # 2. Syntax Accuracy (Multi-class, only at target gap)
            syntax_logits = outputs["syntax_logits"]  # (B, S, T)
            syntax_preds = torch.argmax(syntax_logits, dim=-1)  # (B, S)

            # Filter by targets != -100
            valid_syn_mask = syntax_targets != -100
            valid_syn_preds = syntax_preds[valid_syn_mask]
            valid_syn_targets = syntax_targets[valid_syn_mask]

            if valid_syn_targets.numel() > 0:
                syn_acc = (valid_syn_preds == valid_syn_targets).float().mean()
            else:
                syn_acc = torch.tensor(0.0, device=device)

            # 3. Identification Accuracy (Multi-class, only at target gap)
            word_logits = outputs["word_logits"]  # (B, S, V)
            word_preds = torch.argmax(word_logits, dim=-1)  # (B, S)

            valid_word_mask = word_targets != -100
            valid_word_preds = word_preds[valid_word_mask]
            valid_word_targets = word_targets[valid_word_mask]

            if valid_word_targets.numel() > 0:
                id_acc = (valid_word_preds == valid_word_targets).float().mean()
            else:
                id_acc = torch.tensor(0.0, device=device)

            # Update metrics
            metrics_to_log = {k: v.item() for k, v in loss_dict.items()}
            metrics_to_log["loc_acc"] = loc_acc.item()
            metrics_to_log["syn_acc"] = syn_acc.item()
            metrics_to_log["id_acc"] = id_acc.item()

            metric_monitor.update(metrics_to_log)

    return metric_monitor.get_metrics()


def train_model(train_loader, val_loader):
    """
    Main function to train the model.
    """
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Initialize Model
    model = SyntaxAwareTransformer().to(device)

    # Initialize Loss
    criterion = MultiTaskLoss().to(device)

    # Initialize Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.WARMUP_PCT,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Print metrics
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_metrics['loss']}")
        print(f"Val Loss: {val_metrics['loss']}")
        print(f"Val Metrics: {val_metrics}")

        # Early Stopping and Checkpointing
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")


def generate_submission(test_loader):
    """
    Generates predictions for the test set and saves to submission.csv.
    Implements Consistency Decoding: Score = P(Loc) * P(Word) * P(Syntax=Tag(Word))
    """
    device = get_device()

    # Load Artifacts
    vocab, pos_map, _ = load_or_build_artifacts(load_cached_data=True)

    # Convert pos_map to tensor for broadcasting
    # pos_map is numpy array (Vocab_Size, ) -> Tensor (Vocab_Size, )
    pos_map_tensor = torch.tensor(pos_map, device=device, dtype=torch.long)

    # Load Model
    model = SyntaxAwareTransformer().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "Warning: No trained model found. Using random initialization (likely to fail)."
        )

    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gap_mask = batch["gap_mask"].to(device)  # (B, S)
            row_ids = batch["row_ids"]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            loc_logits = outputs["loc_logits"]  # (B, S)
            syntax_logits = outputs["syntax_logits"]  # (B, S, T)
            word_logits = outputs["word_logits"]  # (B, S, V)

            # Probabilities
            loc_probs = torch.sigmoid(loc_logits)  # (B, S)
            syntax_probs = torch.softmax(syntax_logits, dim=-1)  # (B, S, T)
            word_probs = torch.softmax(word_logits, dim=-1)  # (B, S, V)

            # --- Consistency Decoding ---
            # We want Score(b, s, w) = loc_probs(b, s) * word_probs(b, s, w) * syntax_probs(b, s, tag(w))

            # 1. Expand syntax probs to match vocab dimension
            # Gather syntax probs for the tag corresponding to each word in vocab
            # syntax_probs: (B, S, T)
            # pos_map_tensor: (V)
            # We want (B, S, V).
            # syntax_probs[:, :, pos_map_tensor] performs the gather on the last dim
            syntax_probs_expanded = syntax_probs[:, :, pos_map_tensor]  # (B, S, V)

            # 2. Compute Final Score
            # loc_probs is (B, S), unsqueeze to (B, S, 1) to broadcast
            final_scores = (
                loc_probs.unsqueeze(-1) * word_probs * syntax_probs_expanded
            )  # (B, S, V)

            # 3. Mask out non-gap positions
            # gap_mask is (B, S). We set scores at non-gap positions to -1 (impossible)
            # gap_mask needs to be broadcastable to (B, S, V)
            mask = gap_mask.unsqueeze(-1).bool()  # (B, S, 1)
            final_scores = final_scores.masked_fill(~mask, -1.0)

            # 4. Find Max
            # Flatten S and V dimensions to find global max per batch item
            batch_size, seq_len, vocab_size = final_scores.shape
            final_scores_flat = final_scores.view(batch_size, -1)  # (B, S*V)

            best_indices_flat = torch.argmax(final_scores_flat, dim=1)  # (B,)

            # Convert flat index back to (s, w)
            best_gap_indices = best_indices_flat // vocab_size
            best_word_indices = best_indices_flat % vocab_size

            # --- Reconstruction ---
            input_ids_cpu = input_ids.cpu().numpy()
            best_gap_indices = best_gap_indices.cpu().numpy()
            best_word_indices = best_word_indices.cpu().numpy()

            for i, row_id in enumerate(row_ids):
                gap_idx = best_gap_indices[i]
                word_idx = best_word_indices[i]

                predicted_word = vocab.lookup_token(word_idx)

                # Reconstruct sentence
                # Input format: [SOS, GAP, w1, GAP, w2, GAP, ..., EOS, PAD...]
                # Original tokens are at input_ids[i] excluding SOS, EOS, GAPs, PADs

                # We iterate through the input_ids and rebuild the list of words
                # inserting the predicted word at the correct location.

                current_seq = input_ids_cpu[i]
                reconstructed_tokens = []

                # Iterate through sequence
                # Logic:
                # - If index == gap_idx: Insert predicted word
                # - If token is a regular word (not special): Append to list
                # - Ignore SOS, EOS, PAD, GAP (unless it's the target gap)

                # Note: The gap_idx corresponds to the index in `input_ids`.
                # If gap_idx is 1 (first gap), it comes before first word.
                # If gap_idx is 3 (second gap), it comes after first word.

                # We can iterate the sequence. If we hit the gap_idx, add predicted word.
                # If we hit a word token, add it.

                for seq_idx, token_id in enumerate(current_seq):
                    if seq_idx == gap_idx:
                        reconstructed_tokens.append(predicted_word)

                    # Check if it is a real word
                    # Real words are not in [SOS, EOS, GAP, PAD]
                    # UNK is a real word contextually, but usually we just keep it if it was in input
                    if token_id not in [
                        Config.SOS_IDX,
                        Config.EOS_IDX,
                        Config.GAP_IDX,
                        Config.PAD_IDX,
                    ]:
                        word = vocab.lookup_token(token_id)
                        reconstructed_tokens.append(word)

                    if token_id == Config.EOS_IDX:
                        break

                # Join with spaces
                # Ensure punctuation handling matches dataset requirements if needed.
                # Dataset uses spaces around punctuation usually.
                pred_sentence = " ".join(reconstructed_tokens)

                predictions.append({"id": row_id, "sentence": pred_sentence})

    # Save to CSV
    df_pred = pd.DataFrame(predictions)

    # Sort by ID to ensure correct order
    df_pred = df_pred.sort_values("id")

    # Escape quotes: replace " with "" and wrap in "
    # Pandas to_csv with quoting=csv.QUOTE_NONNUMERIC or manually formatting
    # The requirement: id,"sentence"
    # double quotes to escape sentence text and two double quotes ("") for double quotes within

    def format_csv_row(row):
        sent = row["sentence"]
        # Escape internal quotes
        sent = sent.replace('"', '""')
        # Wrap in quotes
        return f'{row["id"]},"{sent}"'

    with open(Config.SUBMISSION_FILE, "w", encoding="utf-8") as f:
        f.write('id,"sentence"\n')
        for _, row in df_pred.iterrows():
            f.write(format_csv_row(row) + "\n")

    print(f"Submission saved to {Config.SUBMISSION_FILE}")
