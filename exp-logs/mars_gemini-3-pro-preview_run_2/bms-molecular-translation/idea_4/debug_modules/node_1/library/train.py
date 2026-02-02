import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    compute_levenshtein,
    seed_everything,
)
from library.dataset import get_train_dataloader, get_val_dataloader
from library.model import ShowAttendTell
from library.tokenizer import Tokenizer


def train_one_epoch(
    train_loader, model, criterion, optimizer, device, epoch, tokenizer
):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()

    # Set teacher forcing ratio based on config
    teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO

    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        # outputs shape: (batch_size, max_len, vocab_size)
        outputs = model(images, captions, teacher_forcing_ratio=teacher_forcing_ratio)

        # Calculate loss
        # The model outputs predictions for t=1 to max_len-1 (aligned with captions[:, 1:])
        # outputs[:, 0, :] corresponds to the initial step (often unused or SOS)

        # Targets: (batch_size, max_len-1) -> exclude SOS at index 0
        targets = captions[:, 1:]

        # Predictions: (batch_size, max_len-1, vocab_size) -> exclude index 0
        predictions = outputs[:, 1:, :]

        # Reshape for CrossEntropyLoss: (N, C) and (N)
        # N = batch_size * (max_len - 1)
        # C = vocab_size
        loss = criterion(
            predictions.reshape(-1, predictions.size(2)), targets.reshape(-1)
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()

        losses.update(loss.item(), images.size(0))

        if i % 100 == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}] Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def validate(val_loader, model, criterion, device, tokenizer):
    """
    Validates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    levenshtein_distances = AverageMeter()

    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass without teacher forcing
            # We pass captions to define the sequence length, but set ratio=0 to use model predictions
            outputs = model(images, captions, teacher_forcing_ratio=0.0)

            # Loss calculation
            targets = captions[:, 1:]
            predictions = outputs[:, 1:, :]
            loss = criterion(
                predictions.reshape(-1, predictions.size(2)), targets.reshape(-1)
            )

            losses.update(loss.item(), images.size(0))

            # Metric calculation
            # Greedy decoding from logits
            # shape: (batch, seq_len, vocab) -> (batch, seq_len)
            predicted_indices = torch.argmax(predictions, dim=2)

            # Convert to text
            pred_texts = []
            target_texts = []

            for idx in range(images.size(0)):
                # Convert predicted sequence
                p_seq = predicted_indices[idx].cpu().tolist()
                p_text = tokenizer.sequence_to_text(p_seq)
                pred_texts.append(p_text)

                # Convert target sequence
                t_seq = targets[idx].cpu().tolist()
                t_text = tokenizer.sequence_to_text(t_seq)
                target_texts.append(t_text)

            # Compute Levenshtein distance
            batch_lev = compute_levenshtein(pred_texts, target_texts)
            levenshtein_distances.update(batch_lev, images.size(0))

    return losses.avg, levenshtein_distances.avg


def fit(debug=False, load_cached_data=True):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Tokenizer
    tokenizer = Tokenizer(load_cached_data=load_cached_data)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # 2. Data Loaders
    train_loader = get_train_dataloader(
        tokenizer,
        batch_size=Config.BATCH_SIZE,
        debug=debug,
        load_cached_data=load_cached_data,
    )
    val_loader = get_val_dataloader(
        tokenizer,
        batch_size=Config.BATCH_SIZE,
        debug=debug,
        load_cached_data=load_cached_data,
    )

    # 3. Model
    model = ShowAttendTell(vocab_size=vocab_size).to(device)

    # 4. Loss and Optimizer
    # Ignore padding index in loss calculation
    pad_idx = tokenizer.stoi[Config.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    best_lev_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, tokenizer
        )

        # Validate
        val_loss, val_lev = validate(val_loader, model, criterion, device, tokenizer)

        duration = time.time() - start_time

        print(f"Epoch {epoch} completed in {duration:.0f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Levenshtein: {val_lev}")

        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_lev_score = val_lev
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "vocab_size": vocab_size,
                },
                filename=Config.MODEL_SAVE_PATH,
            )
            print(f"New best model saved with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Val Loss: {best_val_loss}, Best Levenshtein: {best_lev_score}"
    )
