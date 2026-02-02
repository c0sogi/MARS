import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import (
    AverageMeter,
    seed_everything,
    save_checkpoint,
    compute_levenshtein,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import DecoderOnlyTransformer
from library.tokenizer import Tokenizer


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, device, epoch, scaler
):
    """
    Trains the model for one epoch using Teacher Forcing.
    """
    model.train()
    losses = AverageMeter()
    start_time = time.time()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Teacher Forcing Inputs and Targets
        # Input: [SOS, A, B, C, ...]
        # Target: [A, B, C, EOS, ...]
        text_input_ids = labels[:, :-1]
        target_ids = labels[:, 1:]

        optimizer.zero_grad()

        with autocast(enabled=True):
            # Forward pass
            # Logits: (B, Seq_Len, Vocab_Size)
            logits = model(images, text_input_ids)

            # Reshape for loss calculation
            # logits: (B * Seq_Len, Vocab_Size)
            # targets: (B * Seq_Len)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), batch_size)

        if step % 100 == 0 or step == len(dataloader) - 1:
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(dataloader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} "
                f"Time: {time.time() - start_time:.2f}s"
            )

    return losses.avg


def validate(model, dataloader, tokenizer, criterion, device, sample_limit=1000):
    """
    Validates the model. Computes Loss on full set and Levenshtein on a subset.
    """
    model.eval()
    losses = AverageMeter()
    predictions = []
    targets = []

    # 1. Compute Validation Loss (Teacher Forcing)
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            text_input_ids = labels[:, :-1]
            target_ids = labels[:, 1:]

            with autocast(enabled=True):
                logits = model(images, text_input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )

            losses.update(loss.item(), batch_size)

            # 2. Collect samples for Levenshtein calculation (Subset)
            # We only generate for the first few batches to save time
            if len(targets) < sample_limit:
                # Generate predictions autoregressively
                # max_len includes SOS/EOS, so we give it enough room
                batch_preds = model.generate(
                    images, tokenizer, max_len=Config.MAX_TEXT_LEN
                )

                # Decode targets
                # labels contains SOS/EOS/PAD. sequence_to_text handles removal.
                batch_targets = [
                    tokenizer.sequence_to_text(seq, remove_special_tokens=True)
                    for seq in labels.cpu().numpy()
                ]

                predictions.extend(batch_preds)
                targets.extend(batch_targets)

    # Compute Levenshtein on the subset
    lev_score = compute_levenshtein(predictions, targets)

    print(f"Validation Loss: {losses.avg:.6f}")
    print(f"Validation Levenshtein (Subset {len(targets)}): {lev_score:.6f}")

    return losses.avg, lev_score


def predict_and_submit(model, dataloader, tokenizer, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("\nStarting prediction on test set...")
    model.eval()
    all_preds = []
    image_ids = []

    # Extract image IDs from the test metadata
    test_df = pd.read_csv(Config.TEST_METADATA)
    test_ids = test_df["image_id"].values

    start_time = time.time()

    with torch.no_grad():
        for step, images in enumerate(dataloader):
            images = images.to(device)

            # Generate predictions
            batch_preds = model.generate(images, tokenizer, max_len=Config.MAX_TEXT_LEN)
            all_preds.extend(batch_preds)

            if step % 50 == 0:
                print(
                    f"Predicted batch {step}/{len(dataloader)} - {time.time() - start_time:.0f}s elapsed"
                )

    # Verify lengths
    if len(all_preds) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(all_preds)}) does not match number of test IDs ({len(test_ids)})"
        )
        # Truncate or pad if necessary, though dataloader should be aligned
        min_len = min(len(all_preds), len(test_ids))
        all_preds = all_preds[:min_len]
        test_ids = test_ids[:min_len]

    # Create submission DataFrame
    submission = pd.DataFrame({"image_id": test_ids, "InChI": all_preds})

    # Save
    submission.to_csv(Config.PREDICTIONS_CSV, index=False)
    print(f"Submission saved to {Config.PREDICTIONS_CSV}")


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Main execution function for training and prediction.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(debug=debug)
    vocab_size = len(tokenizer)

    # 2. Model
    model = DecoderOnlyTransformer(vocab_size=vocab_size)
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total training steps
    num_training_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # 4. Loss & Scaler
    # Ignore padding index in loss
    pad_idx = tokenizer.stoi[Config.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scaler = GradScaler()

    # 5. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    # Check for existing checkpoint to resume
    start_epoch, loaded_best_score = load_checkpoint(
        model, optimizer=optimizer, scheduler=scheduler
    )
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
        best_loss = loaded_best_score

    for epoch in range(start_epoch, epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch, scaler
        )

        # Validate
        val_loss, val_lev = validate(model, val_loader, tokenizer, criterion, device)

        print(
            f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Lev: {val_lev:.6f}"
        )

        # Save Checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            print("New best model found!")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_loss,
            },
            is_best,
        )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Prediction
    print("\nTraining complete. Loading best model for prediction...")
    # Reload best model weights
    load_checkpoint(model, filename=Config.BEST_MODEL_PATH)

    predict_and_submit(model, test_loader, tokenizer, device)
