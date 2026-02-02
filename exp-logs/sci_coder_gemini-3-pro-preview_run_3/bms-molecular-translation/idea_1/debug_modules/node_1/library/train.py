import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    compute_levenshtein,
    seed_everything,
)
from library.dataset import get_dataloaders
from library.model import ShowAndTell


def train_one_epoch(
    train_loader, model, criterion, optimizer, device, epoch, print_freq=100
):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    start = time.time()

    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        # model(images, captions) returns logits for the sequence
        # The model expects captions including <SOS> and <EOS>
        # It internally slices to exclude the last token for input
        logits = model(images, captions)

        # Targets are the captions shifted by one (excluding <SOS>)
        targets = captions[:, 1:]

        # Reshape for CrossEntropyLoss
        # Logits: [B, seq_len-1, vocab_size] -> [B * (seq_len-1), vocab_size]
        # Targets: [B, seq_len-1] -> [B * (seq_len-1)]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        # Record loss
        losses.update(loss.item(), images.size(0))

        if (i + 1) % print_freq == 0:
            print(
                f"Epoch: [{epoch + 1}][{i + 1}/{len(train_loader)}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Time {time.time() - start:.2f}s"
            )

    return losses.avg


def validate(val_loader, model, criterion, device, tokenizer, print_freq=100):
    """
    Evaluates the model on the validation set.
    Computes Loss (Teacher Forcing) and Levenshtein Distance (Greedy Decoding).
    """
    model.eval()

    losses = AverageMeter()
    levenshtein_scores = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)

            # --- 1. Calculate Validation Loss (Teacher Forcing) ---
            logits = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            losses.update(loss.item(), images.size(0))

            # --- 2. Calculate Levenshtein Distance (Greedy Decoding) ---
            # Encode images
            h, c = model.encoder(images)

            # Initialize inputs with <SOS> token
            # Shape: [B, 1]
            start_token = torch.full(
                (images.size(0), 1),
                tokenizer.sos_token_id,
                dtype=torch.long,
                device=device,
            )
            inputs = start_token

            # Store predictions
            # We will collect indices and convert to text later
            preds_indices = []

            # Decoding loop
            # We generate up to Config.MAX_PRED_LEN
            for _ in range(Config.MAX_PRED_LEN):
                # Decoder forward step
                # inputs: [B, 1] -> embeddings -> lstm -> logits [B, 1, V]
                output_logits, (h, c) = model.decoder(inputs, h, c)

                # Greedy selection: argmax
                predicted_token = output_logits.argmax(dim=2)  # [B, 1]

                preds_indices.append(predicted_token)

                # Next input is current prediction
                inputs = predicted_token

            # Concatenate predictions along sequence dimension: [B, L]
            preds_indices = torch.cat(preds_indices, dim=1)

            # Convert to strings
            pred_strs = [tokenizer.sequence_to_text(seq) for seq in preds_indices]
            target_strs = [tokenizer.sequence_to_text(seq) for seq in captions]

            # Compute metric
            batch_score = compute_levenshtein(pred_strs, target_strs)
            levenshtein_scores.update(batch_score, images.size(0))

            if (i + 1) % print_freq == 0:
                print(
                    f"Val: [{i + 1}/{len(val_loader)}] "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"LevDist {levenshtein_scores.val:.4f} ({levenshtein_scores.avg:.4f}) "
                    f"Time {time.time() - start:.2f}s"
                )

    return losses.avg, levenshtein_scores.avg


def train_model(debug=False, epochs=Config.NUM_EPOCHS):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing DataLoaders (Debug={debug})...")
    train_loader, val_loader, _, tokenizer = get_dataloaders(
        debug=debug, debug_size=Config.DEBUG_SIZE
    )

    print("Initializing Model...")
    model = ShowAndTell(vocab_size=len(tokenizer))
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    best_levenshtein = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_levenshtein = validate(
            val_loader, model, criterion, device, tokenizer
        )

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Levenshtein: {val_levenshtein:.6f}")

        # Scheduler Step
        scheduler.step(val_levenshtein)

        # Checkpointing & Early Stopping
        is_best = val_levenshtein < best_levenshtein
        if is_best:
            best_levenshtein = val_levenshtein
            patience_counter = 0
            print(f"New Best Model! Saving checkpoint...")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_levenshtein,
            },
            is_best,
        )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Levenshtein Distance: {best_levenshtein}")
    return model


def generate_submission(model=None):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Data
    _, _, test_loader, tokenizer = get_dataloaders(debug=False)

    # Load Model if not provided
    if model is None:
        print("Loading best model for submission...")
        model = ShowAndTell(vocab_size=len(tokenizer))
        model = model.to(device)
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")
        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Best model not found at {checkpoint_path}. Checking for regular checkpoint."
            )
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint.pth.tar")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    predictions = []
    image_ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for i, (images, batch_ids) in enumerate(test_loader):
            images = images.to(device)

            # Encode
            h, c = model.encoder(images)

            # Initialize inputs
            start_token = torch.full(
                (images.size(0), 1),
                tokenizer.sos_token_id,
                dtype=torch.long,
                device=device,
            )
            inputs = start_token

            batch_preds_indices = []

            # Greedy Decode
            for _ in range(Config.MAX_PRED_LEN):
                output_logits, (h, c) = model.decoder(inputs, h, c)
                predicted_token = output_logits.argmax(dim=2)
                batch_preds_indices.append(predicted_token)
                inputs = predicted_token

            batch_preds_indices = torch.cat(batch_preds_indices, dim=1)

            # Convert to text
            batch_pred_strs = [
                tokenizer.sequence_to_text(seq) for seq in batch_preds_indices
            ]

            predictions.extend(batch_pred_strs)
            image_ids.extend(batch_ids)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches.")

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"image_id": image_ids, "InChI": predictions})

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())
