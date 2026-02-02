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
    compute_levenshtein,
    save_checkpoint,
    load_checkpoint,
)
from library.tokenizer import Tokenizer
from library.dataset import get_dataloaders
from library.model import GFCN


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over the data loader
    for i, (images, labels, label_lengths) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        batch_size = images.size(0)

        # Forward pass
        # Output shape: (B, T, C)
        logits = model(images)

        # Calculate log_softmax for CTC
        # CTC Loss expects log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=2)

        # Permute for CTC Loss: (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)

        # Calculate input lengths
        # The model downsamples width by factor of 4 (see VisualBackbone in model.py)
        # Input width is fixed due to padding in dataset (MAX_WIDTH=2560 -> T=640)
        T = log_probs.size(0)
        input_lengths = torch.full(
            size=(batch_size,), fill_value=T, dtype=torch.long
        ).to(device)

        # Calculate loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()

        losses.update(loss.item(), batch_size)

        if i % 100 == 0 and i > 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(loader)}] Loss {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def validate(model, loader, criterion, tokenizer, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    predictions = []
    ground_truths = []

    print("Validating...")
    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            batch_size = images.size(0)

            # Forward pass
            logits = model(images)

            # Loss calculation
            log_probs = nn.functional.log_softmax(logits, dim=2)
            log_probs = log_probs.permute(1, 0, 2)
            T = log_probs.size(0)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=T, dtype=torch.long
            ).to(device)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            losses.update(loss.item(), batch_size)

            # Decoding
            # logits is (B, T, C)
            decoded_preds = tokenizer.decode_greedy(logits)
            predictions.extend(decoded_preds)

            # Decode targets for metric calculation
            # labels is (B, max_len)
            for j in range(batch_size):
                length = label_lengths[j].item()
                # Get indices for this sample
                indices = labels[j, :length].cpu().numpy()
                # Convert indices to string
                target_str = "".join(
                    [tokenizer.idx_to_char.get(idx, "") for idx in indices]
                )
                ground_truths.append(target_str)

    # Compute Levenshtein distance
    distance = compute_levenshtein(predictions, ground_truths)

    return losses.avg, distance


def generate_submission(model, loader, tokenizer, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    print("Generating submission...")
    with torch.no_grad():
        for i, (images, image_ids) in enumerate(loader):
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Decode
            decoded_preds = tokenizer.decode_greedy(logits)

            for img_id, pred in zip(image_ids, decoded_preds):
                results.append({"image_id": img_id, "InChI": pred})

            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(loader)} batches")

    df_submission = pd.DataFrame(results)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    """
    Main training pipeline.
    """
    # Setup directories
    Config.create_directories()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Tokenizer (handles caching internally based on flag)
    tokenizer = Tokenizer(load_cached_data=load_cached_data)

    # Initialize Data Loaders (handles caching internally based on flag)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=load_cached_data
    )

    # Initialize Model
    # num_classes = vocabulary size (including blank)
    model = GFCN(num_classes=len(tokenizer)).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # Loss Function
    # blank=0 is consistent with Tokenizer initialization
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Training Loop
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, tokenizer, device)

        # Scheduler Step
        scheduler.step(val_metric)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch} Completed in {elapsed:.0f}s")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Val Levenshtein: {val_metric}")  # Full precision

        # Checkpointing
        is_best = val_metric < best_metric
        if is_best:
            best_metric = val_metric
            patience_counter = 0
            print(f"New best model found! Score: {best_metric}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
            },
            is_best,
        )

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Final Submission using the best model
    print("Loading best model for submission...")
    # Re-initialize model to ensure clean state (though load_state_dict handles it)
    model = GFCN(num_classes=len(tokenizer)).to(device)
    _, _ = load_checkpoint(model, filename=Config.BEST_MODEL_PATH)

    generate_submission(model, test_loader, tokenizer, device)
