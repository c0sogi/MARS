import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import AverageMeter, compute_levenshtein
from library.data import get_dataloaders, set_seed
from library.model import AttributeConditionedModel


def train_one_epoch(
    train_loader, model, criterion_seq, criterion_attr, optimizer, device, epoch
):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    seq_losses = AverageMeter()
    attr_losses = AverageMeter()

    # Iterate over training data
    # Using enumerate effectively acts as a progress monitor if we printed every N batches
    for i, (images, target_seqs, target_attrs) in enumerate(train_loader):
        images = images.to(device)
        target_seqs = target_seqs.to(device)
        target_attrs = target_attrs.to(device)

        batch_size = images.size(0)

        # Forward pass
        # Returns seq_logits (B, Seq_Len-1, Vocab) and pred_attributes (B, Attr_Dim)
        seq_logits, pred_attrs = model(images, target_seqs)

        # Calculate Sequence Loss
        # Targets for loss are the sequences shifted by one (excluding SOS)
        # seq_logits corresponds to predictions for positions 1 to T
        loss_seq = criterion_seq(
            seq_logits.reshape(-1, seq_logits.size(-1)), target_seqs[:, 1:].reshape(-1)
        )

        # Calculate Attribute Loss
        loss_attr = criterion_attr(pred_attrs, target_attrs)

        # Combined Loss
        loss = loss_seq + Config.LAMBDA_ATTR_LOSS * loss_attr

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), batch_size)
        seq_losses.update(loss_seq.item(), batch_size)
        attr_losses.update(loss_attr.item(), batch_size)

    return losses.avg, seq_losses.avg, attr_losses.avg


def validate(val_loader, model, criterion_seq, criterion_attr, tokenizer, device):
    """
    Validates the model. Computes Loss and Levenshtein distance.
    """
    model.eval()

    losses = AverageMeter()
    levenshtein_distances = AverageMeter()

    with torch.no_grad():
        for i, (images, target_seqs, target_attrs) in enumerate(val_loader):
            images = images.to(device)
            target_seqs = target_seqs.to(device)
            target_attrs = target_attrs.to(device)
            batch_size = images.size(0)

            # --- 1. Calculate Validation Loss (Teacher Forcing) ---
            seq_logits, pred_attrs = model(images, target_seqs)

            loss_seq = criterion_seq(
                seq_logits.reshape(-1, seq_logits.size(-1)),
                target_seqs[:, 1:].reshape(-1),
            )
            loss_attr = criterion_attr(pred_attrs, target_attrs)
            loss = loss_seq + Config.LAMBDA_ATTR_LOSS * loss_attr

            losses.update(loss.item(), batch_size)

            # --- 2. Calculate Metric (Greedy Decoding Inference) ---
            # We predict sequences from scratch to measure actual generation performance
            pred_seqs = model.predict(images, device=device)

            pred_seqs_np = pred_seqs.cpu().numpy()
            target_seqs_np = target_seqs.cpu().numpy()

            for j in range(batch_size):
                # Decode texts
                pred_text = tokenizer.sequence_to_text(pred_seqs_np[j])
                target_text = tokenizer.sequence_to_text(target_seqs_np[j])

                # Compute distance
                dist = compute_levenshtein(pred_text, target_text)
                levenshtein_distances.update(dist)

    return losses.avg, levenshtein_distances.avg


def generate_submission(test_loader, model, tokenizer, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Predict
            pred_seqs = model.predict(images, device=device)
            pred_seqs_np = pred_seqs.cpu().numpy()

            # Decode
            for j in range(len(ids)):
                text = tokenizer.sequence_to_text(pred_seqs_np[j])
                predictions.append(text)
                image_ids.append(ids[j])

    # Create DataFrame
    df_sub = pd.DataFrame({"image_id": image_ids, "InChI": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main(debug=False):
    # Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    # Set debug_size to a small number (e.g., 1000) to test pipeline quickly if needed
    debug_size = 2000 if debug else None
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        load_cached_data=True, debug_size=debug_size
    )

    # Model
    model = AttributeConditionedModel().to(device)

    # Optimization
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    # Losses
    # Sequence loss: CrossEntropy, ignoring padding
    criterion_seq = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    # Attribute loss: MSE
    criterion_attr = nn.MSELoss()

    # Training Loop
    best_levenshtein = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss, train_seq_loss, train_attr_loss = train_one_epoch(
            train_loader, model, criterion_seq, criterion_attr, optimizer, device, epoch
        )

        # Validate
        val_loss, val_levenshtein = validate(
            val_loader, model, criterion_seq, criterion_attr, tokenizer, device
        )

        # Scheduler Step
        scheduler.step(val_levenshtein)

        elapsed = time.time() - start_time

        # Logging
        print(f"Time: {elapsed:.2f}s")
        print(
            f"Train Loss: {train_loss:.6f} (Seq: {train_seq_loss:.6f}, Attr: {train_attr_loss:.6f})"
        )
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Levenshtein: {val_levenshtein}")  # Full precision

        # Checkpointing & Early Stopping
        if val_levenshtein < best_levenshtein:
            print(
                f"Validation improved ({best_levenshtein} -> {val_levenshtein}). Saving model..."
            )
            best_levenshtein = val_levenshtein
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_levenshtein": best_levenshtein,
            },
            Config.CHECKPOINT_PATH,
        )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Generate Submission
    print("\nTraining finished. Loading best model for submission...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found, using current model state.")

    generate_submission(test_loader, model, tokenizer, device)
