import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    AverageMeter,
    LevenshteinMetric,
    save_checkpoint,
    load_checkpoint,
)
from library.tokenizer import Tokenizer
from library.model import Seq2Seq
from library.dataset import get_loaders


def train_fn(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    scheduler=None,
    epoch=0,
    config=None,
):
    """
    Executes one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    # Determine teacher forcing ratio
    tf_ratio = config.teacher_forcing_ratio if config else 0.5

    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass with teacher forcing
        # Output shape: [batch_size, seq_len, vocab_size]
        output = model(images, text=labels, teacher_forcing_ratio=tf_ratio)

        # Calculate loss
        # We ignore the first token (SOS) in both output and target for loss calculation
        # output[:, 1:] aligns with labels[:, 1:]
        output_flat = output[:, 1:].reshape(-1, output.shape[-1])
        labels_flat = labels[:, 1:].reshape(-1)

        loss = criterion(output_flat, labels_flat)

        loss.backward()

        # Gradient clipping
        if config and config.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.update(loss.item(), batch_size)

    return losses.avg


def eval_fn(val_loader, model, criterion, device, tokenizer):
    """
    Evaluates the model on the validation set.
    Computes Loss and Levenshtein Distance.
    """
    model.eval()
    losses = AverageMeter()
    lev_metric = LevenshteinMetric()

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # 1. Validation Loss
            # For validation loss, we typically turn off teacher forcing or set it to 0
            # to measure the model's autoregressive performance, or use the same as train.
            # Here we use 0.0 to strictly evaluate generation.
            output = model(images, text=labels, teacher_forcing_ratio=0.0)

            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            labels_flat = labels[:, 1:].reshape(-1)

            loss = criterion(output_flat, labels_flat)
            losses.update(loss.item(), batch_size)

            # 2. Levenshtein Score
            # Use the model's predict method for greedy decoding
            preds = model.predict(images)

            # Decode sequences to strings
            decoded_preds = [tokenizer.sequence_to_text(p) for p in preds]
            decoded_labels = [tokenizer.sequence_to_text(l) for l in labels]

            lev_metric.update(decoded_preds, decoded_labels)

    return losses.avg, lev_metric.compute()


def fit(config=None):
    """
    Main training loop with early stopping.
    """
    if config is None:
        config = Config()

    print(f"Starting training on device: {config.device}")

    # Initialize Tokenizer
    tokenizer = Tokenizer(config)
    tokenizer.load_or_build_vocab(load_cached_data=True)

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_loaders(config, tokenizer)

    # Initialize Model
    model = Seq2Seq(config, vocab_size=len(tokenizer)).to(config.device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Loss Function (Ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    best_score = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            config.device,
            epoch=epoch,
            config=config,
        )

        # Evaluate
        val_loss, val_score = eval_fn(
            val_loader, model, criterion, config.device, tokenizer
        )

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{config.epochs} | Time: {elapsed:.0f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val Score:  {val_score}")

        # Checkpoint and Early Stopping
        is_best = val_score < best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
            print("  New best score! Saving model...")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
            },
            is_best,
            config.checkpoint_path,
        )

        if patience_counter >= config.patience:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    return best_score


def inference(config=None):
    """
    Generates submission file using the best trained model.
    """
    if config is None:
        config = Config()

    print("Starting inference on test set...")

    # Ensure tokenizer is loaded
    tokenizer = Tokenizer(config)
    tokenizer.load_or_build_vocab(load_cached_data=True)

    # Get Test Loader
    _, _, test_loader = get_loaders(config, tokenizer)

    # Initialize Model
    model = Seq2Seq(config, vocab_size=len(tokenizer)).to(config.device)

    # Load Best Weights
    if os.path.exists(config.model_save_path):
        load_checkpoint(config.model_save_path, model, device=config.device)
    elif os.path.exists(config.checkpoint_path):
        print("Best model not found, loading last checkpoint.")
        load_checkpoint(config.checkpoint_path, model, device=config.device)
    else:
        print("Warning: No checkpoint found. Using random weights.")

    model.eval()

    pred_inchis = []

    # Generate predictions
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(config.device)
            preds = model.predict(images)

            decoded = [tokenizer.sequence_to_text(p) for p in preds]
            pred_inchis.extend(decoded)

    # Load test metadata to get image IDs in correct order
    test_df = pd.read_csv(config.test_metadata_path)

    # Handle debug subsetting if necessary
    if config.debug and config.subset_size:
        test_df = test_df.iloc[: config.subset_size]

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"image_id": test_df["image_id"].values, "InChI": pred_inchis}
    )

    # Save submission
    submission.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def run():
    """
    Wrapper to run the full pipeline.
    """
    config = Config()
    config.print_config()

    # Train
    fit(config)

    # Inference
    inference(config)
