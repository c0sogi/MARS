import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from collections import Counter

from library.config import Config
from library.utils import (
    set_seed,
    get_device,
    setup_logger,
    AverageMeter,
    EarlyStopping,
)
from library.data_processing import prepare_data, TaggerDataset
from library.models_tagger import QuadHybridBiLSTM


def calculate_class_weights(df, vocab_classes, device):
    """
    Calculates square-root smoothed class weights to handle imbalance.
    Weight = sqrt(Total_Count / Class_Count)
    """
    print("Calculating class weights...")
    # Flatten all classes in the dataframe
    all_classes = []
    for cls_list in df["class"]:
        all_classes.extend(cls_list)

    counter = Counter(all_classes)
    total_count = len(all_classes)
    num_classes = len(vocab_classes)

    weights = torch.ones(num_classes, device=device)

    for cls_name, count in counter.items():
        cls_id = vocab_classes.get_id(cls_name)
        if cls_id is not None:
            # Square-root smoothing
            w = np.sqrt(total_count / count)
            weights[cls_id] = w

    return weights


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, print_freq=100):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()

    for i, batch in enumerate(loader):
        # Move inputs to device
        word_ids = batch["word_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        bpe_ids = batch["bpe_ids"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(word_ids, char_ids, bpe_ids, features)  # (B, S, Num_Classes)

        # Reshape for CrossEntropyLoss: (B, C, S) vs (B, S)
        logits_permuted = logits.permute(0, 2, 1)

        # Compute Loss (reduction='none' so we can mask)
        raw_loss = criterion(logits_permuted, targets)  # (B, S)

        # Apply mask
        masked_loss = raw_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-9)

        # Backward
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config().CLIP_GRAD)

        optimizer.step()

        # Calculate Accuracy
        preds = torch.argmax(logits, dim=2)  # (B, S)
        correct = (preds == targets).float() * mask
        acc = correct.sum() / (mask.sum() + 1e-9)

        # Update meters
        losses.update(loss.item(), word_ids.size(0))
        accuracies.update(acc.item(), word_ids.size(0))

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch: [{epoch}][{i+1}/{len(loader)}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc {accuracies.val:.4f} ({accuracies.avg:.4f}) "
                f"Time {elapsed:.2f}s"
            )

    return losses.avg, accuracies.avg


def validate(model, loader, criterion, device):
    """
    Validates the model.
    """
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            bpe_ids = batch["bpe_ids"].to(device)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            logits = model(word_ids, char_ids, bpe_ids, features)
            logits_permuted = logits.permute(0, 2, 1)

            raw_loss = criterion(logits_permuted, targets)
            masked_loss = raw_loss * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-9)

            preds = torch.argmax(logits, dim=2)
            correct = (preds == targets).float() * mask
            acc = correct.sum() / (mask.sum() + 1e-9)

            losses.update(loss.item(), word_ids.size(0))
            accuracies.update(acc.item(), word_ids.size(0))

    return losses.avg, accuracies.avg


def train_tagger_model(load_cached_data=True):
    """
    Main function to train the Tagger model.
    """
    config = Config()
    set_seed(config.SEED)
    device = get_device()
    logger = setup_logger(
        "tagger_train", os.path.join(config.WORKING_DIR, "train_tagger.log")
    )

    logger.info("Starting Tagger Training Pipeline")

    # 1. Prepare Data
    data_artifacts = prepare_data(load_cached_data=load_cached_data)

    vocab_words = data_artifacts["vocab_words"]
    vocab_chars = data_artifacts["vocab_chars"]
    vocab_classes = data_artifacts["vocab_classes"]
    bpe_tokenizer = data_artifacts["bpe_tokenizer"]
    train_grouped = data_artifacts["train_grouped"]
    val_grouped = data_artifacts["val_grouped"]

    logger.info(
        f"Vocab Sizes - Words: {len(vocab_words)}, Chars: {len(vocab_chars)}, Classes: {len(vocab_classes)}"
    )
    logger.info(
        f"Train Sentences: {len(train_grouped)}, Val Sentences: {len(val_grouped)}"
    )

    # 2. Datasets & DataLoaders
    train_dataset = TaggerDataset(
        train_grouped, vocab_words, vocab_chars, vocab_classes, bpe_tokenizer
    )
    val_dataset = TaggerDataset(
        val_grouped, vocab_words, vocab_chars, vocab_classes, bpe_tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = QuadHybridBiLSTM(
        num_classes=len(vocab_classes),
        vocab_words=vocab_words,
        vocab_chars=vocab_chars,
        vocab_bpe_size=config.BPE_VOCAB_SIZE,
    )
    model.to(device)

    # 4. Loss & Optimizer
    # Calculate class weights
    class_weights = calculate_class_weights(train_grouped, vocab_classes, device)

    # Use reduction='none' to handle masking manually
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        verbose=True,
        path=config.TAGGER_MODEL_PATH,
        trace_func=logger.info,
    )

    # 5. Training Loop
    logger.info("Starting training loop...")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Log full precision
        logger.info(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        # Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")
