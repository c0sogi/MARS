import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from library.config import Config
from library.utils import (
    set_seed,
    get_device,
    setup_logger,
    AverageMeter,
    EarlyStopping,
)
from library.data_processing import prepare_data, Seq2SeqDataset
from library.models_seq2seq import CharTransformer


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, print_freq=100):
    """
    Trains the Seq2Seq model for one epoch using Teacher Forcing.
    """
    model.train()
    losses = AverageMeter()
    start_time = time.time()

    for i, batch in enumerate(loader):
        # Move inputs to device
        src_ids = batch["src_ids"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        class_id = batch["class_id"].to(device)

        # Forward pass
        optimizer.zero_grad()

        # Transformer forward (Teacher Forcing via tgt_in)
        logits = model(src_ids, tgt_in, class_id)  # (B, T, V)

        # Reshape for Loss
        # Logits: (Batch * Tgt_Len, Vocab_Size)
        # Targets: (Batch * Tgt_Len)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = tgt_out.reshape(-1)

        loss = criterion(logits_flat, targets_flat)

        # Backward
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config().CLIP_GRAD)

        optimizer.step()

        # Update meters
        losses.update(loss.item(), src_ids.size(0))

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch: [{epoch}][{i+1}/{len(loader)}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Time {elapsed:.2f}s"
            )

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Validates the Seq2Seq model.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            class_id = batch["class_id"].to(device)

            logits = model(src_ids, tgt_in, class_id)

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_out.reshape(-1)

            loss = criterion(logits_flat, targets_flat)

            losses.update(loss.item(), src_ids.size(0))

    return losses.avg


def train_seq2seq_model(load_cached_data=True):
    """
    Main function to train the Transformer Seq2Seq Fallback model.
    """
    config = Config()
    set_seed(config.SEED)
    device = get_device()
    logger = setup_logger(
        "seq2seq_train", os.path.join(config.WORKING_DIR, "train_seq2seq.log")
    )

    logger.info("Starting Seq2Seq Training Pipeline")

    # 1. Prepare Data
    # We rely on prepare_data to filter the dataset for us (seq2seq_train/val)
    data_artifacts = prepare_data(load_cached_data=load_cached_data)

    vocab_chars = data_artifacts["vocab_chars"]
    vocab_classes = data_artifacts["vocab_classes"]
    df_train = data_artifacts["seq2seq_train"]
    df_val = data_artifacts["seq2seq_val"]

    logger.info(
        f"Vocab Sizes - Chars: {len(vocab_chars)}, Classes: {len(vocab_classes)}"
    )
    logger.info(
        f"Train Samples (Changed Tokens): {len(df_train)}, Val Samples: {len(df_val)}"
    )

    # 2. Datasets & DataLoaders
    train_dataset = Seq2SeqDataset(df_train, vocab_chars, vocab_classes)
    val_dataset = Seq2SeqDataset(df_val, vocab_chars, vocab_classes)

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
    model = CharTransformer(
        vocab_chars_size=len(vocab_chars), vocab_classes_size=len(vocab_classes)
    )
    model.to(device)

    # 4. Loss & Optimizer
    # Ignore padding index in loss calculation
    pad_id = vocab_chars.get_id("<PAD>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        verbose=True,
        path=config.SEQ2SEQ_MODEL_PATH,
        trace_func=logger.info,
    )

    # 5. Training Loop
    logger.info("Starting training loop...")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Log full precision
        logger.info(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss}, "
            f"Val Loss: {val_loss}"
        )

        # Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info("Seq2Seq Training complete.")
