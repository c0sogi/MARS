import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from collections import Counter
from library.config import Config
from library.utils import (
    set_seed,
    EarlyStopping,
    AverageMeter,
    log_training_metrics,
    get_device,
)
from library.data_loader import (
    get_tagger_loaders,
    get_seq2seq_loaders,
    get_data,
)
from library.models_tagger import BiLSTM_CRF
from library.models_seq2seq import CharTransformer


def calculate_class_weights(vocab_classes, train_grouped_df, device):
    """
    Calculates class weights based on training data frequency.
    Uses Square-Root Smoothing: Weight = (Total / Count)^0.5
    """
    if not Config.USE_CLASS_WEIGHTS:
        return None

    print("Calculating class weights...")
    # Flatten all classes from the grouped dataframe
    all_classes = [c for sublist in train_grouped_df["class"] for c in sublist]
    class_counts = Counter(all_classes)

    total_samples = len(all_classes)
    num_classes = len(vocab_classes)

    weights = torch.ones(num_classes, device=device)

    # Iterate over vocab to ensure index alignment
    for token, idx in vocab_classes.stoi.items():
        if token in [Config.PAD_TOKEN]:
            weights[idx] = 0.0  # Mask padding
            continue

        count = class_counts.get(token, 0)
        if count > 0:
            # Smoothing formula: (N / count) ^ power
            raw_weight = (total_samples / count) ** Config.CLASS_WEIGHT_SMOOTHING
            weights[idx] = raw_weight
        else:
            # Fallback for classes not in training data (rare)
            weights[idx] = 1.0

    # Normalize weights so they average to 1 (optional but good for stability)
    # We filter out 0 weights (padding) for normalization
    valid_weights = weights[weights > 0]
    if len(valid_weights) > 0:
        weights = weights / valid_weights.mean()

    print(f"Class Weights (Top 5): {weights[:5]}")
    return weights


def evaluate_tagger(model, data_loader, device):
    """
    Evaluates the Tagger model on a dataset.
    Returns: avg_loss, accuracy
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for batch in data_loader:
            token_ids = batch["token_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            mask = batch["mask"].to(device)

            # Compute Loss
            loss = model.loss(token_ids, char_ids, label_ids, mask)
            loss_meter.update(loss.item(), token_ids.size(0))

            # Compute Accuracy
            # Decode returns (Batch, Seq_Len)
            predictions = model.decode(token_ids, char_ids, mask)

            # Mask out padding for accuracy calculation
            active_mask = mask.view(-1) == 1
            flat_preds = predictions.view(-1)[active_mask]
            flat_labels = label_ids.view(-1)[active_mask]

            correct = (flat_preds == flat_labels).sum().item()
            total = active_mask.sum().item()

            if total > 0:
                acc_meter.update(correct / total, total)

    return loss_meter.avg, acc_meter.avg


def train_tagger(load_cached=True):
    """
    Trains the Bi-LSTM-CRF Tagger.
    """
    set_seed()
    device = get_device()
    print(f"Starting Tagger Training on {device}...")

    # 1. Load Data
    train_loader, val_loader, _, vocab_tokens, vocab_chars, vocab_classes = (
        get_tagger_loaders(load_cached=load_cached)
    )

    # Get raw data for weight calc (cached)
    _, _, _, train_grouped, _, _, _ = get_data(load_cached=load_cached)

    # 2. Calculate Weights
    class_weights = calculate_class_weights(vocab_classes, train_grouped, device)

    # 3. Initialize Model
    model = BiLSTM_CRF(
        vocab_size=len(vocab_tokens),
        char_vocab_size=len(vocab_chars),
        num_classes=len(vocab_classes),
        class_weights=class_weights,
    ).to(device)

    # 4. Optimization
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=Config.TAGGER_MODEL_PATH
    )

    # 5. Training Loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss_meter = AverageMeter()

        for i, batch in enumerate(train_loader):
            token_ids = batch["token_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            # Forward + Loss
            loss = model.loss(token_ids, char_ids, label_ids, mask)

            loss.backward()

            if Config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            optimizer.step()
            train_loss_meter.update(loss.item(), token_ids.size(0))

        # Validation
        val_loss, val_acc = evaluate_tagger(model, val_loader, device)

        # Logging
        elapsed = time.time() - start_time
        log_training_metrics(
            epoch, Config.NUM_EPOCHS, train_loss_meter.avg, val_loss, val_acc, elapsed
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping
        early_stopping(val_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Tagger training complete.")


def evaluate_seq2seq(model, data_loader, criterion, device):
    """
    Evaluates the Seq2Seq model.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for batch in data_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            class_id = batch["class_id"].to(device)

            # Prepare inputs/targets
            # Input to decoder: <SOS> ... char_n
            dec_input = tgt_ids[:, :-1]
            # Target output: char_1 ... <EOS>
            target = tgt_ids[:, 1:]

            logits = model(src_ids, dec_input, class_id)

            # Reshape for loss: (Batch * Seq, Vocab)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_meter.update(loss.item(), src_ids.size(0))

            # Accuracy (Character level, excluding pad)
            preds = torch.argmax(logits, dim=-1)
            mask = target != 0  # Assuming 0 is PAD
            correct = ((preds == target) & mask).sum().item()
            total = mask.sum().item()

            if total > 0:
                acc_meter.update(correct / total, total)

    return loss_meter.avg, acc_meter.avg


def train_seq2seq(load_cached=True):
    """
    Trains the Transformer Seq2Seq Fallback Model.
    """
    set_seed()
    device = get_device()
    print(f"Starting Seq2Seq Training on {device}...")

    # 1. Load Data
    train_loader, val_loader, vocab_chars, vocab_classes = get_seq2seq_loaders(
        load_cached=load_cached
    )

    # 2. Initialize Model
    model = CharTransformer(
        num_chars=len(vocab_chars), num_classes=len(vocab_classes)
    ).to(device)

    # 3. Optimization
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_chars.stoi[Config.PAD_TOKEN])
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=Config.SEQ2SEQ_MODEL_PATH
    )

    # 4. Training Loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss_meter = AverageMeter()

        for i, batch in enumerate(train_loader):
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            class_id = batch["class_id"].to(device)

            # Decoder Input: Remove last token (<EOS> or PAD)
            dec_input = tgt_ids[:, :-1]
            # Target: Remove first token (<SOS>)
            target = tgt_ids[:, 1:]

            optimizer.zero_grad()

            # Forward
            logits = model(src_ids, dec_input, class_id)

            # Loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            loss.backward()

            if Config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            optimizer.step()
            train_loss_meter.update(loss.item(), src_ids.size(0))

        # Validation
        val_loss, val_acc = evaluate_seq2seq(model, val_loader, criterion, device)

        # Logging
        elapsed = time.time() - start_time
        log_training_metrics(
            epoch, Config.NUM_EPOCHS, train_loss_meter.avg, val_loss, val_acc, elapsed
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping
        early_stopping(val_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Seq2Seq training complete.")
