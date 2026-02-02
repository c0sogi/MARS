import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from tqdm import tqdm
import sys

from library.config import Config
from library.utils import set_seed, save_checkpoint, print_metrics
from library.dataset import get_tagger_dataloaders, get_seq2seq_dataloaders
from library.models import BiLSTMTagger, Seq2SeqNormalizer

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def calculate_class_weights(df, vocab_classes, device):
    """
    Calculates square-root smoothed class weights.
    Weight = sqrt(Total / Count)
    """
    # Flatten all classes from the dataframe
    all_classes = [c for seq in df["class"] for c in seq]
    counter = collections.Counter(all_classes)

    total_count = len(all_classes)
    num_classes = len(vocab_classes)

    weights = torch.ones(num_classes, device=device)

    for cls_name, count in counter.items():
        if cls_name in vocab_classes.stoi:
            idx = vocab_classes.stoi[cls_name]
            # Square-root smoothing
            w = (total_count / count) ** Config.SMOOTHING_FACTOR
            weights[idx] = w

    # Normalize weights to average to 1 (optional but helps stability)
    weights = weights / weights.mean()

    return weights


def train_step_tagger(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        word_ids, char_ids, class_ids = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass
        # lengths is not strictly needed if we rely on padding mask in loss,
        # but for packed_sequence in LSTM it's good.
        # Here we simplify and pass padded directly as model handles it.
        logits = model(word_ids, char_ids)

        # Reshape for loss: (batch * seq, num_classes) vs (batch * seq)
        loss = criterion(logits.view(-1, logits.shape[-1]), class_ids.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy calculation (ignoring padding -1)
        preds = torch.argmax(logits, dim=-1)
        mask = class_ids != -1
        correct = (preds == class_ids) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def val_step_tagger(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            word_ids, char_ids, class_ids = [b.to(device) for b in batch]

            logits = model(word_ids, char_ids)
            loss = criterion(logits.view(-1, logits.shape[-1]), class_ids.view(-1))

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = class_ids != -1
            correct = (preds == class_ids) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def train_step_seq2seq(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        src, tgt = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass with teacher forcing
        # Output: (batch, tgt_len, vocab_size)
        outputs = model(src, tgt, teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO)

        # Loss calculation
        # outputs[:, 1:] corresponds to predictions for tgt[:, 1:]
        # We ignore index 0 (<sos>) in outputs (it's 0 initialized) and tgt
        output_logits = outputs[:, 1:].reshape(-1, outputs.shape[-1])
        target_labels = tgt[:, 1:].reshape(-1)

        loss = criterion(output_logits, target_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_step_seq2seq(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            src, tgt = [b.to(device) for b in batch]

            # Turn off teacher forcing for validation loss check?
            # Standard practice for loss calculation often uses TF=0 or TF=1.
            # Using TF=0 (autoregressive) is a better proxy for inference performance,
            # but TF=1 (forcing) is standard for validation loss in many frameworks.
            # Given the model structure, we use TF=0 to see real generation quality proxy.
            outputs = model(src, tgt, teacher_forcing_ratio=0.0)

            output_logits = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            target_labels = tgt[:, 1:].reshape(-1)

            loss = criterion(output_logits, target_labels)
            total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================================
# MAIN TRAINING ROUTINES
# =========================================================================


def train_tagger(load_cached_data=True):
    set_seed()
    print("Initializing Tagger Training...")

    # 1. Load Data
    train_loader, val_loader, vocab_tokens, vocab_chars, vocab_classes = (
        get_tagger_dataloaders(load_cached_data)
    )

    # 2. Model Setup
    model = BiLSTMTagger(
        token_vocab_size=len(vocab_tokens),
        char_vocab_size=len(vocab_chars),
        num_classes=len(vocab_classes),
    ).to(Config.DEVICE)

    # 3. Loss Setup (Class Balancing)
    if Config.USE_CLASS_WEIGHTS:
        print("Calculating class weights...")
        # Access the underlying dataframe from the dataset
        weights = calculate_class_weights(
            train_loader.dataset.df, vocab_classes, Config.DEVICE
        )
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # 4. Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.5
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_step_tagger(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_acc = val_step_tagger(model, val_loader, criterion, Config.DEVICE)

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print_metrics(epoch, metrics)

        scheduler.step(val_loss)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, metrics, Config.TAGGER_MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved to {Config.TAGGER_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break


def train_seq2seq(load_cached_data=True):
    set_seed()
    print("Initializing Seq2Seq Fallback Training...")

    # 1. Load Data
    train_loader, val_loader, vocab_chars = get_seq2seq_dataloaders(load_cached_data)

    # 2. Model Setup
    model = Seq2SeqNormalizer(char_vocab_size=len(vocab_chars)).to(Config.DEVICE)

    # 3. Loss Setup
    # Ignore padding index (0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_step_seq2seq(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss = val_step_seq2seq(model, val_loader, criterion, Config.DEVICE)

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        print_metrics(epoch, metrics)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, metrics, Config.SEQ2SEQ_MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved to {Config.SEQ2SEQ_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break
