import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from typing import Optional, List, Dict

from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    TAGGER_LEARNING_RATE,
    TAGGER_EPOCHS,
    TAGGER_PATIENCE,
    TAGGER_USE_SQRT_CLASS_WEIGHTS,
    SEQ2SEQ_LEARNING_RATE,
    SEQ2SEQ_EPOCHS,
    SEQ2SEQ_PATIENCE,
    SEQ2SEQ_TEACHER_FORCING_RATIO,
)
from library.utils import set_seed


def compute_class_weights(
    train_loader: torch.utils.data.DataLoader, num_classes: int
) -> torch.Tensor:
    """
    Computes class weights based on the frequency of classes in the training data.
    Weight = sqrt(Total_Count / Class_Count).
    """
    print("Computing class weights from training data distribution...")

    # Access the raw data from the dataset
    # The dataset is TaggerDataset, which has .data attribute (List[Dict])
    # Each item has 'class_ids' which is a list of ints
    dataset_data = train_loader.dataset.data

    all_class_ids = []
    for item in dataset_data:
        if "class_ids" in item:
            all_class_ids.extend(item["class_ids"])

    if not all_class_ids:
        print("Warning: No class IDs found in training data. Using uniform weights.")
        return torch.ones(num_classes).to(DEVICE)

    counter = Counter(all_class_ids)
    total_count = len(all_class_ids)

    weights = torch.zeros(num_classes)

    # Calculate weights
    for cls_idx in range(num_classes):
        count = counter.get(cls_idx, 0)
        if count > 0:
            # Square-Root Smoothing
            weights[cls_idx] = np.sqrt(total_count / count)
        else:
            # For unseen classes, assign a weight of 1.0 or similar to avoid 0
            # Ideally unseen classes shouldn't happen in train, but just in case
            weights[cls_idx] = 1.0

    return weights.to(DEVICE)


def train_tagger(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    vocab_classes_len: int,
) -> None:
    """
    Trains the RegexBiLSTMTagger model.
    """
    print("Starting Tagger Training...")

    model = model.to(DEVICE)

    # 1. Loss Function
    class_weights = None
    if TAGGER_USE_SQRT_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_loader, vocab_classes_len)
        print(f"Using computed class weights.")

    # ignore_index=-100 matches the padding value in tagger_collate_fn
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # 2. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=TAGGER_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "tagger_best_model.pth")

    for epoch in range(TAGGER_EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(DEVICE)
            char_ids = batch["char_ids"].to(DEVICE)
            regex_features = batch["regex_features"].to(DEVICE)
            class_ids = batch["class_ids"].to(DEVICE)

            optimizer.zero_grad()

            # Forward
            logits = model(word_ids, char_ids, regex_features)
            # logits: (batch, seq, num_classes)
            # class_ids: (batch, seq)

            # Flatten for loss
            loss = criterion(logits.view(-1, vocab_classes_len), class_ids.view(-1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy calculation (masking padding)
            with torch.no_grad():
                preds = torch.argmax(logits, dim=2)
                mask = class_ids != -100
                correct = (preds == class_ids) & mask
                train_correct += correct.sum().item()
                train_total += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                word_ids = batch["word_ids"].to(DEVICE)
                char_ids = batch["char_ids"].to(DEVICE)
                regex_features = batch["regex_features"].to(DEVICE)
                class_ids = batch["class_ids"].to(DEVICE)

                logits = model(word_ids, char_ids, regex_features)
                loss = criterion(logits.view(-1, vocab_classes_len), class_ids.view(-1))

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=2)
                mask = class_ids != -100
                correct = (preds == class_ids) & mask
                val_correct += correct.sum().item()
                val_total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total if val_total > 0 else 0.0

        # --- LOGGING ---
        print(f"Epoch {epoch + 1}/{TAGGER_EPOCHS}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Train Acc: {avg_train_acc}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val Acc: {avg_val_acc}")

        # --- SCHEDULER & EARLY STOPPING ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Saved model to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {TAGGER_PATIENCE}")

        if patience_counter >= TAGGER_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Tagger training complete.")


def train_seq2seq(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    vocab_chars_len: int,
) -> None:
    """
    Trains the CharLSTMSeq2Seq fallback model.
    """
    print("Starting Seq2Seq Training...")

    model = model.to(DEVICE)

    # 1. Loss Function
    # ignore_index=0 assumes 0 is PAD_TOKEN index in vocab_chars
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 2. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=SEQ2SEQ_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "seq2seq_best_model.pth")

    for epoch in range(SEQ2SEQ_EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            src_char_ids = batch["src_char_ids"].to(DEVICE)
            trg_char_ids = batch["trg_char_ids"].to(DEVICE)
            class_ids = batch["class_ids"].to(DEVICE)

            optimizer.zero_grad()

            # Forward
            # Output: (batch, trg_len, vocab_size)
            output = model(
                src_char_ids,
                class_ids,
                trg_char_ids,
                teacher_forcing_ratio=SEQ2SEQ_TEACHER_FORCING_RATIO,
            )

            # Calculate Loss
            # trg_char_ids includes SOS at start and EOS at end.
            # Usually output[t] corresponds to prediction for trg[t].
            # However, standard seq2seq implementation:
            # Input to decoder at step t is trg[t], output is prediction for trg[t+1].
            # The model implementation loop:
            # Input t=0 is SOS. Prediction is for trg[1].
            # So we compare output[:, t, :] with trg[:, t+1] ?
            # Let's check model.py:
            # Loop t from 1 to trg_len. Input token is trg[:, 0] (SOS) initially.
            # outputs[:, t, :] = prediction.
            # So outputs[:, 1, :] predicts trg[:, 1].
            # outputs[:, 0, :] is 0 (initialized).
            # So we should exclude index 0 from outputs and compare with trg[:, 1:].

            output_dim = output.shape[-1]

            # Slice output to remove the 0th index (which is all zeros)
            output_valid = output[:, 1:, :].reshape(-1, output_dim)
            # Slice target to remove the 0th index (SOS token)
            trg_valid = trg_char_ids[:, 1:].reshape(-1)

            loss = criterion(output_valid, trg_valid)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                src_char_ids = batch["src_char_ids"].to(DEVICE)
                trg_char_ids = batch["trg_char_ids"].to(DEVICE)
                class_ids = batch["class_ids"].to(DEVICE)

                # Turn off teacher forcing for validation loss calculation?
                # Usually for loss calculation we still provide target inputs but with 0 forcing ratio
                # or just use same logic. The model forward accepts the ratio.
                # If ratio is 0, it uses its own predictions.
                # Standard practice for validation LOSS is often using teacher forcing to measure perplexity,
                # but for generation quality, we use 0.
                # Let's use 0 to see how well it generates on its own.
                output = model(
                    src_char_ids, class_ids, trg_char_ids, teacher_forcing_ratio=0.0
                )

                output_dim = output.shape[-1]
                output_valid = output[:, 1:, :].reshape(-1, output_dim)
                trg_valid = trg_char_ids[:, 1:].reshape(-1)

                loss = criterion(output_valid, trg_valid)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- LOGGING ---
        print(f"Epoch {epoch + 1}/{SEQ2SEQ_EPOCHS}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")

        # --- SCHEDULER & EARLY STOPPING ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Saved model to {checkpoint_path}")
        else:
            patience_counter += 1
            print(
                f"EarlyStopping counter: {patience_counter} out of {SEQ2SEQ_PATIENCE}"
            )

        if patience_counter >= SEQ2SEQ_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Seq2Seq training complete.")
