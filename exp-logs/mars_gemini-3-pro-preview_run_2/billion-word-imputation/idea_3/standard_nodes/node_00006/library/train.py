import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.vocabulary import WordVocabulary
from library.dataset import InfillingDataset, collate_fn
from library.model import DualHeadTransformer
from library.utils import set_seed, calculate_metrics, save_checkpoint


def train_model(
    debug=Config.DEBUG,
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.TRAIN_BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
):
    """
    Main function to train the Dual-Head Transformer model.

    Args:
        debug (bool): If True, runs on a small subset of data.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # 1. Setup
    # Override Config.DEBUG based on argument to ensure Dataset picks it up
    Config.DEBUG = debug
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Vocabulary
    # Initialize and build/load vocabulary
    vocab = WordVocabulary()
    # We always ensure the vocab is built from train data
    vocab.build_from_corpus(
        Config.TRAIN_DATA_PATH,
        vocab_size=Config.TARGET_VOCAB_SIZE,
        save_path=Config.TARGET_VOCAB_PATH,
        load_cached=True,
    )
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size}")

    # 3. Data Loading
    print("Initializing Datasets...")
    train_dataset = InfillingDataset(split="train", vocabulary=vocab)
    val_dataset = InfillingDataset(split="val", vocabulary=vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = DualHeadTransformer(vocab_size=vocab_size)
    model.to(device)

    # 5. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    # 6. Loss Functions
    # Location Loss: Binary Cross Entropy over the sequence
    criterion_loc = nn.BCEWithLogitsLoss()
    # Word Loss: Cross Entropy (Multi-class)
    criterion_word = nn.CrossEntropyLoss()

    # 7. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Training Loop...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # --- Train Phase ---
        model.train()
        total_train_loss = 0.0
        train_loc_acc_sum = 0.0
        train_word_acc_sum = 0.0
        num_train_batches = 0

        for batch in train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loc_labels = batch["loc_label"].to(device)
            word_labels = batch["word_label"].to(device)

            optimizer.zero_grad()

            # Forward pass
            loc_logits, word_logits = model(input_ids, attention_mask)

            # --- Loss Calculation ---
            # 1. Location Loss
            loss_loc = criterion_loc(loc_logits, loc_labels)

            # 2. Word Loss
            # We only calculate word loss at the ground truth location.
            # loc_labels is (Batch, Seq), one-hot-ish.
            # Get the index of the insertion point (argmax of label)
            target_loc_indices = torch.argmax(loc_labels, dim=1)  # (Batch,)

            # Gather word logits at the target indices
            # word_logits: (Batch, Seq, Vocab)
            # We want: (Batch, Vocab)
            batch_indices = torch.arange(input_ids.size(0), device=device)
            target_word_logits = word_logits[batch_indices, target_loc_indices, :]

            loss_word = criterion_word(target_word_logits, word_labels)

            # Total Loss
            loss = (Config.LAMBDA_LOC * loss_loc) + (Config.LAMBDA_WORD * loss_word)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            # Metrics tracking
            total_train_loss += loss.item()

            with torch.no_grad():
                metrics = calculate_metrics(
                    loc_logits, word_logits, loc_labels, word_labels
                )
                train_loc_acc_sum += metrics["loc_acc"]
                train_word_acc_sum += metrics["word_acc"]

            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches
        avg_train_loc_acc = train_loc_acc_sum / num_train_batches
        avg_train_word_acc = train_word_acc_sum / num_train_batches

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_loc_acc_sum = 0.0
        val_word_acc_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                loc_labels = batch["loc_label"].to(device)
                word_labels = batch["word_label"].to(device)

                loc_logits, word_logits = model(input_ids, attention_mask)

                # Loss
                loss_loc = criterion_loc(loc_logits, loc_labels)

                target_loc_indices = torch.argmax(loc_labels, dim=1)
                batch_indices = torch.arange(input_ids.size(0), device=device)
                target_word_logits = word_logits[batch_indices, target_loc_indices, :]

                loss_word = criterion_word(target_word_logits, word_labels)

                loss = (Config.LAMBDA_LOC * loss_loc) + (Config.LAMBDA_WORD * loss_word)

                total_val_loss += loss.item()

                # Metrics
                metrics = calculate_metrics(
                    loc_logits, word_logits, loc_labels, word_labels
                )
                val_loc_acc_sum += metrics["loc_acc"]
                val_word_acc_sum += metrics["word_acc"]

                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        avg_val_loc_acc = val_loc_acc_sum / num_val_batches
        avg_val_word_acc = val_word_acc_sum / num_val_batches

        epoch_time = time.time() - start_time

        # --- Logging ---
        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time}s")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Train Loc Acc: {avg_train_loc_acc}")
        print(f"Train Word Acc: {avg_train_word_acc}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val Loc Acc: {avg_val_loc_acc}")
        print(f"Val Word Acc: {avg_val_word_acc}")

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < (best_val_loss - Config.MIN_DELTA):
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Validation loss improved. Saving checkpoint...")
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss, Config.MODEL_SAVE_PATH
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("-" * 30)

    print("Training complete.")
