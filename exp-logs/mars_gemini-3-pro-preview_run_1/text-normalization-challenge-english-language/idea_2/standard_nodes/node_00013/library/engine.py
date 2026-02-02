import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from library.config import Config
from library.utils import set_seed, get_device
from library.model import LSTMTagger


def calculate_class_weights(dataset, vocab_classes, device):
    """
    Calculates class weights inversely proportional to their frequency in the training data.
    """
    # Extract all class indices from the dataset
    # We iterate over the dataset to get class counts.
    # Since the dataset is grouped by sentence, we need to flatten the class lists.
    # Accessing the underlying dataframe is faster than iterating via __getitem__

    print("Calculating class weights...")
    all_classes = []

    # The dataset object has a .df attribute as per library/data_loader.py
    # We use the 'class' column.
    if hasattr(dataset, "df"):
        # Flatten the list of classes in each row
        raw_classes_lists = dataset.df["class"].tolist()
        # Flatten list of lists
        all_raw_classes = [c for sublist in raw_classes_lists for c in sublist]

        # Convert to indices
        # Note: This might be slow if the dataset is huge, but it's done once.
        # A faster way is to count strings first, then map to indices.
        class_counts = Counter(all_raw_classes)
    else:
        # Fallback if df is not accessible directly (though it should be)
        class_counts = Counter()
        for i in range(len(dataset)):
            item = dataset[i]
            # item['class_ids'] includes padding, we should ignore padding in count if possible
            # but usually padding is handled by ignore_index in loss.
            # Here we just want rough statistics.
            c_ids = item["class_ids"].tolist()
            # Map back to tokens to count? No, we need counts per ID.
            # This path is slow, assuming df path works.
            pass

    # Initialize weights array
    num_classes = len(vocab_classes)
    weights = torch.ones(num_classes)

    # Total valid tokens
    total_count = sum(class_counts.values())

    for class_token, count in class_counts.items():
        class_id = vocab_classes.stoi.get(class_token)
        if class_id is not None:
            # Formula: total / (num_classes * count)
            # Or simply inverse frequency: 1 / count (normalized later)
            # We use a smoothed inverse frequency
            weights[class_id] = total_count / (len(class_counts) * count)

    # Handle special tokens
    pad_id = vocab_classes.stoi.get(Config.PAD_TOKEN, 0)
    unk_id = vocab_classes.stoi.get(Config.UNK_TOKEN, 0)

    # We usually ignore PAD in loss, but set weight to 0 just in case
    weights[pad_id] = 0.0

    return weights.to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_tokens = 0
    total_tokens = 0

    pad_class_id = dataloader.dataset.pad_class_id

    for batch in dataloader:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        class_ids = batch["class_ids"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)  # [batch, seq_len, num_classes]

        # Reshape for loss calculation
        # Flatten batch and sequence dimensions
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = class_ids.view(-1)

        # Calculate loss
        loss = criterion(logits_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for Transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        running_loss += loss.item()

        # Accuracy calculation
        predictions = torch.argmax(logits, dim=-1)

        # Mask out padding for accuracy
        mask = class_ids != pad_class_id
        correct = (predictions == class_ids) & mask

        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluation loop.
    """
    model.eval()
    running_loss = 0.0
    correct_tokens = 0
    total_tokens = 0

    pad_class_id = dataloader.dataset.pad_class_id

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            class_ids = batch["class_ids"].to(device)

            logits = model(input_ids)

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = class_ids.view(-1)

            loss = criterion(logits_flat, targets_flat)

            running_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            mask = class_ids != pad_class_id
            correct = (predictions == class_ids) & mask

            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def train_model(train_loader, val_loader, vocab_tokens, vocab_classes):
    """
    Main function to train the model with Early Stopping.
    """
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Initialize Model
    model = LSTMTagger(
        vocab_size=len(vocab_tokens),
        num_classes=len(vocab_classes),
        pad_token_id=vocab_tokens.stoi.get(Config.PAD_TOKEN, 0),
    ).to(device)

    # Calculate Class Weights for Loss
    class_weights = calculate_class_weights(train_loader.dataset, vocab_classes, device)

    # Loss Function
    # ignore_index handles padding in targets
    pad_class_id = vocab_classes.stoi.get(Config.PAD_TOKEN, 0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_class_id)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc:  {train_acc}")

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss:   {val_loss}")
        print(f"Val Acc:    {val_acc}")

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return model
