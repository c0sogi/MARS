import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import calculate_f1_score
from library.model import DualStreamAttentionDAN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move inputs to device
        title_text = batch["title_text"].to(device)
        title_offsets = batch["title_offsets"].to(device)
        body_text = batch["body_text"].to(device)
        body_offsets = batch["body_offsets"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(title_text, title_offsets, body_text, body_offsets)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, loader, criterion, device, threshold=Config.THRESHOLD):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            # Move inputs to device
            title_text = batch["title_text"].to(device)
            title_offsets = batch["title_offsets"].to(device)
            body_text = batch["body_text"].to(device)
            body_offsets = batch["body_offsets"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            logits = model(title_text, title_offsets, body_text, body_offsets)

            # Compute loss
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Compute F1 Score for this batch
            # calculate_f1_score expects numpy arrays or tensors, handles sigmoid internally
            batch_f1 = calculate_f1_score(logits, targets, threshold=threshold)

            # Weighted average accumulation
            batch_size = targets.size(0)
            total_f1 += batch_f1 * batch_size
            total_samples += batch_size
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_f1


def train_model(model, train_loader, val_loader, device, num_epochs=Config.NUM_EPOCHS):
    """
    Main training loop with Early Stopping.
    """
    print(f"Starting training on device: {device}")

    # Optimizer and Loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val F1 Score: {val_f1}")

        # Checkpoint and Early Stopping
        if val_f1 > best_val_f1:
            print(
                f"Validation F1 improved from {best_val_f1} to {val_f1}. Saving model..."
            )
            best_val_f1 = val_f1
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation F1 did not improve. Patience: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Explicit garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training complete. Best Val F1: {best_val_f1}")


def predict_test(vocab, test_loader, device, threshold=Config.THRESHOLD):
    """
    Generates predictions for the test set and saves submission file.
    """
    print("Starting inference on test set...")

    # Initialize model architecture
    model = DualStreamAttentionDAN(
        vocab_size=vocab.get_vocab_size(),
        num_classes=vocab.get_num_tags(),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        init_range=Config.INIT_RANGE,
    )

    # Load best weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file {Config.MODEL_PATH} not found.")

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ids_list = []
    tags_list = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs
            title_text = batch["title_text"].to(device)
            title_offsets = batch["title_offsets"].to(device)
            body_text = batch["body_text"].to(device)
            body_offsets = batch["body_offsets"].to(device)
            ids = batch["ids"].numpy()  # Keep IDs on CPU

            # Forward pass
            logits = model(title_text, title_offsets, body_text, body_offsets)

            # Apply Sigmoid
            probs = torch.sigmoid(logits)

            # Thresholding
            # Convert to CPU numpy for processing
            probs_np = probs.cpu().numpy()

            # Generate tags for each sample in batch
            for i in range(len(ids)):
                sample_probs = probs_np[i]
                # Get indices where prob > threshold
                pred_indices = np.where(sample_probs > threshold)[0]

                # Convert indices to string
                # vocab.indices_to_tags handles the joining
                tag_str = vocab.indices_to_tags(pred_indices)

                ids_list.append(ids[i])
                tags_list.append(tag_str)

    # Create Submission DataFrame
    df_submission = pd.DataFrame({"Id": ids_list, "Tags": tags_list})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
