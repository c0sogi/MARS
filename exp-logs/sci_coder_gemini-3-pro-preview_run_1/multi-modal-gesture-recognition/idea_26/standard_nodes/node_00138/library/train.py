import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_data_loaders
from library.model import SAMPNet


def decode_predictions(frame_logits, lengths):
    """
    Decodes frame-wise logits into a list of gesture sequences.
    Applies Median Filter and Run-Length Encoding (RLE) with length filtering.
    """
    predictions = []
    # Get hard predictions: (B, T)
    preds = torch.argmax(frame_logits, dim=2).cpu().numpy()

    for i in range(preds.shape[0]):
        length = lengths[i]
        raw_pred = preds[i, :length]

        # 1. Median Filter (Smoothing)
        if len(raw_pred) >= Config.MEDIAN_FILTER_KERNEL:
            filtered_pred = scipy.signal.medfilt(
                raw_pred, kernel_size=Config.MEDIAN_FILTER_KERNEL
            )
        else:
            filtered_pred = raw_pred

        # 2. Run-Length Encoding & Filtering
        sequence = []
        if len(filtered_pred) > 0:
            current_val = filtered_pred[0]
            current_len = 1

            def append_if_valid(val, seq_len):
                # Filter out background (0) and short segments
                if val != 0 and seq_len >= Config.MIN_SEGMENT_LENGTH:
                    sequence.append(int(val))

            for j in range(1, len(filtered_pred)):
                val = filtered_pred[j]
                if val == current_val:
                    current_len += 1
                else:
                    append_if_valid(current_val, current_len)
                    current_val = val
                    current_len = 1
            # Append the last segment
            append_if_valid(current_val, current_len)

        predictions.append(sequence)

    return predictions


def decode_ground_truth(frame_labels, lengths):
    """
    Decodes frame-wise ground truth labels into a list of gesture sequences.
    Collapses duplicates and removes background class (0).
    """
    labels = frame_labels.cpu().numpy()
    ground_truths = []

    for i in range(labels.shape[0]):
        length = lengths[i]
        seq_raw = labels[i, :length]

        sequence = []
        if len(seq_raw) > 0:
            current_val = seq_raw[0]
            if current_val != 0:
                sequence.append(int(current_val))

            for j in range(1, len(seq_raw)):
                val = seq_raw[j]
                if val != current_val:
                    if val != 0:
                        sequence.append(int(val))
                    current_val = val
        ground_truths.append(sequence)

    return ground_truths


def train_one_epoch(model, loader, optimizer, criterion_frame, criterion_aux, device):
    """
    Executes one training epoch.
    Computes composite loss: Frame Loss + 0.2 * Aux Loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        skel, audio, labels, aux_targets, lengths = batch
        if skel is None:
            continue

        skel = skel.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        aux_targets = aux_targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, aux_preds = model(skel, audio, lengths)

        # Compute Frame Loss (CrossEntropy)
        # Flatten outputs and targets: (B*T, C) and (B*T)
        loss_frame = criterion_frame(
            logits.view(-1, Config.NUM_CLASSES), labels.view(-1)
        )

        # Compute Auxiliary Loss (BCE)
        if aux_preds is not None:
            loss_aux = criterion_aux(aux_preds, aux_targets)
            loss = loss_frame + Config.AUX_LOSS_WEIGHT * loss_aux
        else:
            loss = loss_frame

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the Levenshtein Error Rate.
    """
    model.eval()
    all_preds = []
    all_truths = []

    with torch.no_grad():
        for batch in loader:
            skel, audio, labels, aux_targets, lengths = batch
            if skel is None:
                continue

            skel = skel.to(device)
            audio = audio.to(device)

            # Forward pass
            logits, _ = model(skel, audio, lengths)

            # Decode sequences
            batch_preds = decode_predictions(logits, lengths)
            batch_truths = decode_ground_truth(labels, lengths)

            all_preds.extend(batch_preds)
            all_truths.extend(batch_truths)

    # Compute metric
    error_rate = compute_levenshtein(all_preds, all_truths)
    return error_rate


def run_training(debug=False):
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(debug=debug)

    # 3. Model
    model = SAMPNet().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Loss Functions
    # Define class weights: Background (index 0) gets 0.5, others 1.0
    class_weights = torch.ones(Config.NUM_CLASSES).to(device)
    class_weights[0] = Config.BACKGROUND_CLASS_WEIGHT

    criterion_frame = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )
    criterion_aux = nn.BCELoss()

    # 6. Training Loop
    best_error_rate = float("inf")
    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_frame, criterion_aux, device
        )
        val_error = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.8f} | Val Error Rate: {val_error:.10f}"
        )

        # Checkpointing
        if val_error < best_error_rate:
            best_error_rate = val_error
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation Error Rate: {best_error_rate:.10f}")
