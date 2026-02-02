import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    decode_predictions_rle,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import GCINet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ground_truth_sequence(labels_tensor):
    """
    Converts frame-wise ground truth labels tensor to a list of gesture IDs.
    Simple RLE extraction since GT is clean (constructed from start/end frames).
    """
    labels = labels_tensor.cpu().numpy()
    # RLE logic: find changes
    # Append 0 to handle last element
    labels_ext = np.concatenate([labels, [labels[-1] + 1]])
    changes = np.where(labels_ext[:-1] != labels_ext[1:])[0]

    sequence = []
    start = 0
    for end in changes:
        lbl = labels[start]
        if lbl != Config.LABEL_MAP["background"]:
            sequence.append(int(lbl))
        start = end + 1
    return sequence


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move inputs to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Output: (B, T, NumClasses+1)
        logits = model(skeleton, audio, lengths, mask)

        # Reshape for Loss: (B*T, NumClasses+1) vs (B*T)
        # We flatten the batch and time dimensions
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)

        # Calculate Loss
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Standard practice for RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in dataloader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            logits = model(skeleton, audio, lengths, mask)

            # Loss calculation
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            num_batches += 1

            # Decode Predictions and Ground Truth for Metric
            # Iterate over batch
            batch_size = logits.shape[0]
            for i in range(batch_size):
                # Get valid length for this sequence
                length = lengths[i].item()

                # Slice valid logits and labels
                seq_logits = logits[i, :length, :]
                seq_labels = labels[i, :length]

                # Decode Prediction
                pred_seq = decode_predictions_rle(seq_logits)
                all_predictions.append(pred_seq)

                # Decode Ground Truth
                gt_seq = get_ground_truth_sequence(seq_labels)
                all_ground_truths.append(gt_seq)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute Levenshtein Error Rate
    ler_score = compute_normalized_levenshtein(all_predictions, all_ground_truths)

    return avg_loss, ler_score


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = GestureDataset(split="train", load_cached_data=load_cached_data)
    val_dataset = GestureDataset(split="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = GCINet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 5. Loss Function
    # Weights: Background (0) gets 0.5, others get 1.0
    class_weights = torch.ones(Config.NUM_CLASSES + 1, device=device)
    class_weights[0] = Config.BG_WEIGHT

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 6. Training Loop
    best_ler = float("inf")
    patience = 10
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ler = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val LER: {val_ler:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_ler < best_ler:
            best_ler = val_ler
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! (LER: {val_ler:.10f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training Complete. Best Validation LER: {best_ler:.10f}")
