import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import groupby
from torch.utils.data import DataLoader

from library.config import Config
from library.model import SCRNet
from library.data_loader import GestureDataset, collate_fn
from library.utils import set_seed, compute_levenshtein, batch_decode


def get_gt_sequences(dense_labels, lengths):
    """
    Converts dense frame-wise labels into a list of gesture ID sequences.
    Used for ground truth generation during validation.
    """
    gt_sequences = []
    dense_labels_np = dense_labels.cpu().numpy()
    lengths_np = lengths.cpu().numpy()

    for i in range(len(dense_labels_np)):
        valid_len = lengths_np[i]
        # Extract valid frames
        valid_seq = dense_labels_np[i, :valid_len]

        # Run-Length Encoding to get sequence, ignoring background (0)
        # We do not apply min_gesture_length filtering to GT, assuming GT is correct.
        seq = [int(k) for k, g in groupby(valid_seq) if k != 0]
        gt_sequences.append(seq)

    return gt_sequences


def train_model(
    num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    # 1. Reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")
    print(f"Config: Epochs={num_epochs}, Batch={batch_size}, Debug={debug}")

    # 2. Data Loading
    # Load datasets
    train_dataset = GestureDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, is_train=True, load_cached_data=True
    )
    val_dataset = GestureDataset(
        metadata_path=Config.VAL_METADATA_PATH, is_train=False, load_cached_data=True
    )

    if debug:
        # Subset for debugging
        indices = torch.arange(min(len(train_dataset), 20))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print("Debug mode: Reduced dataset size.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model Setup
    model = SCRNet().to(device)

    # 4. Loss Setup
    # Class weights: Background (0) gets 0.7, others 1.0
    class_weights = torch.ones(Config.NUM_CLASSES).to(device)
    class_weights[0] = Config.BACKGROUND_WEIGHT

    # We use reduction='mean' to supervise padding. Cite {solution_lesson_node_00076}
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
    )

    # 5. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 6. Training Loop
    best_val_score = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (skeletons, audios, labels, lengths) in enumerate(train_loader):
            skeletons = skeletons.to(device)
            audios = audios.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            # Forward
            logits = model(skeletons, audios, lengths)  # (B, T, C)

            # Reshape for loss: (B*T, C) vs (B*T)
            B, T, C = logits.shape

            # Compute loss over all frames, including padding (Supervised Padding)
            # Cite {solution_lesson_node_00076}
            loss = criterion(logits.reshape(-1, C), labels.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 7. Validation Loop
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for skeletons, audios, labels, lengths in val_loader:
                skeletons = skeletons.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                logits = model(skeletons, audios, lengths)

                # Decode predictions
                batch_preds = batch_decode(logits, lengths)
                val_preds.extend(batch_preds)

                # Decode targets (from dense labels to sequence)
                batch_targets = get_gt_sequences(labels, lengths)
                val_targets.extend(batch_targets)

        # Compute Metric
        val_score = compute_levenshtein(val_preds, val_targets)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Levenshtein: {val_score}"
        )

        # 8. Checkpointing
        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with score: {best_val_score}")

    print("Training complete.")
    return best_val_score
