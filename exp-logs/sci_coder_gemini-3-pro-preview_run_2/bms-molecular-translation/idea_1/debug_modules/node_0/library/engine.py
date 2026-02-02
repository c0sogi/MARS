import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

from library.config import Config
from library.dataset import ChemicalDataset
from library.model import CRNN
from library.utils import AverageMeter, compute_levenshtein
from library.tokenizer import Tokenizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    """
    Custom collate function to pad variable length target sequences.
    """
    # batch is a list of tuples: (image, label_seq, label_len)
    images, sequences, lengths = zip(*batch)

    # Stack images: (B, C, H, W)
    images = torch.stack(images, 0)

    # Stack lengths: (B,)
    lengths = torch.stack(lengths, 0)

    # Pad sequences with BLANK_IDX: (B, Max_Len)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=Config.BLANK_IDX
    )

    return images, padded_seqs, lengths


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, targets, target_lengths) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        batch_size = images.size(0)

        # Forward pass
        # Output shape: (B, T, C) -> (Batch, Time, Classes)
        log_probs = model(images)

        # CTC Loss expects:
        # log_probs: (T, B, C)
        log_probs_ctc = log_probs.permute(1, 0, 2)

        # Input lengths are constant (T) for all images in batch because of fixed image width
        T = log_probs.size(1)
        input_lengths = torch.full(
            size=(batch_size,), fill_value=T, dtype=torch.long
        ).to(device)

        loss = criterion(log_probs_ctc, targets, input_lengths, target_lengths)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_NORM)
        optimizer.step()

        losses.update(loss.item(), batch_size)

    return losses.avg


def validate(model, loader, criterion, tokenizer, device):
    model.eval()
    losses = AverageMeter()
    levenshtein_scores = AverageMeter()

    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            targets_dev = targets.to(device)
            target_lengths = target_lengths.to(device)
            batch_size = images.size(0)

            # Forward
            log_probs = model(images)  # (B, T, C)
            log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, C)

            T = log_probs.size(1)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=T, dtype=torch.long
            ).to(device)

            # Loss
            loss = criterion(log_probs_ctc, targets_dev, input_lengths, target_lengths)
            losses.update(loss.item(), batch_size)

            # Decoding for Metric
            # Get argmax indices: (B, T)
            preds = torch.argmax(log_probs, dim=2)
            decoded_preds = tokenizer.decode_batch(preds)

            # Reconstruct target strings from indices
            decoded_targets = []
            targets_cpu = targets.cpu()

            for i in range(batch_size):
                length = target_lengths[i].item()
                # Slice the padded tensor to get the actual sequence
                seq = targets_cpu[i][:length]
                seq_list = seq.tolist()
                # Map indices to characters
                t_str = "".join([tokenizer.idx2char.get(idx, "") for idx in seq_list])
                decoded_targets.append(t_str)

            # Compute Metric
            score = compute_levenshtein(decoded_preds, decoded_targets)
            levenshtein_scores.update(score, batch_size)

    return losses.avg, levenshtein_scores.avg


def fit(epochs=Config.EPOCHS, load_cached_data=True):
    set_seed(42)
    device = Config.DEVICE
    print(f"Initializing Model on {device}...")

    model = CRNN().to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # CTC Loss
    criterion = nn.CTCLoss(blank=Config.BLANK_IDX, reduction="mean", zero_infinity=True)

    # Datasets
    print("Loading Datasets...")
    train_dataset = ChemicalDataset(mode="train", load_cached_data=load_cached_data)
    val_dataset = ChemicalDataset(mode="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    tokenizer = Tokenizer()

    best_lev = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_lev = validate(model, val_loader, criterion, tokenizer, device)

        duration = time.time() - start_time

        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {duration:.0f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val Levenshtein: {val_lev}"
        )

        # Early Stopping & Checkpointing
        if val_lev < best_lev:
            best_lev = val_lev
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Levenshtein: {best_lev})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def predict_and_submit(model_path):
    print("Loading Best Model for Inference...")
    device = Config.DEVICE
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = ChemicalDataset(mode="test", load_cached_data=True)
    # Test loader doesn't need custom collate because it returns (image, id_string)
    # Default collate handles stacking images and creating a list of strings
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    tokenizer = Tokenizer()
    results = []

    print("Generating Predictions...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Forward
            log_probs = model(images)  # (B, T, C)

            # Greedy Decode
            preds = torch.argmax(log_probs, dim=2)  # (B, T)
            decoded_strs = tokenizer.decode_batch(preds)

            for img_id, pred_str in zip(image_ids, decoded_strs):
                results.append({"image_id": img_id, "InChI": pred_str})

    # Save Submission
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Total predictions: {len(df_sub)}")
