import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import set_seed, calculate_f1_score
from library.tokenizer import TextProcessor
from library.dataset import StackExchangeDataset
from library.model import FastTextClassifier
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Tokenizer
    print("Initializing Tokenizer...")
    tokenizer = TextProcessor()
    tokenizer.fit(load_cached_data=True)

    # 3. Data Loading
    print("Preparing Datasets...")
    train_dataset = StackExchangeDataset(
        metadata_path=Config.TRAIN_METADATA,
        tokenizer=tokenizer,
        split_name="train",
        load_cached_data=True,
    )

    val_dataset = StackExchangeDataset(
        metadata_path=Config.VAL_METADATA,
        tokenizer=tokenizer,
        split_name="val",
        load_cached_data=True,
    )

    # Fast Baseline: Limit training samples to ensure speed
    MAX_TRAIN_SAMPLES = 500000
    if len(train_dataset) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training set from {len(train_dataset)} to {MAX_TRAIN_SAMPLES} for fast baseline..."
        )
        indices = torch.randperm(len(train_dataset))[:MAX_TRAIN_SAMPLES]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=StackExchangeDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=StackExchangeDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    # 4. Model Initialization
    vocab_size = tokenizer.get_vocab_size()
    num_classes = tokenizer.get_num_tags()

    print(
        f"Model Config - Vocab: {vocab_size}, Classes: {num_classes}, Embed: {Config.EMBEDDING_DIM}"
    )

    model = FastTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout=Config.DROPOUT,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        device=device,
    )

    print("Starting Training...")
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # 6. Final Validation & Metrics
    print("Performing Final Validation...")
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()
    all_preds = []
    all_targets = []
    all_lengths = []

    with torch.no_grad():
        for text, offsets, targets, _ in val_loader:
            text = text.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)

            logits = model(text, offsets)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.TAG_THRESHOLD).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Calculate sequence lengths from offsets
            # Offsets are start indices. Length[i] = Offset[i+1] - Offset[i]
            # Total length of 'text' is needed for the last element
            batch_offsets = offsets.cpu().numpy()
            total_tokens = text.size(0)

            # Append total_tokens to calculate the length of the last sequence
            extended_offsets = np.append(batch_offsets, total_tokens)
            lengths = extended_offsets[1:] - extended_offsets[:-1]
            all_lengths.append(lengths)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_lengths = np.concatenate(all_lengths)

    # Calculate Final Metric
    final_f1 = f1_score(all_targets, all_preds, average="samples", zero_division=0)
    print(f"Final Validation Metric: {final_f1}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate F1 per sample
    # We can do this efficiently using boolean operations
    # Intersection: (preds * targets).sum(axis=1)
    # Union: (preds + targets).clip(0, 1).sum(axis=1) (but for binary, sum of ones is simply sum)
    # F1 = 2 * TP / (2*TP + FP + FN) = 2 * intersection / (sum(preds) + sum(targets))

    numerator = 2 * (all_preds * all_targets).sum(axis=1)
    denominator = all_preds.sum(axis=1) + all_targets.sum(axis=1)

    # Avoid division by zero
    sample_f1s = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    # Error magnitude = 1 - F1
    errors = 1.0 - sample_f1s

    # Correlation
    corr, p_val = pearsonr(all_lengths, errors)
    print(
        f"Correlation between Input Sequence Length and Error Magnitude (1-F1): {corr:.4f} (p={p_val:.4e})"
    )

    # 8. Submission
    print("Generating Submission...")
    test_dataset = StackExchangeDataset(
        metadata_path=Config.TEST_METADATA,
        tokenizer=tokenizer,
        split_name="test",
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=StackExchangeDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    trainer.generate_submission(test_loader)
    print("Workflow Completed.")


if __name__ == "__main__":
    main()
