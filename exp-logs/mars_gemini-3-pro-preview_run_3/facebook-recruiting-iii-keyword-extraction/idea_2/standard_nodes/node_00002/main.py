import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_processing import get_dataloaders, setup_directories
from library.model import NBOWModel
from library.trainer import Trainer, run_inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override defaults for a fast baseline execution
    Config.NUM_EPOCHS = 3

    setup_directories()
    set_seed(Config.SEED)

    print(f"Configuration: Device={Config.DEVICE}, Epochs={Config.NUM_EPOCHS}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # load_cached_data=True ensures we use pre-processed numpy files if they exist
    train_loader, val_loader, test_loader, mlb = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = NBOWModel(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=len(mlb.classes_),
        dropout=Config.DROPOUT,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # 4. Training
    # ==========================================
    print("Starting training...")
    trainer = Trainer(model, optimizer, criterion, device=Config.DEVICE)
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Load the best model checkpoint for analysis and inference
    print("Loading best model checkpoint...")
    load_checkpoint(trainer.model, trainer.optimizer, path=Config.MODEL_SAVE_PATH)

    # ==========================================
    # 5. Validation & Failure Analysis
    # ==========================================
    print("Running Validation Analysis...")
    trainer.model.eval()

    val_f1_scores = []
    val_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, offsets, targets = batch

            # Move data to device
            input_ids = input_ids.to(Config.DEVICE)
            offsets = offsets.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            logits = trainer.model(input_ids, offsets)
            probs = torch.sigmoid(logits)

            # Binarize predictions (threshold 0.5)
            preds = (probs >= 0.5).float()

            # Calculate F1 per sample
            # TP = (preds * targets).sum()
            # FP = (preds * (1-targets)).sum()
            # FN = ((1-preds) * targets).sum()
            tp = (preds * targets).sum(dim=1)
            fp = (preds * (1 - targets)).sum(dim=1)
            fn = ((1 - preds) * targets).sum(dim=1)

            # F1 = 2*TP / (2*TP + FP + FN)
            # Add epsilon to avoid division by zero
            f1_batch = 2 * tp / (2 * tp + fp + fn + 1e-9)

            val_f1_scores.append(f1_batch.cpu().numpy())

            # Calculate Sequence Lengths for Failure Analysis
            # offsets is a 1D tensor of start indices: [start_0, start_1, ...]
            # The length of sequence i is start_{i+1} - start_i
            batch_offsets = offsets.cpu().numpy()
            total_tokens = input_ids.size(0)

            # Create array of next offsets to compute diffs
            next_offsets = np.append(batch_offsets[1:], total_tokens)
            lengths_batch = next_offsets - batch_offsets
            val_lengths.append(lengths_batch)

    # Concatenate results
    val_f1_scores = np.concatenate(val_f1_scores)
    val_lengths = np.concatenate(val_lengths)

    # Compute Final Metric (Mean F1-Score)
    final_metric = np.mean(val_f1_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Input Length and Error
    # Error is defined as 1 - F1 Score
    errors = 1.0 - val_f1_scores
    correlation, _ = pearsonr(val_lengths, errors)

    print(
        f"Failure Analysis: Correlation between Input Length and Error (1-F1): {correlation}"
    )

    # ==========================================
    # 6. Submission
    # ==========================================
    # run_inference handles prediction loop and CSV creation
    run_inference(trainer, test_loader, mlb)


if __name__ == "__main__":
    main()
