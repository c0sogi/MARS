import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nltk
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    compute_levenshtein,
)
from library.dataset import get_dataloaders
from library.model import DecoderOnlyTransformer
from library.train import train_one_epoch
from library.tokenizer import Tokenizer


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000  # Sufficient sample for a baseline
    Config.BATCH_SIZE = 32  # Fit comfortably on A100
    Config.NUM_WORKERS = 4

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        debug=Config.DEBUG
    )
    vocab_size = len(tokenizer)
    print(f"Vocabulary Size: {vocab_size}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = DecoderOnlyTransformer(vocab_size=vocab_size)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # Loss & Scaler
    pad_idx = tokenizer.stoi[Config.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scaler = GradScaler()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting Training...")
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.EPOCHS} ---")

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch, scaler
        )

        # Validation Step (Loss Only for Speed during training)
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                text_input_ids = labels[:, :-1]
                target_ids = labels[:, 1:]

                # Forward pass (no autocast needed for validation loss usually, but consistent with train)
                logits = model(images, text_input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )

                val_running_loss += loss.item()
                val_batches += 1

        val_loss = val_running_loss / val_batches if val_batches > 0 else 0.0
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Save Checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print("New best model saved.")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_val_loss,
            },
            is_best,
        )

    # -------------------------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis & Final Validation ---")
    # Load best model
    load_checkpoint(model, filename=Config.BEST_MODEL_PATH)
    model.eval()

    val_predictions = []
    val_targets = []
    val_inchi_lengths = []

    # We need to correlate error with input features.
    # We'll use InChI length as a proxy for complexity/input feature.

    print("Generating predictions on validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)

            # Generate predictions
            batch_preds = model.generate(images, tokenizer, max_len=Config.MAX_TEXT_LEN)

            # Decode targets
            batch_targets = [
                tokenizer.sequence_to_text(seq, remove_special_tokens=True)
                for seq in labels.cpu().numpy()
            ]

            val_predictions.extend(batch_preds)
            val_targets.extend(batch_targets)

            # Record lengths for failure analysis
            val_inchi_lengths.extend([len(t) for t in batch_targets])

            if i % 10 == 0:
                print(f"Validated batch {i}/{len(val_loader)}")

    # Compute Metrics
    lev_distances = []
    for p, t in zip(val_predictions, val_targets):
        lev_distances.append(nltk.edit_distance(p, t))

    mean_lev_distance = np.mean(lev_distances)
    print(f"Final Validation Metric: {mean_lev_distance}")

    # Failure Analysis: Correlation
    if len(lev_distances) > 1:
        correlation = np.corrcoef(lev_distances, val_inchi_lengths)[0, 1]
        print(f"Correlation (Levenshtein Error vs InChI Length): {correlation:.4f}")
        if correlation > 0.3:
            print(
                "Analysis: Strong positive correlation. Longer molecules are harder to predict."
            )
        elif correlation < -0.3:
            print("Analysis: Negative correlation.")
        else:
            print("Analysis: Weak correlation.")
    else:
        print("Not enough samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 6. Test Prediction & Submission
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    test_predictions = []

    # Test loader is already initialized
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device)
            batch_preds = model.generate(images, tokenizer, max_len=Config.MAX_TEXT_LEN)
            test_predictions.extend(batch_preds)

            if i % 10 == 0:
                print(f"Test batch {i}/{len(test_loader)}")

    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_METADATA)
    if Config.DEBUG:
        test_meta = test_meta.head(Config.DEBUG_SAMPLE_SIZE)

    test_ids = test_meta["image_id"].values

    # Ensure lengths match
    if len(test_predictions) != len(test_ids):
        print(
            f"Warning: Preds ({len(test_predictions)}) != IDs ({len(test_ids)}). Truncating to minimum."
        )
        min_len = min(len(test_predictions), len(test_ids))
        test_predictions = test_predictions[:min_len]
        test_ids = test_ids[:min_len]

    submission_df = pd.DataFrame({"image_id": test_ids, "InChI": test_predictions})

    submission_df.to_csv(Config.PREDICTIONS_CSV, index=False)
    print(f"Submission saved to {Config.PREDICTIONS_CSV}")


if __name__ == "__main__":
    run_pipeline()
