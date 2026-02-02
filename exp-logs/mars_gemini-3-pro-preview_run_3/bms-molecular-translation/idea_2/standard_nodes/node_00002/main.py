import sys
import os
import torch
import pandas as pd
import numpy as np
import nltk
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.utils import compute_levenshtein, load_checkpoint
from library.tokenizer import Tokenizer
from library.model import InChiModel


def main():
    # --- 1. Configuration & Setup ---
    print("--- Setting up Baseline Run ---")
    Config.setup()

    # Fast baseline settings to ensure completion within 2 hours
    Config.NUM_EPOCHS = 1
    # Using a subset for training to demonstrate the pipeline quickly
    TRAIN_SUBSET_SIZE = 30000

    # --- 2. Training ---
    print("\n--- Initializing Trainer ---")
    # Initialize trainer (loads full datasets by default if debug=False)
    trainer = Trainer(load_cached_data=True, debug=False)

    # Optimization: Replace train_loader with a smaller subset for speed
    print(
        f"Optimizing: Replacing train loader with subset of {TRAIN_SUBSET_SIZE} samples."
    )
    subset_train_loader, _, _ = get_dataloaders(
        tokenizer=trainer.tokenizer,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_subset_size=TRAIN_SUBSET_SIZE,
    )
    trainer.train_loader = subset_train_loader

    # Reset scheduler with new steps per epoch based on the subset size
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(trainer.train_loader),
        epochs=Config.NUM_EPOCHS,
        pct_start=0.1,
    )

    print("--- Starting Training ---")
    trainer.fit()

    # --- 3. Validation & Metric Calculation ---
    print("\n--- Starting Validation Assessment ---")

    # Load best model
    best_model_path = os.path.join(
        Config.WORKING_DIR, "checkpoints", "model_best.pth.tar"
    )
    if not os.path.exists(best_model_path):
        # Fallback if no improvement was recorded (unlikely with 1 epoch but safe to handle)
        best_model_path = os.path.join(
            Config.WORKING_DIR, "checkpoints", "checkpoint.pth.tar"
        )

    load_checkpoint(best_model_path, trainer.model)
    trainer.model.eval()

    val_predictions = []
    val_targets = []
    val_target_lengths = []

    # Use the validation loader from the trainer (Full validation set)
    print(f"Validating on {len(trainer.val_loader.dataset)} samples...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(trainer.val_loader):
            images = images.to(Config.DEVICE)

            # Predict using greedy decoding
            pred_seqs = trainer.model.predict(
                images, trainer.tokenizer, max_len=Config.MAX_LEN
            )

            # Decode sequences to strings
            for j, seq in enumerate(pred_seqs):
                pred_str = trainer.tokenizer.sequence_to_text(seq)
                val_predictions.append(pred_str)

                # Decode target (remove padding/special tokens)
                target_seq = labels[j]
                target_str = trainer.tokenizer.sequence_to_text(target_seq)
                val_targets.append(target_str)
                val_target_lengths.append(len(target_str))

            # Print progress periodically
            if i % 100 == 0:
                print(f"Validation Batch {i}/{len(trainer.val_loader)}")

    # Compute Metric
    metric = compute_levenshtein(val_predictions, val_targets)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # --- 4. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate individual levenshtein distances for analysis
    lev_distances = []
    for p, t in zip(val_predictions, val_targets):
        lev_distances.append(nltk.edit_distance(p, t))

    lev_distances = np.array(lev_distances)
    val_target_lengths = np.array(val_target_lengths)

    # Correlation: Error vs Target Length
    if len(lev_distances) > 1:
        corr_len, _ = pearsonr(lev_distances, val_target_lengths)
        print(f"Correlation (Error vs InChI Length): {corr_len:.4f}")

        if abs(corr_len) > 0.3:
            print("Insight: Model struggles significantly more with longer sequences.")
        else:
            print("Insight: Error is relatively independent of sequence length.")

        # Basic stats
        print(f"Mean Error: {np.mean(lev_distances):.4f}")
        print(f"Max Error: {np.max(lev_distances)}")

    # --- 5. Submission Generation ---
    # Generates predictions for the test set and saves to submission.csv
    trainer.predict_test()


if __name__ == "__main__":
    main()
