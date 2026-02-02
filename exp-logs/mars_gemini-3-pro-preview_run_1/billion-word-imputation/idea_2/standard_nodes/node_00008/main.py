import sys
import os
import torch
import numpy as np
import pandas as pd
import logging

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger, calculate_levenshtein
from library.vocab import get_vocabulary
from library.data import get_dataloaders
from library.model import GapTransformer
from library.engine import run_training, validate, generate_submission


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Starting runfile execution...")

    # 2. Configure Fast Baseline
    # Limit training to 1 epoch and truncating data to ensure completion within time limits
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 512

    # 3. Load Vocabulary
    logger.info("Loading vocabulary...")
    vocab = get_vocabulary(load_cached_data=True)

    # 4. Load Data
    logger.info("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab, load_cached_data=True
    )

    # Optimization: Truncate training data for fast baseline execution
    # Target: 500,000 samples
    TARGET_TRAIN_SIZE = 500000
    if hasattr(train_loader.dataset, "df"):
        current_size = len(train_loader.dataset.df)
        if current_size > TARGET_TRAIN_SIZE:
            logger.info(
                f"Truncating training dataset from {current_size} to {TARGET_TRAIN_SIZE} samples."
            )
            train_loader.dataset.df = train_loader.dataset.df.iloc[:TARGET_TRAIN_SIZE]

    # 5. Initialize Model
    logger.info("Initializing model...")
    model = GapTransformer(vocab_size=len(vocab))

    # 6. Train
    logger.info("Starting training...")
    # run_training handles the training loop, optimizer, and saving the best model
    model = run_training(model, train_loader, val_loader, vocab)

    # 7. Final Validation
    logger.info("Performing final validation on full hold-out set...")
    device = torch.device(Config.DEVICE)

    # Compute metric on the full validation set
    val_metric = validate(model, val_loader, vocab, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric}")

    # 8. Failure Analysis
    logger.info("Performing failure analysis...")
    # Analyze a subset of validation data for correlation to save time vs full set
    ANALYSIS_SIZE = 10000

    model.eval()
    lengths = []
    errors = []

    pad_idx = vocab.get_pad_index()
    unk_idx = vocab.get_unk_index()
    no_insert_idx = vocab.get_no_insert_index()

    samples_processed = 0

    with torch.no_grad():
        for batch in val_loader:
            if samples_processed >= ANALYSIS_SIZE:
                break

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # Predict
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(input_ids)

            # Mask special tokens to ensure valid word selection
            logits[:, :, pad_idx] = float("-inf")
            logits[:, :, unk_idx] = float("-inf")
            logits[:, :, no_insert_idx] = float("-inf")

            B, L, V = logits.shape
            flat_logits = logits.view(B, -1)
            _, flat_indices = torch.max(flat_logits, dim=1)

            pred_pos = (flat_indices // V).cpu().numpy()
            pred_word_idx = (flat_indices % V).cpu().numpy()

            input_ids_np = input_ids.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for b in range(B):
                curr_input = input_ids_np[b]
                curr_target = targets_np[b]

                # Extract valid tokens (remove padding)
                valid_tokens = []
                for tid in curr_input:
                    if tid == pad_idx:
                        break
                    valid_tokens.append(vocab.itos[tid])

                # --- Ground Truth Reconstruction ---
                # Find target insertion index
                t_indices = np.where(curr_target[: len(valid_tokens)] != no_insert_idx)[
                    0
                ]
                if len(t_indices) > 0:
                    t_pos = t_indices[0]
                    t_word = vocab.itos[curr_target[t_pos]]
                    ref_tokens = list(valid_tokens)
                    ref_tokens.insert(t_pos + 1, t_word)
                    ref_sent = " ".join(ref_tokens)
                else:
                    ref_sent = " ".join(valid_tokens)

                # --- Prediction Reconstruction ---
                p_pos = pred_pos[b]
                # Clamp position to valid range
                if p_pos >= len(valid_tokens):
                    p_pos = len(valid_tokens) - 1

                p_word = vocab.itos[pred_word_idx[b]]
                hyp_tokens = list(valid_tokens)
                hyp_tokens.insert(p_pos + 1, p_word)
                hyp_sent = " ".join(hyp_tokens)

                # Metric Calculation
                dist = calculate_levenshtein(ref_sent, hyp_sent)

                # Store data for correlation
                lengths.append(len(valid_tokens))
                errors.append(dist)

            samples_processed += B

    # Compute Correlation
    if len(lengths) > 1:
        correlation_matrix = np.corrcoef(lengths, errors)
        correlation = correlation_matrix[0, 1]
        print(
            f"Correlation between Input Length and Error Magnitude: {correlation:.4f}"
        )
    else:
        print("Not enough samples for failure analysis.")

    # 9. Submission
    THRESHOLD = 7.70033
    if val_metric < THRESHOLD:
        logger.info(
            f"Validation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, vocab)
    else:
        logger.info(
            f"Validation metric {val_metric} >= {THRESHOLD}. Skipping submission."
        )

    logger.info("Runfile execution completed.")


if __name__ == "__main__":
    main()
