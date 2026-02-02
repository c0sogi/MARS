import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm

# Import library modules
from library.config import Config, set_seed
from library.utils import get_device, save_checkpoint, logger
from library.vocab import Vocabulary
from library.dataset import get_dataloaders
from library.model import GatedInfillingModel
from library.engine import Engine


def run():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for Fast Baseline to ensure completion within 2 hours
    Config.DEBUG_SAMPLE_SIZE = 100000  # Train/Val on 100k samples
    Config.NUM_EPOCHS = 2  # Limited epochs for baseline
    Config.BATCH_SIZE = 256  # Larger batch size for A100

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)
    device = get_device()
    logger.info(f"Execution Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    logger.info("--- Data Preparation ---")
    logger.info("Building Vocabulary...")
    vocab = Vocabulary()
    # Build or load vocabulary. Uses DEBUG_SAMPLE_SIZE if building from scratch.
    vocab.build_from_corpus(load_cached_data=True)

    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab, load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    logger.info("--- Model Initialization ---")
    model = GatedInfillingModel(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        kernel_size=Config.KERNEL_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        padding_idx=vocab.pad_token_id,
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR for fast convergence
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps, pct_start=0.3
    )

    # Engine instance for training helpers
    engine = Engine(model, optimizer, scheduler, vocab, device)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    logger.info("--- Starting Training ---")
    best_val_loss = float("inf")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train one epoch
        engine.train_one_epoch(train_loader, epoch)

        # Validation (Standard CrossEntropy/Accuracy for monitoring)
        val_metrics = engine.validate(val_loader, epoch)

        # Save Checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, best_val_loss, Config.MODEL_FILE)
            logger.info(
                f"New best model saved at epoch {epoch} with loss {best_val_loss:.4f}"
            )

    # --------------------------------------------------------------------------
    # 5. Validation Metric (Levenshtein) & Failure Analysis
    # --------------------------------------------------------------------------
    logger.info("--- Performing Detailed Validation & Failure Analysis ---")

    # Load best model for evaluation
    checkpoint = torch.load(Config.MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    levenshtein_distances = []
    sentence_lengths = []

    # Pre-fetch special token IDs
    pad_id = vocab.stoi[vocab.TOKEN_PAD]
    unk_id = vocab.stoi[vocab.TOKEN_UNK]
    no_insert_id = vocab.stoi[vocab.TOKEN_NO_INSERT]
    start_id = vocab.stoi[vocab.TOKEN_START]
    end_id = vocab.stoi[vocab.TOKEN_END]
    mask_ids = [pad_id, unk_id, no_insert_id, start_id, end_id]

    # Disable gradients for inference
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            logits = model(input_ids)  # (Batch, Seq_Len, Vocab)
            probs = torch.softmax(logits, dim=-1)

            # Mask special tokens in probability space to prevent invalid predictions
            for mid in mask_ids:
                probs[:, :, mid] = 0.0

            # Move to CPU for string reconstruction and metric calculation
            batch_probs = probs.cpu().numpy()
            batch_input = input_ids.cpu().numpy()
            batch_targets = targets.cpu().numpy()

            for i in range(len(batch_input)):
                seq = batch_input[i]
                tgt = batch_targets[i]
                p = batch_probs[i]

                # --- 1. Reconstruct Ground Truth Sentence ---
                # Find effective sequence length (up to [END] token)
                try:
                    end_pos = np.where(seq == end_id)[0][0]
                except IndexError:
                    end_pos = len(seq)

                # Base tokens (excluding START at 0 and END/PAD)
                # seq[1] corresponds to the first actual word
                base_tokens = [vocab.get_token(tid) for tid in seq[1:end_pos]]

                # Identify where the insertion should happen
                # target != -100 (ignore_index) and target != NO_INSERT
                valid_target_mask = (tgt != -100) & (tgt != no_insert_id)
                true_gap_indices = np.where(valid_target_mask)[0]

                true_tokens = list(base_tokens)
                if len(true_gap_indices) > 0:
                    t_idx = true_gap_indices[0]
                    t_word_id = tgt[t_idx]
                    t_word = vocab.get_token(t_word_id)

                    # Logic: t_idx is the index in `seq`. `seq[t_idx]` is the token BEFORE the gap.
                    # `seq[1]` is base_tokens[0]. `seq[t_idx]` is base_tokens[t_idx - 1].
                    # We insert AFTER base_tokens[t_idx - 1], which is index `t_idx` in the list.
                    ins_pos = min(t_idx, len(true_tokens))
                    true_tokens.insert(ins_pos, t_word)

                true_sentence = " ".join(true_tokens)

                # --- 2. Reconstruct Predicted Sentence ---
                # Mask invalid gap positions for prediction
                p[0, :] = 0.0  # Cannot insert before START
                p[end_pos:, :] = 0.0  # Cannot insert after END or in PAD
                if end_pos > 0:
                    p[end_pos - 1, :] = (
                        0.0  # Cannot insert after last word (before END) per task spec
                    )

                # Find max probability (Gap, Word)
                flat_idx = np.argmax(p)
                pred_gap_idx, pred_word_idx = np.unravel_index(flat_idx, p.shape)
                pred_word = vocab.get_token(pred_word_idx)

                pred_tokens = list(base_tokens)
                ins_pos_pred = min(pred_gap_idx, len(pred_tokens))
                pred_tokens.insert(ins_pos_pred, pred_word)

                pred_sentence = " ".join(pred_tokens)

                # --- 3. Compute Metric ---
                dist = nltk.edit_distance(true_sentence, pred_sentence)
                levenshtein_distances.append(dist)
                sentence_lengths.append(len(true_tokens))

    # Output Final Metric
    if len(levenshtein_distances) > 0:
        mean_levenshtein = np.mean(levenshtein_distances)
        print(f"Final Validation Metric: {mean_levenshtein}")

        # Failure Analysis
        correlation = np.corrcoef(sentence_lengths, levenshtein_distances)[0, 1]
        print(
            f"Correlation between Sentence Length and Error (Levenshtein): {correlation}"
        )
    else:
        print("Final Validation Metric: N/A (No validation samples)")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    logger.info("--- Generating Submission ---")
    engine.generate_submission(test_loader, Config.SUBMISSION_FILE)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
    logger.info("Run Completed Successfully.")


if __name__ == "__main__":
    run()
