import os
import torch
import numpy as np
import pandas as pd
from nltk.metrics import edit_distance
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.utils import SOS_TOKEN, EOS_TOKEN


def run():
    # --------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # --------------------------------------------------------------------------
    # Override Config parameters to ensure execution within time limits
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50000  # Use 50k samples for fast training/val
    Config.MAX_EPOCHS = 2  # Limit to 2 epochs
    Config.BATCH_SIZE = 128  # Efficient batch size for A100
    Config.NUM_WORKERS = 4

    # Initialize directories and seeds
    Config.setup()
    Config.set_seed()

    print(f"--- Configuration ---")
    print(f"DEBUG Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Epochs: {Config.MAX_EPOCHS}")
    print("-" * 30)

    # --------------------------------------------------------------------------
    # 2. Training Loop
    # --------------------------------------------------------------------------
    # Initialize Trainer (this also builds/loads the vocabulary)
    trainer = Trainer()

    # Load Data
    print("Loading dataloaders...")
    # Cite debug_lesson_3: Disable cache loading to ensure new sample size is used
    train_loader, val_loader, test_loader = get_dataloaders(
        trainer.vocab, load_cached_data=False
    )

    # Initialize Scheduler manually (usually done in fit())
    total_steps = len(train_loader) * Config.MAX_EPOCHS
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps
    )

    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(Config.MAX_EPOCHS):
        # Run Training Epoch
        trainer.train_epoch(train_loader, epoch)

        # Run Validation (Loss Calculation)
        val_loss = trainer.validate(val_loader, epoch)

        # Checkpoint Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(trainer.model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 3. Validation Metric (Levenshtein Distance)
    # --------------------------------------------------------------------------
    print("-" * 30)
    print("Loading best model for final evaluation...")
    if os.path.exists(Config.MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=trainer.device)
        )
    else:
        print("Warning: Model checkpoint not found. Using current weights.")

    trainer.model.eval()

    lev_distances = []
    lengths = []
    gap_positions = []

    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=-1)

    print("Computing Levenshtein distance on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            gap_idxs = batch["gap_idx"].to(trainer.device)
            target_ids = batch["target_id"].to(trainer.device)

            batch_size = input_ids.size(0)

            # Forward Pass
            loc_logits, id_logits = trainer.model(input_ids, attention_mask)

            # Truncate mask if sequence was truncated in model
            if attention_mask.size(1) > loc_logits.size(1):
                attention_mask = attention_mask[:, : loc_logits.size(1)]

            # Calculate Scores: P(Loc) * P(Word)
            p_loc = sigmoid(loc_logits).unsqueeze(-1)
            p_word = softmax(id_logits)
            scores = p_loc * p_word

            # Mask padding
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(scores)
            scores = scores * mask_expanded

            # Get Predictions
            scores_flat = scores.view(batch_size, -1)
            best_flat_indices = torch.argmax(scores_flat, dim=1)

            best_pos = best_flat_indices // trainer.vocab_size
            best_word_idx = best_flat_indices % trainer.vocab_size

            # Move to CPU for reconstruction
            input_ids_cpu = input_ids.cpu().numpy()
            best_pos_cpu = best_pos.cpu().numpy()
            best_word_idx_cpu = best_word_idx.cpu().numpy()
            target_ids_cpu = target_ids.cpu().numpy()
            gap_idxs_cpu = gap_idxs.cpu().numpy()

            # Iterate over batch to reconstruct sentences
            for i in range(batch_size):
                curr_ids = input_ids_cpu[i]

                # Determine real length (up to EOS)
                eos_mask = curr_ids == trainer.vocab.stoi.get(EOS_TOKEN, 3)
                if eos_mask.any():
                    real_len = np.argmax(eos_mask) + 1
                else:
                    real_len = len(curr_ids)

                valid_tokens = list(curr_ids[:real_len])

                # --- 1. Reconstruct Prediction ---
                pred_tokens = list(valid_tokens)
                p_pos = best_pos_cpu[i]
                p_word = best_word_idx_cpu[i]
                # Insert predicted word after predicted position
                # list.insert(idx, obj) inserts before idx. To insert after p_pos, we use p_pos + 1
                insert_idx_pred = min(p_pos + 1, len(pred_tokens))
                pred_tokens.insert(insert_idx_pred, p_word)

                # --- 2. Reconstruct Ground Truth ---
                gt_tokens = list(valid_tokens)
                g_pos = gap_idxs_cpu[i]
                g_word = target_ids_cpu[i]
                # Gap index in dataset is defined as 'remove_idx - 1'
                # So we insert after g_pos to restore original
                insert_idx_gt = min(g_pos + 1, len(gt_tokens))
                gt_tokens.insert(insert_idx_gt, g_word)

                # Decode to strings
                pred_str_tokens = trainer.vocab.decode(pred_tokens)
                gt_str_tokens = trainer.vocab.decode(gt_tokens)

                # Clean Special Tokens
                special = {SOS_TOKEN, EOS_TOKEN, Config.PAD_TOKEN, Config.MASK_TOKEN}
                pred_clean = [t for t in pred_str_tokens if t not in special]
                gt_clean = [t for t in gt_str_tokens if t not in special]

                pred_sent = " ".join(pred_clean)
                gt_sent = " ".join(gt_clean)

                # Compute Metric
                dist = edit_distance(pred_sent, gt_sent)
                lev_distances.append(dist)

                # Collect Features for Failure Analysis
                lengths.append(len(gt_sent))
                gap_positions.append(g_pos)

    final_metric = np.mean(lev_distances)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("-" * 30)
    print("Performing Failure Analysis...")

    if len(lev_distances) > 1:
        # Correlation: Error vs Sentence Length
        if np.std(lengths) > 0:
            corr_len, _ = pearsonr(lev_distances, lengths)
        else:
            corr_len = 0.0

        # Correlation: Error vs Gap Position
        if np.std(gap_positions) > 0:
            corr_pos, _ = pearsonr(lev_distances, gap_positions)
        else:
            corr_pos = 0.0

        print(f"Correlation (Error vs Sentence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Gap Position): {corr_pos:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    print("-" * 30)
    THRESHOLD = 7.70033

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric:.5f} < {THRESHOLD}. Generating submission...")
        trainer.generate_submission(test_loader)
    else:
        print(f"Metric {final_metric:.5f} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
