import os
import sys
import torch
import numpy as np
import pandas as pd
import nltk
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.tokenizer import get_tokenizer


def calculate_correlation(x, y):
    """Calculates Pearson correlation coefficient between two arrays."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline execution
    # We limit the data size and epochs to ensure runtime < 2 hours
    Config.DEBUG_SAMPLE_SIZE = 200000  # Use 200k samples for training/val
    Config.NUM_EPOCHS = 1  # Train for 1 epoch
    Config.BATCH_SIZE = 64  # Efficient batch size for A100
    Config.PATIENCE = 1  # Strict early stopping

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print(
        f"Configuration Configured: Samples={Config.DEBUG_SAMPLE_SIZE}, Epochs={Config.NUM_EPOCHS}"
    )

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    print("\n--- Starting Training Pipeline ---")
    trainer = Trainer(config=Config)
    trainer.fit()

    # ---------------------------------------------------------
    # 3. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n--- Starting Validation Evaluation ---")

    # We need to compute Levenshtein distance, which is not done by the Trainer by default.
    # We will run inference on the validation set manually.

    val_loader = trainer.val_loader
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = trainer.device

    # Load the best model weights saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model for evaluation.")

    model.eval()

    levenshtein_distances = []
    sentence_lengths = []

    print("Computing Levenshtein distance on validation set...")

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            word_target = batch["word_target"].to(device)
            gap_idx = batch["gap_idx"].to(device)

            # Forward pass
            loc_logits, word_logits = model(input_ids)

            # --- Predictions ---
            # 1. Location
            loc_probs = torch.sigmoid(loc_logits.squeeze(-1))
            # Mask padding
            padding_mask = input_ids != tokenizer.pad_token_id
            loc_probs = loc_probs * padding_mask.float()
            pred_loc_idxs = torch.argmax(loc_probs, dim=1)

            # 2. Word
            batch_indices = torch.arange(input_ids.size(0), device=device)
            best_word_logits = word_logits[batch_indices, pred_loc_idxs, :]
            pred_word_idxs = torch.argmax(best_word_logits, dim=1)

            # --- Reconstruction & Metric ---
            # Move to CPU
            input_ids_np = input_ids.cpu().numpy()
            word_target_np = word_target.cpu().numpy()
            gap_idx_np = gap_idx.cpu().numpy()
            pred_loc_idxs_np = pred_loc_idxs.cpu().numpy()
            pred_word_idxs_np = pred_word_idxs.cpu().numpy()

            for i in range(len(input_ids_np)):
                # Skip invalid targets (truncated/padding edge cases)
                if word_target_np[i] == -100:
                    continue

                # Decode base sentence tokens
                curr_input_ids = input_ids_np[i]
                tokens = []
                for tid in curr_input_ids:
                    if tid == tokenizer.pad_token_id:
                        break
                    tokens.append(tokenizer.idx2word.get(tid, tokenizer.unk_token))

                # Track length for failure analysis
                sentence_lengths.append(len(tokens))

                # Reconstruct Ground Truth Sentence
                gt_tokens = list(tokens)
                gt_word = tokenizer.idx2word.get(word_target_np[i], tokenizer.unk_token)
                # gap_idx is the index *before* the missing word. Insert at gap_idx + 1.
                gt_insert_pos = min(gap_idx_np[i] + 1, len(gt_tokens))
                gt_tokens.insert(gt_insert_pos, gt_word)
                gt_sentence = " ".join(gt_tokens)

                # Reconstruct Predicted Sentence
                pred_tokens = list(tokens)
                pred_word = tokenizer.idx2word.get(
                    pred_word_idxs_np[i], tokenizer.unk_token
                )
                pred_insert_pos = min(pred_loc_idxs_np[i] + 1, len(pred_tokens))
                pred_tokens.insert(pred_insert_pos, pred_word)
                pred_sentence = " ".join(pred_tokens)

                # Calculate Levenshtein Distance
                dist = nltk.edit_distance(gt_sentence, pred_sentence)
                levenshtein_distances.append(dist)

    # Compute Final Metric
    if len(levenshtein_distances) > 0:
        final_metric = np.mean(levenshtein_distances)
    else:
        final_metric = 0.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    if len(levenshtein_distances) > 1:
        correlation = calculate_correlation(levenshtein_distances, sentence_lengths)
        print(
            f"Correlation between Error (Levenshtein) and Sentence Length: {correlation:.4f}"
        )

        # Additional insight
        avg_dist = np.mean(levenshtein_distances)
        max_dist = np.max(levenshtein_distances)
        print(f"Error Stats - Avg: {avg_dist:.4f}, Max: {max_dist}")
    else:
        print("Insufficient data for failure analysis.")

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    print("\n--- Generating Submission ---")
    # The trainer class has a built-in method for this that handles the test set
    trainer.generate_submission()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
