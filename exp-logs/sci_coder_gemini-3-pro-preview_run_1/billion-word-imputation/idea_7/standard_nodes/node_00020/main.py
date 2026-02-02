import sys
import os
import torch
import numpy as np
import pandas as pd
import nltk
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders, load_or_build_artifacts
from library.engine import train_model
from library.inference import generate_submission, ConsistencyDecoder
from library.model import SyntaxAwareTransformer


def configure_baseline():
    """
    Overrides Config parameters for a fast baseline run within time limits.
    """
    # Enable Debug mode to subsample data
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20000  # 20k samples for fast training/val

    # Training settings
    Config.NUM_EPOCHS = 1  # Single pass is sufficient for baseline
    Config.BATCH_SIZE = 64  # Safe batch size for A100
    Config.VAL_BATCH_SIZE = 64

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(
        f"Configuration set: DEBUG={Config.DEBUG}, SIZE={Config.DEBUG_SIZE}, EPOCHS={Config.NUM_EPOCHS}"
    )


def calculate_levenshtein(s1, s2):
    """Computes Levenshtein distance between two strings."""
    return nltk.edit_distance(s1, s2)


def reconstruct_sentence(seq_ids, gap_idx, word_str, vocab):
    """
    Reconstructs a sentence from token IDs, inserting a word at a specific gap.

    Args:
        seq_ids (list/array): List of token IDs (including SOS, GAP, EOS).
        gap_idx (int): The index in seq_ids where the word should be inserted.
        word_str (str): The word to insert.
        vocab (Vocabulary): Vocab object for lookup.

    Returns:
        str: Reconstructed sentence.
    """
    tokens = []
    # seq_ids format: [SOS, GAP, w1, GAP, w2, ..., GAP, EOS, PAD...]

    for idx, token_id in enumerate(seq_ids):
        # Insert predicted/target word at the specific gap index
        if idx == gap_idx:
            tokens.append(word_str)

        # Stop at EOS
        if token_id == Config.EOS_IDX:
            break

        # Add normal words (skip special tokens)
        # Note: We skip GAP tokens here. The target gap is handled by the `if idx == gap_idx` check.
        if token_id not in [
            Config.SOS_IDX,
            Config.GAP_IDX,
            Config.PAD_IDX,
            Config.EOS_IDX,
        ]:
            tokens.append(vocab.lookup_token(token_id))

    return " ".join(tokens)


def validate_and_analyze(model, val_loader, vocab, pos_map, device):
    """
    Runs inference on validation set to compute Levenshtein metric and failure analysis.
    """
    print("Running validation inference for Levenshtein metric...")
    model.eval()
    decoder = ConsistencyDecoder(pos_map, device)

    levenshtein_scores = []
    sentence_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gap_mask = batch["gap_mask"].to(device)

            # Targets
            target_gap_idxs = batch["target_gap_idx"]
            target_word_ids = batch["target_word_id"]

            # Model Inference
            outputs = model(input_ids, attention_mask=attention_mask)

            # Decode Predictions
            pred_gap_idxs, pred_word_idxs = decoder.decode(
                outputs["loc_logits"],
                outputs["syntax_logits"],
                outputs["word_logits"],
                gap_mask,
            )

            # Move to CPU for string reconstruction
            input_ids_cpu = input_ids.cpu().numpy()
            pred_gap_idxs = pred_gap_idxs.cpu().numpy()
            pred_word_idxs = pred_word_idxs.cpu().numpy()
            target_gap_idxs = np.array(target_gap_idxs)
            target_word_ids = np.array(target_word_ids)

            # Iterate batch
            for i in range(len(input_ids_cpu)):
                # Get Data
                seq = input_ids_cpu[i]

                # Ground Truth Info
                gt_gap = target_gap_idxs[i]
                gt_word = vocab.lookup_token(target_word_ids[i])

                # Predicted Info
                p_gap = pred_gap_idxs[i]
                p_word = vocab.lookup_token(pred_word_idxs[i])

                # Reconstruct
                gt_sent = reconstruct_sentence(seq, gt_gap, gt_word, vocab)
                pred_sent = reconstruct_sentence(seq, p_gap, p_word, vocab)

                # Metric
                dist = calculate_levenshtein(gt_sent, pred_sent)
                levenshtein_scores.append(dist)
                sentence_lengths.append(len(gt_sent))

    # Compute Aggregate Metric
    final_metric = np.mean(levenshtein_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    if len(sentence_lengths) > 1:
        corr, _ = pearsonr(sentence_lengths, levenshtein_scores)
        print(f"Correlation (Sentence Length vs Error): {corr}")
    else:
        print("Insufficient data for correlation analysis.")

    return final_metric


def main():
    # 1. Setup
    configure_baseline()
    set_seed(Config.SEED)
    device = get_device()

    # 2. Data Loading & Training
    # get_dataloaders handles vocab building/loading internally
    print("--- Phase 1: Training ---")
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Train the model
    train_model(train_loader, val_loader)

    # 3. Validation & Analysis
    print("\n--- Phase 2: Validation & Analysis ---")
    # Load artifacts for decoding
    vocab, pos_map, _ = load_or_build_artifacts(load_cached_data=True)

    # Load best model
    model = SyntaxAwareTransformer().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print(
            "Warning: Checkpoint not found, using random weights (expect poor performance)."
        )

    # Run validation
    metric = validate_and_analyze(model, val_loader, vocab, pos_map, device)

    # 4. Submission Logic
    print("\n--- Phase 3: Submission ---")
    THRESHOLD = 7.214528751275944

    if metric < THRESHOLD:
        print(
            f"Metric {metric:.4f} is below threshold {THRESHOLD:.4f}. Generating submission..."
        )
        # Generate submission for test set
        generate_submission(batch_size=Config.BATCH_SIZE)
    else:
        print(
            f"Metric {metric:.4f} is above threshold {THRESHOLD:.4f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
