import os
import sys
import pandas as pd
import numpy as np
import torch
import nltk

# 1. Configuration Override for Fast Baseline
from library.config import Config

# Override Config to run within time limits
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 100000  # Process 100k samples for speed
Config.NUM_EPOCHS = 1  # Train for only 1 epoch
Config.BATCH_SIZE = 512  # Efficient batch size for A100
Config.WORKING_DIR = "./working/idea_4_run"  # Separate working dir to avoid conflicts
Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.npy")

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# 2. Import Library Modules
from library.utils import set_seed, setup_logger
from library.vocab import get_vocab
from library.dataset import get_dataloaders
from library.model import InterleavedTransformer
from library.engine import fit_model, generate_submission


def run():
    # Setup
    set_seed(42)
    logger = setup_logger("runfile")
    logger.info("Starting Fast Baseline Run")

    # 3. Data Preparation
    # We pass load_cached_data=False to force dataset creation with our DEBUG settings
    logger.info("Initializing Vocabulary...")
    vocab = get_vocab(load_cached_data=False)

    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab, load_cached_data=False
    )

    # 4. Model Initialization
    logger.info("Initializing Model...")
    model = InterleavedTransformer()

    # 5. Training
    logger.info("Starting Training...")
    fit_model(model, train_loader, val_loader)

    # 6. Validation (Levenshtein Metric)
    logger.info("Calculating Validation Metric (Levenshtein Distance)...")
    model.eval()
    device = Config.DEVICE
    model.to(device)

    distances = []
    lengths = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_word_id = batch["target_word_id"].to(device)
            target_gap_idx = batch["target_gap_idx"].to(device)

            # Forward pass
            loc_logits, id_logits = model(input_ids, token_type_ids, attention_mask)

            # Compute probabilities
            loc_probs = torch.sigmoid(loc_logits).squeeze(-1)
            id_probs = torch.softmax(id_logits, dim=-1)
            max_id_probs, best_word_indices = torch.max(id_probs, dim=-1)

            # Fusion Score
            combined_scores = loc_probs * max_id_probs

            # Mask invalid tokens
            mask = (token_type_ids == 1) & (attention_mask == 1)
            combined_scores[~mask] = -1.0

            # Get predictions
            best_gap_indices = torch.argmax(combined_scores, dim=1)
            final_word_ids = best_word_indices.gather(
                1, best_gap_indices.unsqueeze(1)
            ).squeeze(1)

            # Move to CPU for reconstruction
            batch_size = input_ids.size(0)
            input_ids_cpu = input_ids.cpu().numpy()
            target_gap_idx_cpu = target_gap_idx.cpu().numpy()
            target_word_id_cpu = target_word_id.cpu().numpy()
            best_gap_indices_cpu = best_gap_indices.cpu().numpy()
            final_word_ids_cpu = final_word_ids.cpu().numpy()

            for i in range(batch_size):
                # Skip invalid targets (should not happen in val loader usually, but good safety)
                if target_gap_idx_cpu[i] == -1:
                    continue

                # Extract base words from interleaved sequence
                # Sequence: [GAP, w0, GAP, w1, GAP, ...]
                seq = input_ids_cpu[i]
                current_words = []
                for idx, tid in enumerate(seq):
                    if tid == Config.PAD_IDX:
                        break
                    if idx % 2 == 1:  # Words are at odd indices
                        current_words.append(tid)

                # Reconstruct Ground Truth
                gt_words = list(current_words)
                # target_gap_idx corresponds to the index in the word list where the word was removed
                gt_insert_pos = target_gap_idx_cpu[i]
                gt_words.insert(gt_insert_pos, target_word_id_cpu[i])
                gt_sentence = vocab.decode(gt_words)

                # Reconstruct Prediction
                pred_words = list(current_words)
                # best_gap_indices is index in interleaved sequence.
                # Gap 0 -> Index 0. Gap 1 -> Index 2. Gap k -> Index 2*k.
                # Insert position in word list = gap_index // 2
                pred_insert_pos = best_gap_indices_cpu[i] // 2
                if pred_insert_pos > len(pred_words):
                    pred_insert_pos = len(pred_words)

                pred_words.insert(pred_insert_pos, final_word_ids_cpu[i])
                pred_sentence = vocab.decode(pred_words)

                # Calculate Distance
                dist = nltk.edit_distance(pred_sentence, gt_sentence)
                distances.append(dist)
                lengths.append(len(gt_sentence))

    final_metric = np.mean(distances)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    if len(distances) > 0:
        df_analysis = pd.DataFrame({"distance": distances, "length": lengths})
        correlation = df_analysis["distance"].corr(df_analysis["length"])
        print(f"Correlation between error magnitude and input features: {correlation}")
    else:
        print("Correlation between error magnitude and input features: NaN")

    # 8. Submission
    THRESHOLD = 7.70033
    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, test_loader, vocab)
    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
