import os
import sys
import pandas as pd
import numpy as np
import torch
import nltk
from transformers import AutoTokenizer

# Import library components
from library.config import Config
from library.train import train_model
from library.predict import generate_submission
from library.model import DualHeadTransformer
from library.vocabulary import WordVocabulary
from library.utils import set_seed, load_checkpoint, insert_word_in_sentence


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Adjust Config for Fast Baseline (A100 GPU available)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200000  # Train on 200k samples
    Config.NUM_EPOCHS = 1  # 1 Epoch is sufficient for baseline
    Config.TRAIN_BATCH_SIZE = 64  # Efficient batch size for A100
    Config.VAL_BATCH_SIZE = 128  # Larger batch for inference

    set_seed(Config.SEED)

    print("=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Training Samples: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # ------------------------------------------------------------------
    # 2. Training
    # ------------------------------------------------------------------
    print("\n=== Starting Training ===")
    train_model(
        debug=Config.DEBUG,
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.TRAIN_BATCH_SIZE,
    )

    # ------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # ------------------------------------------------------------------
    print("\n=== Starting Validation Assessment ===")

    # Load validation data
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Use full validation set for metric calculation
    df_val_eval = df_val

    # Load resources
    vocab = WordVocabulary()
    vocab.load(Config.TARGET_VOCAB_PATH)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)

    model = DualHeadTransformer(vocab_size=len(vocab))
    model.to(Config.DEVICE)
    load_checkpoint(Config.MODEL_SAVE_PATH, model)
    model.eval()

    levenshtein_distances = []
    lengths = []

    print(f"Evaluating on {len(df_val_eval)} validation samples...")

    # Process in batches
    batch_size = Config.VAL_BATCH_SIZE

    for i in range(0, len(df_val_eval), batch_size):
        batch_df = df_val_eval.iloc[i : i + batch_size]
        original_sentences = batch_df["sentence"].tolist()

        corrupted_sentences = []
        ground_truth_sentences = []

        # Manually corrupt sentences to simulate test task
        for sent in original_sentences:
            words = sent.split()
            # Skip if too short to remove middle word
            if len(words) < 3:
                continue

            # Remove random word (not first or last)
            remove_idx = np.random.randint(1, len(words) - 1)

            prefix = words[:remove_idx]
            suffix = words[remove_idx + 1 :]
            corrupted = " ".join(prefix + suffix)

            corrupted_sentences.append(corrupted)
            ground_truth_sentences.append(sent)

        if not corrupted_sentences:
            continue

        # Tokenize
        encodings = tokenizer(
            corrupted_sentences,
            truncation=True,
            max_length=Config.MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(Config.DEVICE)
        attention_mask = encodings["attention_mask"].to(Config.DEVICE)

        # Inference
        with torch.no_grad():
            loc_logits, word_logits = model(input_ids, attention_mask)

            # Get Predictions
            pred_loc_indices = torch.argmax(loc_logits, dim=1).cpu().numpy()

            batch_indices = torch.arange(input_ids.size(0), device=Config.DEVICE)
            target_word_logits = word_logits[
                batch_indices, torch.argmax(loc_logits, dim=1), :
            ]
            pred_word_ids = torch.argmax(target_word_logits, dim=1).cpu().numpy()

        # Reconstruct and Calculate Metric
        for j in range(len(corrupted_sentences)):
            corrupted_text = corrupted_sentences[j]
            gt_text = ground_truth_sentences[j]

            # Decode word
            pred_word = vocab.id_to_token(int(pred_word_ids[j]))
            if pred_word == vocab.unk_token:
                pred_word = "the"  # Statistical fallback
            if pred_word == vocab.pad_token:
                pred_word = ""

            # Insert word
            reconstructed = insert_word_in_sentence(
                corrupted_text, pred_word, pred_loc_indices[j], tokenizer
            )

            # Compute Levenshtein Distance
            dist = nltk.edit_distance(reconstructed, gt_text)
            levenshtein_distances.append(dist)
            lengths.append(len(gt_text))

    # Report Metric
    mean_lev_dist = np.mean(levenshtein_distances)
    print(f"Final Validation Metric: {mean_lev_dist}")

    # Failure Analysis
    if len(lengths) > 1:
        correlation = np.corrcoef(lengths, levenshtein_distances)[0, 1]
        print(
            f"Correlation between Error (Levenshtein) and Sentence Length: {correlation:.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Submission
    # ------------------------------------------------------------------
    THRESHOLD = 7.8943

    if mean_lev_dist < THRESHOLD:
        print(
            f"\nMetric ({mean_lev_dist:.4f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # Run full inference on test set
        generate_submission(debug=False, batch_size=Config.VAL_BATCH_SIZE)
    else:
        print(
            f"\nMetric ({mean_lev_dist:.4f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
