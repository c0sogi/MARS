import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import ProjectConfig, TrainingConfig, DataConfig, set_seed
from library.data_utils import (
    build_vocabularies,
    build_knowledge_base,
    load_dataset_raw,
    TaggerDataset,
    Seq2SeqDataset,
    collate_fn_tagger,
    collate_fn_seq2seq,
)
from library.models import MultiGranularityTagger, Seq2SeqNormalizer
from library.engine import Trainer
from library.inference import CascadePredictor, generate_submission_file


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides for Fast Baseline
    # ---------------------------------------------------------
    print("Initializing Fast Baseline Run...")
    set_seed(TrainingConfig.SEED)

    # Override Config for speed
    # We use 1 epoch and limited data to ensure completion within 2 hours
    TrainingConfig.TAGGER_EPOCHS = 1
    TrainingConfig.SEQ_EPOCHS = 1

    # Subsampling sizes
    TAGGER_TRAIN_SIZE = 200000
    SEQ_TRAIN_SIZE = 100000
    VAL_SUBSET_SIZE = 10000  # For training loop validation only

    device = torch.device(TrainingConfig.DEVICE)
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Resource Building (Vocabs & KB)
    # ---------------------------------------------------------
    print("\n--- Building Resources ---")
    # Build vocabs on full data (fast enough and necessary for dimensions)
    vocab_words, vocab_chars, vocab_classes = build_vocabularies(load_cached_data=True)

    # Build Knowledge Base on full data (critical for high accuracy)
    # This might take ~1-2 mins but is worth it as it handles >90% of cases
    kb_dict = build_knowledge_base(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Data Loading & Subsampling
    # ---------------------------------------------------------
    print("\n--- Loading and Subsampling Data ---")
    df_train_full = load_dataset_raw("train")
    df_val_full = load_dataset_raw("val")

    # Subsample for Tagger (General distribution)
    df_train_tagger = df_train_full.sample(
        n=min(TAGGER_TRAIN_SIZE, len(df_train_full)), random_state=TrainingConfig.SEED
    )

    # Subsample for Seq2Seq (Focus on changed tokens)
    # Filter first, then sample
    df_changed = df_train_full[df_train_full["before"] != df_train_full["after"]]
    df_train_seq = df_changed.sample(
        n=min(SEQ_TRAIN_SIZE, len(df_changed)), random_state=TrainingConfig.SEED
    )

    # Subset val for training loop speed
    df_val_subset = df_val_full.sample(
        n=min(VAL_SUBSET_SIZE, len(df_val_full)), random_state=TrainingConfig.SEED
    )

    print(f"Tagger Train Size: {len(df_train_tagger)}")
    print(f"Seq2Seq Train Size: {len(df_train_seq)}")
    print(f"Validation Full Size: {len(df_val_full)}")

    # Create Datasets
    train_tagger_ds = TaggerDataset(
        df_train_tagger, vocab_words, vocab_chars, vocab_classes
    )
    val_tagger_ds = TaggerDataset(
        df_val_subset, vocab_words, vocab_chars, vocab_classes
    )

    train_seq_ds = Seq2SeqDataset(df_train_seq, vocab_chars, vocab_classes)
    val_seq_ds = Seq2SeqDataset(
        df_val_subset, vocab_chars, vocab_classes
    )  # Seq2SeqDataset internally filters for changes

    # Create Loaders
    train_tagger_loader = DataLoader(
        train_tagger_ds,
        batch_size=TrainingConfig.TAGGER_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )
    val_tagger_loader = DataLoader(
        val_tagger_ds,
        batch_size=TrainingConfig.TAGGER_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )

    train_seq_loader = DataLoader(
        train_seq_ds,
        batch_size=TrainingConfig.SEQ_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_seq2seq,
        num_workers=DataConfig.NUM_WORKERS,
    )
    val_seq_loader = DataLoader(
        val_seq_ds,
        batch_size=TrainingConfig.SEQ_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_seq2seq,
        num_workers=DataConfig.NUM_WORKERS,
    )

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    trainer = Trainer(device=device)

    # --- Train Tagger ---
    print("\n--- Training Tagger ---")
    tagger_model = MultiGranularityTagger(
        len(vocab_words), len(vocab_chars), len(vocab_classes)
    )
    trainer.train_tagger(
        tagger_model,
        train_tagger_loader,
        val_tagger_loader,
        epochs=TrainingConfig.TAGGER_EPOCHS,
    )

    # Free memory
    del tagger_model, train_tagger_loader, val_tagger_loader, df_train_tagger
    torch.cuda.empty_cache()

    # --- Train Seq2Seq ---
    print("\n--- Training Seq2Seq ---")
    seq_model = Seq2SeqNormalizer(len(vocab_chars), len(vocab_classes))
    trainer.train_seq2seq(
        seq_model, train_seq_loader, val_seq_loader, epochs=TrainingConfig.SEQ_EPOCHS
    )

    # Free memory
    del seq_model, train_seq_loader, val_seq_loader, df_train_seq
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 5. Full Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n--- Performing Full Validation ---")

    # Initialize Predictor (loads best models from disk)
    predictor = CascadePredictor(device=device)

    # Create loader for full validation set
    # We use TaggerDataset structure for the predictor input
    val_full_ds = TaggerDataset(
        df_val_full, vocab_words, vocab_chars, vocab_classes, is_test=True
    )
    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )

    correct_count = 0
    total_count = 0
    failures = []

    print(f"Validating on {len(df_val_full)} samples...")

    # Validation Loop
    for batch in val_full_loader:
        # Get ground truth
        # Note: TaggerDataset in test mode doesn't return classes, but we have raw_texts and ids.
        # We need to map ids back to the dataframe to get 'after' truth.
        # However, df_val_full is aligned with the loader (shuffle=False).
        # We can just iterate the dataframe in chunks or index it.
        # Simpler: The batch contains 'ids'. We can look up ground truth if we indexed df_val_full.
        pass

    # Efficient Validation:
    # Instead of complex lookups, we'll iterate the loader and the dataframe simultaneously
    # since shuffle=False and order is preserved.

    # Extract ground truth 'after' column
    ground_truth_after = df_val_full["after"].astype(str).values
    input_before = df_val_full["before"].astype(str).values
    input_ids = df_val_full["id"].values

    batch_start = 0

    for batch in val_full_loader:
        batch_size = len(batch["raw_texts"])

        # Predict
        predictions = predictor.predict_batch(batch)

        # Extract predicted strings
        pred_strs = [p["after"] for p in predictions]

        # Get corresponding ground truth
        truth_strs = ground_truth_after[batch_start : batch_start + batch_size]
        before_strs = input_before[batch_start : batch_start + batch_size]

        # Compare
        for i in range(batch_size):
            p_str = pred_strs[i]
            t_str = truth_strs[i]

            if p_str == t_str:
                correct_count += 1
            else:
                # Record failure for analysis
                failures.append(
                    {
                        "before": before_strs[i],
                        "expected": t_str,
                        "predicted": p_str,
                        "len_before": len(before_strs[i]),
                    }
                )

        batch_start += batch_size
        total_count += batch_size

        if total_count % 100000 == 0:
            print(f"Validated {total_count} samples...")

    accuracy = correct_count / total_count
    print(f"Final Validation Metric: {accuracy}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    if len(failures) > 0:
        df_failures = pd.DataFrame(failures)
        print(f"Total Failures: {len(df_failures)}")

        # Correlation between error and length
        # We construct a vector for all validation samples: 1 if error, 0 if correct
        # And a vector for lengths
        # This is memory intensive for 1.7M rows, so we approximate using the failure stats vs global stats

        avg_len_global = df_val_full["before"].str.len().mean()
        avg_len_failures = df_failures["len_before"].mean()

        print(f"Average Token Length (Global): {avg_len_global:.4f}")
        print(f"Average Token Length (Failures): {avg_len_failures:.4f}")

        # Calculate Point Biserial Correlation approx
        # r_pb = (M1 - M0) / sn * sqrt(n1 * n0 / n^2)
        # M1 = mean length of errors, M0 = mean length of correct
        # We can derive M0 from Global Mean and M1
        n = total_count
        n1 = len(failures)
        n0 = n - n1

        if n0 > 0:
            sum_len_global = df_val_full["before"].str.len().sum()
            sum_len_failures = df_failures["len_before"].sum()
            mean_len_correct = (sum_len_global - sum_len_failures) / n0

            # std dev of length (global)
            std_len_global = df_val_full["before"].str.len().std()

            if std_len_global > 0:
                corr = (
                    (avg_len_failures - mean_len_correct)
                    / std_len_global
                    * np.sqrt((n1 * n0) / (n**2))
                )
                print(f"Correlation (Error vs Input Length): {corr:.6f}")
            else:
                print("Correlation undefined (std dev is 0)")

        print("\nSample Failures:")
        print(df_failures[["before", "expected", "predicted"]].head(5).to_string())
    else:
        print("No failures detected!")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.9861543320467205

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy} > {THRESHOLD}. Generating submission..."
        )
        generate_submission_file(batch_size=512)
    else:
        print(f"\nValidation accuracy {accuracy} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
