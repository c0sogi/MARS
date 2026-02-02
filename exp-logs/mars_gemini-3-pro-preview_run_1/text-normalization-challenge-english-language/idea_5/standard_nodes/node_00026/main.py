import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import from library
from library.config import Config
from library.data_utils import (
    build_vocabularies,
    build_knowledge_base,
    load_and_group_data,
    load_raw_data,
)
from library.datasets import (
    TaggerDataset,
    tagger_collate_fn,
    Seq2SeqDataset,
    seq2seq_collate_fn,
)
from library.models import BiLSTMTagger, Seq2SeqModel
from library.engine import Trainer
from library.inference import NormalizationPipeline


def run_pipeline():
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Prepare Resources (Vocabs & KB)
    # We load cached if available, otherwise build.
    # Note: In a fresh run, these might need building.
    print("Preparing vocabularies...")
    # We need raw train data to build vocab if not cached
    df_train_raw = load_raw_data("train")
    vocab_words, vocab_chars, vocab_classes = build_vocabularies(
        df_train_raw, load_cached_data=True
    )

    print("Preparing Knowledge Base...")
    knowledge_base = build_knowledge_base(df_train_raw, load_cached_data=True)

    # 3. Prepare DataLoaders
    print("Loading and grouping data...")
    df_train_grouped = load_and_group_data("train", load_cached_data=True)
    df_val_grouped = load_and_group_data("val", load_cached_data=True)

    # --- SUBSAMPLING FOR FAST BASELINE ---
    # Limit training data to ensure completion within 2 hours.
    # Tagger: 50k sentences
    # Seq2Seq: 50k changed tokens
    MAX_TAGGER_TRAIN_SAMPLES = 50000
    MAX_SEQ2SEQ_TRAIN_SAMPLES = 50000

    # Shuffle before slicing to get representative sample
    df_train_grouped = df_train_grouped.sample(
        frac=1, random_state=Config.SEED
    ).reset_index(drop=True)

    df_train_tagger = df_train_grouped.iloc[:MAX_TAGGER_TRAIN_SAMPLES]
    # We use full validation set for accurate metric calculation
    df_val_tagger = df_val_grouped

    print(f"Tagger Training Samples: {len(df_train_tagger)}")
    print(f"Tagger Validation Samples: {len(df_val_tagger)}")

    # Tagger Datasets
    train_tagger_ds = TaggerDataset(
        df_train_tagger, vocab_words, vocab_chars, vocab_classes, split="train"
    )
    val_tagger_ds = TaggerDataset(
        df_val_tagger, vocab_words, vocab_chars, vocab_classes, split="val"
    )

    train_tagger_loader = DataLoader(
        train_tagger_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=tagger_collate_fn,
    )
    val_tagger_loader = DataLoader(
        val_tagger_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=tagger_collate_fn,
    )

    # Seq2Seq Datasets
    # Need raw data for Seq2Seq (token level)
    # We can extract changed tokens from the raw train dataframe
    print("Preparing Seq2Seq data...")
    # Filter for changed tokens
    df_changed = df_train_raw[df_train_raw["before"] != df_train_raw["after"]]
    # Subsample
    if len(df_changed) > MAX_SEQ2SEQ_TRAIN_SAMPLES:
        df_changed = df_changed.sample(
            n=MAX_SEQ2SEQ_TRAIN_SAMPLES, random_state=Config.SEED
        )

    # For validation of Seq2Seq, we can use a subset of val raw data
    df_val_raw = load_raw_data("val")
    df_val_changed = df_val_raw[df_val_raw["before"] != df_val_raw["after"]]
    if len(df_val_changed) > 10000:
        df_val_changed = df_val_changed.sample(n=10000, random_state=Config.SEED)

    train_seq2seq_ds = Seq2SeqDataset(df_changed, vocab_chars, split="train")
    val_seq2seq_ds = Seq2SeqDataset(df_val_changed, vocab_chars, split="val")

    train_seq2seq_loader = DataLoader(
        train_seq2seq_ds,
        batch_size=Config.SEQ2SEQ_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=seq2seq_collate_fn,
    )
    val_seq2seq_loader = DataLoader(
        val_seq2seq_ds,
        batch_size=Config.SEQ2SEQ_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=seq2seq_collate_fn,
    )

    # 4. Initialize Models
    print("Initializing models...")
    tagger_model = BiLSTMTagger(
        vocab_size=len(vocab_words),
        num_classes=len(vocab_classes),
        char_vocab_size=len(vocab_chars),
    )

    seq2seq_model = Seq2SeqModel(num_chars=len(vocab_chars))

    trainer = Trainer(device=device)

    # 5. Train Tagger
    # Calculate class weights
    class_weights = trainer.get_class_weights(df_train_tagger, vocab_classes)

    # Override epochs for speed if necessary, but Config.NUM_EPOCHS=15 is okay with subsampled data
    # We will trust the Early Stopping in Trainer
    tagger_model = trainer.train_tagger(
        tagger_model, train_tagger_loader, val_tagger_loader, class_weights
    )

    # 6. Train Seq2Seq
    seq2seq_model = trainer.train_seq2seq(
        seq2seq_model, train_seq2seq_loader, val_seq2seq_loader
    )

    # 7. Full Validation & Metric Calculation
    print("\nPerforming Full Validation Pipeline...")
    tagger_model.eval()
    seq2seq_model.eval()

    total_tokens = 0
    correct_tokens = 0

    # Failure Analysis Data
    error_data = []

    # Identity classes
    identity_classes = {"PLAIN", "PUNCT", "VERBATIM"}

    # Special tokens for generation
    sos_idx = vocab_chars.stoi[Config.SOS_TOKEN]
    eos_idx = vocab_chars.stoi[Config.EOS_TOKEN]

    # Iterate over validation set (sentence level)
    # We iterate the loader to get batches, but we need the raw "before" and "after" which aren't in the batch dict fully.
    # However, df_val_grouped aligns with val_tagger_loader.
    # We can iterate them together using zip, or just iterate the dataframe and batch manually.
    # Using the loader is faster for GPU inference.

    # To map back to raw text, we can use the 'row_ids' in the batch and a lookup map.
    # Build ID lookup for Val
    print("Building Val ID lookup map...")
    # df_val_raw was loaded earlier
    val_id_to_before = dict(zip(df_val_raw["id"], df_val_raw["before"]))
    val_id_to_after = dict(zip(df_val_raw["id"], df_val_raw["after"]))

    with torch.no_grad():
        for batch in val_tagger_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            batch_row_ids = batch["row_ids"]

            # Tagger Prediction
            logits = tagger_model(word_ids, char_ids)
            pred_class_indices = torch.argmax(logits, dim=-1).cpu().numpy()

            # Collect OOV for batch processing
            oov_tokens = []
            oov_coords = []  # (batch_idx, token_idx)

            # Temporary storage for predictions
            # Shape: [batch_size, seq_len] (ragged)
            batch_predictions = [[None] * len(ids) for ids in batch_row_ids]

            for i, sent_ids in enumerate(batch_row_ids):
                seq_len = len(sent_ids)
                for j in range(seq_len):
                    row_id = sent_ids[j]
                    raw_token = val_id_to_before.get(row_id, "")

                    class_idx = pred_class_indices[i, j]
                    class_str = vocab_classes.itos.get(class_idx, Config.UNK_TOKEN)

                    # Logic
                    if (raw_token, class_str) in knowledge_base:
                        batch_predictions[i][j] = knowledge_base[(raw_token, class_str)]
                    elif class_str in identity_classes:
                        batch_predictions[i][j] = raw_token
                    else:
                        # Fallback
                        oov_tokens.append(raw_token)
                        oov_coords.append((i, j))

            # Process OOV
            if len(oov_tokens) > 0:
                src_indices_list = []
                for t in oov_tokens:
                    chars = list(t)
                    indices = vocab_chars.lookup_indices(
                        chars, unk_token=Config.UNK_TOKEN
                    )
                    src_indices_list.append(torch.tensor(indices, dtype=torch.long))

                src_tensor = pad_sequence(
                    src_indices_list, batch_first=True, padding_value=0
                ).to(device)

                gen_indices = seq2seq_model.generate(
                    src_tensor, sos_idx, eos_idx, max_len=Config.SEQ2SEQ_MAX_LEN
                )
                gen_indices = gen_indices.cpu().numpy()

                for k, indices in enumerate(gen_indices):
                    chars = []
                    for idx in indices:
                        if idx == eos_idx:
                            break
                        if idx == sos_idx:
                            continue
                        chars.append(vocab_chars.itos.get(idx, ""))
                    norm_text = "".join(chars)

                    # Update prediction
                    b_idx, t_idx = oov_coords[k]
                    batch_predictions[b_idx][t_idx] = norm_text

            # Compare with Ground Truth
            for i, sent_ids in enumerate(batch_row_ids):
                for j, row_id in enumerate(sent_ids):
                    pred = batch_predictions[i][j]
                    actual = val_id_to_after.get(row_id, "")

                    # Fallback for None (should not happen logic wise but safety)
                    if pred is None:
                        pred = val_id_to_before.get(row_id, "")

                    is_correct = pred == actual
                    total_tokens += 1
                    if is_correct:
                        correct_tokens += 1
                    else:
                        # Record error for failure analysis
                        raw_tok = val_id_to_before.get(row_id, "")
                        error_data.append({"len_before": len(raw_tok), "is_error": 1})

                    # Add correct samples to error data for correlation (is_error=0)
                    # To save memory, maybe sample correct ones?
                    # Or just add all. 1.7M is manageable for simple list of dicts.
                    if is_correct:
                        raw_tok = val_id_to_before.get(row_id, "")
                        error_data.append({"len_before": len(raw_tok), "is_error": 0})

    final_metric = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    if len(error_data) > 0:
        df_errors = pd.DataFrame(error_data)
        # Correlation between error and length
        corr = df_errors["is_error"].corr(df_errors["len_before"])
        print(f"Correlation between Error and Input Length: {corr:.4f}")

        # Additional stats
        error_rate = df_errors["is_error"].mean()
        print(f"Overall Error Rate: {error_rate:.4%}")
    else:
        print("No errors found or empty validation set.")

    # 9. Submission
    THRESHOLD = 0.9861543320467205
    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")
        # Use the Inference Pipeline class
        # It reloads models from disk, so we ensure they are saved.
        # Trainer saves best models to Config.TAGGER_MODEL_PATH etc.

        pipeline = NormalizationPipeline()
        pipeline.predict()
        print("Submission generated successfully.")
    else:
        print(
            f"\nMetric {final_metric} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run_pipeline()
