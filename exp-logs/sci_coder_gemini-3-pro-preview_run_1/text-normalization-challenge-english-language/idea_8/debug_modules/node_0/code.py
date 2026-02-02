import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config, set_seed
from library.vocabulary import build_vocabularies
from library.data_manager import DataManager
from library.datasets import (
    TaggerDataset,
    TaggerCollator,
    Seq2SeqDataset,
    Seq2SeqCollator,
)
from library.models import BiLSTMTagger, TransformerSeq2Seq
from library.engine import Trainer
from library.inference import InferencePipeline


def run_demonstration():
    print("============================================================")
    print("   Text Normalization Pipeline Demonstration")
    print("============================================================")

    # 1. Setup & Configuration Override for Speed
    # ------------------------------------------------------------
    set_seed(42)

    # Override Config to use a small subset and lightweight models
    print("[Demo] Configuring for rapid execution (DEBUG mode)...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Process only 2000 sentences/tokens

    Config.TAGGER_EPOCHS = 1
    Config.SEQ2SEQ_EPOCHS = 1
    Config.BATCH_SIZE = 32

    # Reduce Tagger dimensions
    Config.TAGGER_EMBED_DIM = 64
    Config.TAGGER_HIDDEN_DIM = 64
    Config.TAGGER_CHAR_EMBED_DIM = 16
    Config.TAGGER_CNN_FILTERS = 16

    # Reduce Seq2Seq dimensions
    Config.SEQ2SEQ_D_MODEL = 64
    Config.SEQ2SEQ_NHEAD = 2
    Config.SEQ2SEQ_NUM_ENCODER_LAYERS = 2
    Config.SEQ2SEQ_NUM_DECODER_LAYERS = 2
    Config.SEQ2SEQ_DIM_FEEDFORWARD = 128

    # Ensure output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Build Vocabularies
    # ------------------------------------------------------------
    print("\n[Demo] Building Vocabularies...")
    # force rebuild to ensure it uses the debug subset if logic depended on it,
    # though vocab usually builds from full train. Here we just build once.
    # Note: The provided build_vocabularies loads full train.csv.
    # Ideally, for a fast demo, we'd want to avoid reading 7M rows, but we must use the provided function.
    # However, since we can't modify library files, we proceed.
    # The read_csv in build_vocabularies might take a few seconds.
    vocab_tokens, vocab_chars, vocab_classes = build_vocabularies(load_cached=False)

    print(f"   Token Vocab Size: {len(vocab_tokens)}")
    print(f"   Char Vocab Size:  {len(vocab_chars)}")
    print(f"   Class Vocab Size: {len(vocab_classes)}")

    # 3. Data Management & Preparation
    # ------------------------------------------------------------
    print("\n[Demo] Initializing DataManager...")
    dm = DataManager(vocab_tokens, vocab_chars, vocab_classes)

    # Build Knowledge Base
    kb = dm.get_knowledge_base(load_cached=False)
    print(f"   Knowledge Base Entries: {len(kb)}")

    # Compute Class Weights
    class_weights = dm.get_class_weights(load_cached=False)
    print(f"   Class Weights Computed: {class_weights.shape}")

    # 4. Tagger Training Workflow
    # ------------------------------------------------------------
    print("\n[Demo] --- Stage 1: Bi-LSTM Tagger Training ---")

    # Prepare Data
    df_train_grouped = dm.get_tagger_data("train", load_cached=False)
    df_val_grouped = dm.get_tagger_data("val", load_cached=False)

    print(f"   Tagger Train Sentences: {len(df_train_grouped)}")
    print(f"   Tagger Val Sentences:   {len(df_val_grouped)}")

    # Datasets & Loaders
    train_tagger_ds = TaggerDataset(
        df_train_grouped, vocab_tokens, vocab_chars, vocab_classes, split="train"
    )
    val_tagger_ds = TaggerDataset(
        df_val_grouped, vocab_tokens, vocab_chars, vocab_classes, split="val"
    )
    tagger_collator = TaggerCollator(vocab_tokens, vocab_chars)

    train_tagger_loader = torch.utils.data.DataLoader(
        train_tagger_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=tagger_collator,
        shuffle=True,
        num_workers=0,
    )
    val_tagger_loader = torch.utils.data.DataLoader(
        val_tagger_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=tagger_collator,
        shuffle=False,
        num_workers=0,
    )

    # Initialize Model
    tagger_model = BiLSTMTagger(
        vocab_size=len(vocab_tokens),
        char_vocab_size=len(vocab_chars),
        num_classes=len(vocab_classes),
        token_pad_idx=vocab_tokens.stoi[Config.PAD_TOKEN],
        char_pad_idx=vocab_chars.stoi[Config.PAD_TOKEN],
    )

    # Train
    trainer = Trainer(device=Config.DEVICE)
    best_tagger_acc = trainer.train_tagger(
        tagger_model, train_tagger_loader, val_tagger_loader, class_weights
    )

    # Verify Checkpoint
    if not os.path.exists(Config.TAGGER_MODEL_PATH):
        raise AssertionError("Tagger model checkpoint was not saved.")
    print(f"   Tagger Saved. Best Val Acc: {best_tagger_acc:.4f}")

    # 5. Seq2Seq Training Workflow
    # ------------------------------------------------------------
    print("\n[Demo] --- Stage 2: Transformer Seq2Seq Training ---")

    # Prepare Data
    df_train_seq = dm.get_seq2seq_data("train", load_cached=False)
    df_val_seq = dm.get_seq2seq_data("val", load_cached=False)

    # Handle case where debug subset might have no changes (unlikely but possible)
    if len(df_train_seq) == 0:
        print("   [Warning] No changed tokens in debug subset. Creating dummy data.")
        df_train_seq = pd.DataFrame(
            {
                "before": ["123", "abc"],
                "after": ["one two three", "a b c"],
                "class": ["CARDINAL", "LETTERS"],
            }
        )
        df_val_seq = df_train_seq.copy()

    print(f"   Seq2Seq Train Tokens: {len(df_train_seq)}")

    # Datasets & Loaders
    train_seq_ds = Seq2SeqDataset(df_train_seq, vocab_chars, vocab_classes)
    val_seq_ds = Seq2SeqDataset(df_val_seq, vocab_chars, vocab_classes)
    seq_collator = Seq2SeqCollator(vocab_chars)

    train_seq_loader = torch.utils.data.DataLoader(
        train_seq_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=seq_collator,
        shuffle=True,
        num_workers=0,
    )
    val_seq_loader = torch.utils.data.DataLoader(
        val_seq_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=seq_collator,
        shuffle=False,
        num_workers=0,
    )

    # Initialize Model
    seq2seq_model = TransformerSeq2Seq(
        char_vocab_size=len(vocab_chars),
        num_classes=len(vocab_classes),
        pad_idx=vocab_chars.stoi[Config.PAD_TOKEN],
        sos_idx=vocab_chars.stoi[Config.SOS_TOKEN],
        eos_idx=vocab_chars.stoi[Config.EOS_TOKEN],
    )

    # Train
    best_seq_loss = trainer.train_seq2seq(
        seq2seq_model, train_seq_loader, val_seq_loader
    )

    # Verify Checkpoint
    if not os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
        raise AssertionError("Seq2Seq model checkpoint was not saved.")
    print(f"   Seq2Seq Saved. Best Val Loss: {best_seq_loss:.4f}")

    # 6. Inference & Submission
    # ------------------------------------------------------------
    print("\n[Demo] --- Stage 3: Inference Pipeline ---")

    # Initialize Pipeline (Loads models from disk)
    pipeline = InferencePipeline()

    # Run Prediction on Test Set (Debug subset)
    pipeline.predict(batch_size=Config.BATCH_SIZE)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission Generated: {Config.SUBMISSION_PATH}")
    print(f"   Total Predictions: {len(df_sub)}")
    print("   Sample Predictions:")
    print(df_sub.head())

    # Final Validation
    assert "id" in df_sub.columns and "after" in df_sub.columns
    print("\n[Demo] Successfully completed all stages.")


if __name__ == "__main__":
    run_demonstration()
