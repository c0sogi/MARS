import os
import sys
import torch
import pandas as pd
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seed, get_device
from library.preprocessing import (
    build_vocabularies,
    process_tagger_data,
    process_seq2seq_data,
    build_knowledge_base,
)
from library.data_loader import get_tagger_loaders, get_seq2seq_loaders
from library.model_tagger import MultiGranularityTagger
from library.model_seq2seq import TransformerFallback
from library.trainer import TaggerTrainer, Seq2SeqTrainer
from library.inference import NormalizationPipeline


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Setting up Configuration for Demo...")

    # Patch Config for speed and isolation
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 500  # Small subset for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.BPE_VOCAB_SIZE = 1000

    # Redirect paths to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update artifact paths
    Config.VOCAB_WORDS_PATH = os.path.join(Config.WORKING_DIR, "vocab_words.parquet")
    Config.VOCAB_CHARS_PATH = os.path.join(Config.WORKING_DIR, "vocab_chars.parquet")
    Config.VOCAB_CLASSES_PATH = os.path.join(
        Config.WORKING_DIR, "vocab_classes.parquet"
    )
    Config.VOCAB_BPE_MODEL_PATH = os.path.join(Config.WORKING_DIR, "bpe_tokenizer")

    Config.TRAIN_TAGGER_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "train_tagger_data.parquet"
    )
    Config.VAL_TAGGER_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "val_tagger_data.parquet"
    )
    Config.TEST_TAGGER_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "test_tagger_data.parquet"
    )

    Config.TRAIN_SEQ2SEQ_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "train_seq2seq_data.parquet"
    )
    Config.VAL_SEQ2SEQ_DATA_PATH = os.path.join(
        Config.WORKING_DIR, "val_seq2seq_data.parquet"
    )

    Config.KNOWLEDGE_BASE_PATH = os.path.join(
        Config.WORKING_DIR, "knowledge_base.parquet"
    )

    Config.TAGGER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "tagger_demo.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(Config.WORKING_DIR, "seq2seq_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    set_seed(Config.SEED)
    device = get_device()
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")

    # ==========================================
    # 2. Data Processing & Vocab Generation
    # ==========================================
    print("\n[2] Demonstrating Data Processing...")

    # Load raw train data (subset handled by logic or we just load full and it gets sliced later)
    # Since build_vocabularies uses the dataframe passed to it, we load a small chunk manually to speed up vocab build
    df_train_full = pd.read_csv(Config.TRAIN_FILE, nrows=2000)

    # FIX: Save subset to disk and update Config so internal loaders use the small dataset
    # This prevents the tokenizer from seeing the full dataset's character set (which exceeds the demo vocab size)
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    df_train_full.to_csv(subset_train_path, index=False)
    Config.TRAIN_FILE = subset_train_path
    print(f"    [Fix] Updated Config.TRAIN_FILE to {subset_train_path}")

    # Build Vocabs
    word_vocab, char_vocab, class_vocab, bpe_tokenizer = build_vocabularies(
        df_train_full, load_cached=False
    )

    print(f"    Word Vocab Size: {len(word_vocab)}")
    print(f"    Char Vocab Size: {len(char_vocab)}")
    print(f"    Class Vocab Size: {len(class_vocab)}")

    # Assertions
    assert len(word_vocab) > 0, "Word vocabulary is empty"
    assert len(class_vocab) > 0, "Class vocabulary is empty"
    assert os.path.exists(Config.VOCAB_WORDS_PATH), "Word vocab file not saved"

    # Build Knowledge Base
    kb = build_knowledge_base(df_train_full, load_cached=False)
    assert os.path.exists(Config.KNOWLEDGE_BASE_PATH), "Knowledge Base not saved"
    print("    Knowledge Base built and saved.")

    # ==========================================
    # 3. Data Loaders & Batch Verification
    # ==========================================
    print("\n[3] Verifying Data Loaders...")

    # Tagger Loaders
    # Note: get_tagger_loaders handles processing and caching internally.
    # We force load_cached=False to ensure it uses our new directory and small data
    train_loader, val_loader, test_loader, _, _, _, _ = get_tagger_loaders(
        debug=True, load_cached=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    word_ids = batch["word_ids"]
    label_ids = batch["label_ids"]

    print(f"    Tagger Batch - Word IDs Shape: {word_ids.shape}")
    print(f"    Tagger Batch - Label IDs Shape: {label_ids.shape}")

    # Assertions
    assert word_ids.dim() == 2, "Word IDs should be 2D (Batch, Seq)"
    assert word_ids.shape[0] <= Config.BATCH_SIZE, "Batch size mismatch"
    assert word_ids.shape == label_ids.shape, "Word and Label shapes mismatch"

    # Seq2Seq Loaders
    seq_train_loader, seq_val_loader, _, _ = get_seq2seq_loaders(
        debug=True, load_cached=False
    )

    if len(seq_train_loader) > 0:
        seq_batch = next(iter(seq_train_loader))
        src_ids = seq_batch["src_ids"]
        tgt_ids = seq_batch["tgt_ids"]
        print(f"    Seq2Seq Batch - Src IDs Shape: {src_ids.shape}")
        print(f"    Seq2Seq Batch - Tgt IDs Shape: {tgt_ids.shape}")
        assert src_ids.dim() == 2, "Src IDs should be 2D"
    else:
        print(
            "    Seq2Seq Train Loader is empty (no changes in debug subset). Skipping batch check."
        )

    # ==========================================
    # 4. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[4] Verifying Models...")

    # Tagger
    tagger_model = MultiGranularityTagger(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        bpe_vocab_size=Config.BPE_VOCAB_SIZE,
        class_vocab_size=len(class_vocab),
        pad_idx=Config.PAD_IDX,
    ).to(device)

    # Move batch to device
    word_ids = word_ids.to(device)
    char_ids = batch["char_ids"].to(device)
    bpe_ids = batch["bpe_ids"].to(device)
    mask = batch["mask"].to(device)

    # Forward
    tagger_logits = tagger_model(word_ids, char_ids, bpe_ids, mask)
    print(f"    Tagger Logits Shape: {tagger_logits.shape}")

    assert tagger_logits.shape[0] == word_ids.shape[0], "Batch dimension mismatch"
    assert tagger_logits.shape[1] == word_ids.shape[1], "Sequence dimension mismatch"
    assert tagger_logits.shape[2] == len(class_vocab), "Class dimension mismatch"

    # Seq2Seq
    seq2seq_model = TransformerFallback(
        char_vocab_size=len(char_vocab),
        class_vocab_size=len(class_vocab),
        pad_idx=Config.PAD_IDX,
    ).to(device)

    if len(seq_train_loader) > 0:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        class_id = seq_batch["class_id"].to(device)

        # Teacher forcing input (remove last token)
        tgt_input = tgt_ids[:, :-1]

        seq_logits = seq2seq_model(src_ids, tgt_input, class_id)
        print(f"    Seq2Seq Logits Shape: {seq_logits.shape}")

        assert seq_logits.shape[0] == src_ids.shape[0], "Batch dimension mismatch"
        assert (
            seq_logits.shape[1] == tgt_input.shape[1]
        ), "Target sequence length mismatch"
        assert seq_logits.shape[2] == len(char_vocab), "Vocab dimension mismatch"

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Running Training Loops (Fast Demo)...")

    # Tagger Training
    print("    Training Tagger...")
    tagger_trainer = TaggerTrainer(debug=True)
    tagger_trainer.train()
    assert os.path.exists(Config.TAGGER_MODEL_PATH), "Tagger model checkpoint not found"

    # Seq2Seq Training
    print("    Training Seq2Seq...")
    seq2seq_trainer = Seq2SeqTrainer(debug=True)
    seq2seq_trainer.train()
    # Checkpoint might not exist if validation loss didn't improve or if skipped,
    # but with 1 epoch and init weights, it usually saves once.
    if not seq2seq_trainer.skip_training:
        assert os.path.exists(
            Config.SEQ2SEQ_MODEL_PATH
        ), "Seq2Seq model checkpoint not found"

    # ==========================================
    # 6. Inference Pipeline
    # ==========================================
    print("\n[6] Running Inference Pipeline...")

    pipeline = NormalizationPipeline(debug=True)
    pipeline.run_inference()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(df_sub)}")
    print(f"    Submission Columns: {list(df_sub.columns)}")

    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
