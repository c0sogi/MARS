import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import (
    Vocabulary,
    KnowledgeBase,
    TaggerDataset,
    Seq2SeqDataset,
    SubmissionDataset,
    process_tagger_data,
    process_seq2seq_data,
    collate_tagger,
    collate_seq2seq,
    collate_submission,
)
from library.models import AttentionBiLSTMTagger, TransformerSeq2Seq
from library.engine import TaggerEngine, Seq2SeqEngine, generate_submission


def run_demo():
    # =========================================================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # =========================================================================
    print(">>> 1. Setting up Demo Configuration...")

    # Define a demo working directory to avoid overwriting main experiment files
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed and demo purposes
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.TAGGER_MODEL_PATH = os.path.join(DEMO_DIR, "tagger_demo.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(DEMO_DIR, "seq2seq_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to demo dir
    Config.VOCAB_TOKENS_PATH = os.path.join(DEMO_DIR, "vocab_tokens.parquet")
    Config.VOCAB_CHARS_PATH = os.path.join(DEMO_DIR, "vocab_chars.parquet")
    Config.VOCAB_CLASSES_PATH = os.path.join(DEMO_DIR, "vocab_classes.parquet")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(DEMO_DIR, "knowledge_base.parquet")

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script
    Config.MAX_VOCAB_SIZE = 2000

    # Reduce Model Size for speed
    Config.TAGGER_EMBEDDING_DIM = 64
    Config.TAGGER_HIDDEN_DIM = 64
    Config.TAGGER_RNN_LAYERS = 1
    Config.TAGGER_ATTENTION_HEADS = 2
    Config.TAGGER_CNN_FILTERS = 16

    Config.SEQ2SEQ_EMBEDDING_DIM = 64
    Config.SEQ2SEQ_HIDDEN_DIM = 64
    Config.SEQ2SEQ_LAYERS = 1
    Config.SEQ2SEQ_HEADS = 2

    Config.setup()
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. DATA SUBSET CREATION
    # =========================================================================
    print("\n>>> 2. Creating Data Subsets...")

    # Load a small chunk of training data
    # We need enough rows to get some variety in classes
    df_train_full = pd.read_csv(
        Config.TRAIN_FILE, nrows=5000, dtype=str, keep_default_na=False
    )

    # Ensure we have complete sentences
    last_sent_id = df_train_full.iloc[-1]["sentence_id"]
    df_train_subset = df_train_full[df_train_full["sentence_id"] != last_sent_id].copy()

    # Split into Train/Val for the demo
    unique_sents = df_train_subset["sentence_id"].unique()
    split_idx = int(len(unique_sents) * 0.8)
    train_sents = unique_sents[:split_idx]
    val_sents = unique_sents[split_idx:]

    df_train = df_train_subset[df_train_subset["sentence_id"].isin(train_sents)].copy()
    df_val = df_train_subset[df_train_subset["sentence_id"].isin(val_sents)].copy()

    print(f"    Train Subset: {len(df_train)} tokens")
    print(f"    Val Subset: {len(df_val)} tokens")

    # Load small chunk of test data
    df_test = pd.read_csv(
        Config.TEST_FILE, nrows=1000, dtype=str, keep_default_na=False
    )
    # Ensure complete sentences for test too
    last_test_sent_id = df_test.iloc[-1]["sentence_id"]
    df_test = df_test[df_test["sentence_id"] != last_test_sent_id].copy()
    print(f"    Test Subset: {len(df_test)} tokens")

    # =========================================================================
    # 3. VOCABULARY & KNOWLEDGE BASE
    # =========================================================================
    print("\n>>> 3. Building Vocabulary and Knowledge Base...")

    vocab = Vocabulary()
    vocab.build(df_train)
    vocab.save()

    kb = KnowledgeBase()
    kb.build(df_train)
    kb.save()

    # Validation
    assert len(vocab.token2id) > 2, "Vocabulary failed to build tokens"
    assert len(vocab.char2id) > 4, "Vocabulary failed to build chars"
    assert len(vocab.class2id) > 0, "Vocabulary failed to build classes"
    print("    Vocabulary and KB built successfully.")

    # =========================================================================
    # 4. DATA PROCESSING & LOADERS
    # =========================================================================
    print("\n>>> 4. Processing Data and Creating Loaders...")

    # --- Tagger Data ---
    df_train_grouped = process_tagger_data(df_train, vocab)
    df_val_grouped = process_tagger_data(df_val, vocab)

    train_dataset_tagger = TaggerDataset(df_train_grouped, vocab)
    val_dataset_tagger = TaggerDataset(df_val_grouped, vocab)

    train_loader_tagger = DataLoader(
        train_dataset_tagger,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_tagger,
    )
    val_loader_tagger = DataLoader(
        val_dataset_tagger,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_tagger,
    )

    # --- Seq2Seq Data ---
    # Ensure we have at least one changed token for demonstration
    # If the subset is too clean, we artificially create a change
    if (df_train["before"] == df_train["after"]).all():
        print("    (Injecting synthetic change for Seq2Seq demo)")
        new_row = df_train.iloc[0].copy()
        new_row["before"] = "123"
        new_row["after"] = "one two three"
        new_row["class"] = "CARDINAL"
        df_train = pd.concat([df_train, pd.DataFrame([new_row])], ignore_index=True)

    df_train_seq2seq = process_seq2seq_data(df_train, vocab)
    df_val_seq2seq = process_seq2seq_data(df_val, vocab)

    # Handle case where validation set might have no changes
    if len(df_val_seq2seq) == 0:
        df_val_seq2seq = df_train_seq2seq.iloc[:2].copy()

    train_dataset_seq2seq = Seq2SeqDataset(df_train_seq2seq, vocab)
    val_dataset_seq2seq = Seq2SeqDataset(df_val_seq2seq, vocab)

    train_loader_seq2seq = DataLoader(
        train_dataset_seq2seq,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_seq2seq,
    )
    val_loader_seq2seq = DataLoader(
        val_dataset_seq2seq,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_seq2seq,
    )

    # --- Test Data ---
    df_test_grouped = process_tagger_data(df_test, vocab)
    test_dataset = SubmissionDataset(df_test_grouped, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_submission,
    )

    print("    DataLoaders ready.")

    # =========================================================================
    # 5. MODEL INITIALIZATION
    # =========================================================================
    print("\n>>> 5. Initializing Models...")

    num_tokens = len(vocab.token2id)
    num_chars = len(vocab.char2id)
    num_classes = len(vocab.class2id)

    tagger_model = AttentionBiLSTMTagger(num_tokens, num_chars, num_classes)
    seq2seq_model = TransformerSeq2Seq(num_chars, num_classes)

    print(f"    Tagger initialized (Vocab: {num_tokens}, Classes: {num_classes})")
    print(f"    Seq2Seq initialized (Chars: {num_chars})")

    # =========================================================================
    # 6. TRAINING SIMULATION
    # =========================================================================
    print("\n>>> 6. Training Simulation...")

    # --- Train Tagger ---
    print("    Training Tagger...")
    tagger_engine = TaggerEngine(
        tagger_model, device, train_loader_tagger, val_loader_tagger
    )
    tagger_engine.fit(epochs=1)

    # Verify Tagger Checkpoint
    assert os.path.exists(
        Config.TAGGER_MODEL_PATH
    ), "Tagger model checkpoint not saved."

    # --- Train Seq2Seq ---
    print("    Training Seq2Seq...")
    seq2seq_engine = Seq2SeqEngine(
        seq2seq_model, device, train_loader_seq2seq, val_loader_seq2seq
    )
    seq2seq_engine.fit(epochs=1)

    # Verify Seq2Seq Checkpoint
    assert os.path.exists(
        Config.SEQ2SEQ_MODEL_PATH
    ), "Seq2Seq model checkpoint not saved."

    # =========================================================================
    # 7. INFERENCE & SUBMISSION GENERATION
    # =========================================================================
    print("\n>>> 7. Generating Submission...")

    # Reload best models (simulating inference phase)
    tagger_engine.load_checkpoint(Config.TAGGER_MODEL_PATH)
    seq2seq_engine.load_checkpoint(Config.SEQ2SEQ_MODEL_PATH)

    generate_submission(tagger_engine, seq2seq_engine, test_loader, kb, vocab)

    # =========================================================================
    # 8. VALIDATION OF RESULTS
    # =========================================================================
    print("\n>>> 8. Validating Results...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded with {len(df_sub)} rows.")

    # Format Check
    required_cols = ["id", "after"]
    if not all(col in df_sub.columns for col in required_cols):
        raise ValueError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Count Check
    # Note: df_test contains tokens. df_sub should have one row per token.
    # The grouping logic in process_tagger_data groups by sentence, but generate_submission flattens back to tokens.
    assert len(df_sub) == len(
        df_test
    ), f"Submission row count ({len(df_sub)}) does not match test data ({len(df_test)})"

    # Sample Check
    print("    Sample Predictions:")
    print(df_sub.head(3))

    print("\n>>> DEMO COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_demo()
