import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import from provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.features import (
    RegexFeaturizer,
    BPETokenizerWrapper,
    GlobalPriorMap,
    prepare_bpe_training_data,
)
from library.data_loader import (
    build_vocabularies,
    get_tagger_loader,
    get_fallback_loader,
    KnowledgeBase,
    process_tagger_data,
)
from library.models import PentaHybridTagger, CharLSTMSeq2Seq
from library.training import TaggerTrainer, FallbackTrainer, compute_class_weights

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=== Starting Library Demonstration ===\n")

    # 1. SETUP & CONFIGURATION OVERRIDE
    # ---------------------------------------------------------
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config to use demo paths and small settings for speed
    Config.WORKING_DIR = DEMO_DIR
    Config.INPUT_DIR = "./working/demo_data"  # Virtual input dir
    os.makedirs(Config.INPUT_DIR, exist_ok=True)

    # Update Config Paths
    Config.TRAIN_DATA = os.path.join(Config.INPUT_DIR, "train_subset.csv")
    Config.VAL_DATA = os.path.join(Config.INPUT_DIR, "val_subset.csv")
    Config.TEST_DATA = os.path.join(Config.INPUT_DIR, "test_subset.csv")

    # Update Cache Paths based on new Working Dir
    Config.VOCAB_WORDS_PATH = os.path.join(DEMO_DIR, "vocab_words.json")
    Config.VOCAB_CHARS_PATH = os.path.join(DEMO_DIR, "vocab_chars.json")
    Config.VOCAB_CLASSES_PATH = os.path.join(DEMO_DIR, "vocab_classes.json")
    Config.BPE_MODEL_PREFIX = os.path.join(DEMO_DIR, "bpe_tokenizer")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(DEMO_DIR, "knowledge_base.parquet")
    Config.PRIORS_PATH = os.path.join(DEMO_DIR, "priors.parquet")
    Config.TAGGER_MODEL_PATH = os.path.join(DEMO_DIR, "tagger_best_model.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(DEMO_DIR, "seq2seq_best_model.pth")

    # Reduce Hyperparameters for Demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.VOCAB_SIZE_BPE = 500  # Small vocab for small data
    Config.MAX_SEQ_LEN = 32  # Shorter sequences
    Config.MAX_WORD_LEN = 15

    print("Configuration updated for demo execution.")

    # 2. DATA SUBSET CREATION
    # ---------------------------------------------------------
    print("\n--- Creating Data Subsets ---")
    # Load a small chunk of the real training data
    full_train = pd.read_csv("./metadata/train.csv", nrows=2000)

    # Create Train/Val Split from this chunk
    # Ensure we don't split sentences
    unique_sents = full_train["sentence_id"].unique()
    split_idx = int(len(unique_sents) * 0.8)
    train_sents = unique_sents[:split_idx]
    val_sents = unique_sents[split_idx:]

    df_train = full_train[full_train["sentence_id"].isin(train_sents)].copy()
    df_val = full_train[full_train["sentence_id"].isin(val_sents)].copy()

    # Create a dummy test set (just reuse val data structure but drop targets)
    df_test = df_val.copy()
    # Test usually doesn't have 'after' or 'class', but our loader expects 'before' and 'id'
    # The 'class' column is dropped inside process_tagger_data for test split usually,
    # but let's ensure the file on disk looks like a test file.
    df_test_save = df_test[["sentence_id", "token_id", "before", "id"]].copy()

    # Save to demo input directory
    df_train.to_csv(Config.TRAIN_DATA, index=False)
    df_val.to_csv(Config.VAL_DATA, index=False)
    df_test_save.to_csv(Config.TEST_DATA, index=False)

    print(f"Created train subset: {len(df_train)} rows")
    print(f"Created val subset: {len(df_val)} rows")
    print(f"Created test subset: {len(df_test_save)} rows")

    # 3. FEATURE ENGINEERING DEMONSTRATION
    # ---------------------------------------------------------
    print("\n--- Testing Feature Engineering ---")

    # A. Regex Featurizer
    regex = RegexFeaturizer()
    sample_token = "$1,200.50"
    feats = regex.get_features(sample_token)
    print(f"Regex features for '{sample_token}': {feats[:5]}... (Len: {len(feats)})")
    assert len(feats) == Config.REGEX_DIM, "Regex feature dimension mismatch"
    assert feats[0] == 0.0, "Should not be all digits"

    # B. BPE Tokenizer
    # Prepare corpus
    corpus_path = os.path.join(DEMO_DIR, "bpe_corpus.txt")
    prepare_bpe_training_data(Config.TRAIN_DATA, corpus_path)

    bpe = BPETokenizerWrapper()
    bpe.train(corpus_path)
    encoded = bpe.encode([sample_token])
    print(f"BPE Encoded '{sample_token}': {encoded}")
    assert len(encoded) == 1, "Encoding should return list of lists"

    # C. Global Priors
    priors = GlobalPriorMap()
    priors.build(Config.TRAIN_DATA, load_cached_data=False)
    prior_vec = priors.get_priors(["the"])  # 'the' is usually PLAIN
    print(f"Prior vector shape for 'the': {prior_vec.shape}")
    assert prior_vec.shape == (1, Config.PRIOR_DIM)

    # 4. DATA LOADING DEMONSTRATION
    # ---------------------------------------------------------
    print("\n--- Testing Data Loaders ---")

    # Build Vocabs
    word_vocab, char_vocab, class_vocab = build_vocabularies(
        Config.TRAIN_DATA, load_cached_data=False
    )
    print(f"Word Vocab Size: {len(word_vocab)}")
    print(f"Char Vocab Size: {len(char_vocab)}")
    print(f"Class Vocab Size: {len(class_vocab)}")

    # Tagger Loader
    print("Initializing Tagger Loader...")
    tagger_loader, vocabs = get_tagger_loader(
        "train", batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Fetch one batch
    batch = next(iter(tagger_loader))
    print("Tagger Batch Keys:", batch.keys())
    print(f"Word IDs Shape: {batch['word_ids'].shape}")
    print(f"Char IDs Shape: {batch['char_ids'].shape}")

    assert batch["word_ids"].shape[0] == Config.BATCH_SIZE
    assert batch["word_ids"].shape[1] == Config.MAX_SEQ_LEN
    assert batch["char_ids"].shape[2] == Config.MAX_WORD_LEN

    # Fallback Loader (Seq2Seq)
    print("Initializing Fallback Loader...")
    fallback_loader = get_fallback_loader(
        "train", batch_size=Config.BATCH_SIZE, load_cached_data=False
    )
    if fallback_loader and len(fallback_loader) > 0:
        fb_batch = next(iter(fallback_loader))
        print("Fallback Batch Keys:", fb_batch.keys())
        print(f"Src IDs Shape: {fb_batch['src_ids'].shape}")
        print(f"Tgt IDs Shape: {fb_batch['tgt_ids'].shape}")
    else:
        print(
            "No changed tokens found in subset for fallback loader (possible if subset is too small/clean)."
        )

    # 5. MODEL INSTANTIATION & FORWARD PASS
    # ---------------------------------------------------------
    print("\n--- Testing Models ---")

    # A. PentaHybridTagger
    tagger_model = PentaHybridTagger(
        num_words=len(word_vocab),
        num_chars=len(char_vocab),
        num_bpe=Config.VOCAB_SIZE_BPE,  # Size used in training BPE
        num_classes=len(class_vocab),
    ).to(device)

    # Forward pass
    with torch.no_grad():
        logits = tagger_model(
            batch["word_ids"].to(device),
            batch["char_ids"].to(device),
            batch["bpe_ids"].to(device),
            batch["regex_feats"].to(device),
            batch["prior_feats"].to(device),
        )
    print(f"Tagger Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, Config.MAX_SEQ_LEN, len(class_vocab))

    # B. CharLSTMSeq2Seq
    seq2seq_model = CharLSTMSeq2Seq(
        num_chars=len(char_vocab), num_classes=len(class_vocab)
    ).to(device)

    if fallback_loader and len(fallback_loader) > 0:
        src = fb_batch["src_ids"].to(device)
        tgt = fb_batch["tgt_ids"].to(device)
        cls = fb_batch["class_id"].to(device)

        # Forward pass (training mode)
        decoder_input = tgt[:, :-1]
        with torch.no_grad():
            outputs = seq2seq_model(src, decoder_input, cls)
        print(f"Seq2Seq Output Shape: {outputs.shape}")
        # Output is (Batch, Tgt_Seq_Len, Vocab)
        assert outputs.shape[0] == src.shape[0]
        assert outputs.shape[2] == len(char_vocab)

    # 6. TRAINING LOOP DEMONSTRATION
    # ---------------------------------------------------------
    print("\n--- Testing Training Loops ---")

    # Compute weights
    # Note: process_tagger_data saves labels to numpy, which compute_class_weights reads
    # We need to ensure the file exists. get_tagger_loader calls process_tagger_data.
    # The file name is hardcoded in `compute_class_weights` as "train_tagger_labels.npy"
    # But `process_tagger_data` uses prefix "{split}_tagger".
    # Let's check the file location.
    # In `process_tagger_data`: cache_prefix = .../{split_name}_tagger
    # In `compute_class_weights`: cache_path = .../train_tagger_labels.npy
    # So it matches if split is 'train'.

    class_weights = compute_class_weights(len(class_vocab))

    # Tagger Trainer
    tagger_trainer = TaggerTrainer(
        tagger_model, tagger_loader, tagger_loader, class_weights
    )
    tagger_trainer.train(num_epochs=1)

    # Fallback Trainer
    if fallback_loader and len(fallback_loader) > 0:
        fallback_trainer = FallbackTrainer(
            seq2seq_model, fallback_loader, fallback_loader
        )
        fallback_trainer.train(num_epochs=1)

    # 7. INFERENCE & UTILS DEMONSTRATION
    # ---------------------------------------------------------
    print("\n--- Testing Inference Utilities ---")

    # Knowledge Base
    kb = KnowledgeBase()
    kb.build(Config.TRAIN_DATA, load_cached_data=False)

    # Pick a token that exists in train
    sample_row = df_train.iloc[0]
    token, cls, after = sample_row["before"], sample_row["class"], sample_row["after"]

    lookup = kb.get(token, cls)
    print(f"KB Lookup for ('{token}', '{cls}'): {lookup}")
    # It might be None if the logic in KB build filtered it or if it's unique,
    # but for a small set it should likely be there.

    # Seq2Seq Generation
    if fallback_loader and len(fallback_loader) > 0:
        # Generate for one sample
        src_sample = fb_batch["src_ids"][0:1].to(device)
        cls_sample = fb_batch["class_id"][0:1].to(device)

        gen_out = seq2seq_model.generate(src_sample, cls_sample, max_len=20)
        print(f"Generated Sequence IDs: {gen_out.cpu().numpy()}")

        # Decode back to text
        gen_text = "".join([char_vocab.to_token(idx.item()) for idx in gen_out[0]])
        print(f"Generated Text (Raw): {gen_text}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
