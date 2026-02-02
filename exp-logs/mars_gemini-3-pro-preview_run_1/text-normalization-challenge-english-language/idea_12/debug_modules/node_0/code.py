import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.features import RegexFeatureExtractor
from library.data_utils import build_vocabularies, KnowledgeBase
from library.dataset import (
    TaggerDataset,
    FallbackDataset,
    collate_fn_tagger,
    collate_fn_fallback,
)
from library.models import MorphEnhancedTagger, Seq2SeqFallback
from library.engine import Engine, set_seed


def run_demo():
    print("=== Starting Text Normalization Pipeline Demo ===")

    # 1. Setup Directories and Data Subsets
    # We create subsets to ensure the pipeline runs in seconds rather than minutes/hours.
    demo_dir = "./working/demo_execution"
    data_dir = "./working/demo_data"
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("Creating data subsets for rapid demonstration...")
    # Read first 2000 rows of train (enough to get some sentence groups)
    df_train = pd.read_csv(
        "./metadata/train.csv", nrows=2000, dtype=str, keep_default_na=False
    )
    # Ensure we don't cut a sentence in half
    last_sent_id = df_train.iloc[-1]["sentence_id"]
    df_train = df_train[df_train["sentence_id"] != last_sent_id]
    df_train.to_csv(os.path.join(data_dir, "train_subset.csv"), index=False)

    # Read val
    df_val = pd.read_csv(
        "./metadata/val.csv", nrows=500, dtype=str, keep_default_na=False
    )
    df_val.to_csv(os.path.join(data_dir, "val_subset.csv"), index=False)

    # Read test
    df_test = pd.read_csv(
        "./metadata/test.csv", nrows=200, dtype=str, keep_default_na=False
    )
    df_test.to_csv(os.path.join(data_dir, "test_subset.csv"), index=False)

    # 2. Override Configuration
    print("Overriding Config for demo environment...")
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_DATA = os.path.join(data_dir, "train_subset.csv")
    Config.VAL_DATA = os.path.join(data_dir, "val_subset.csv")
    Config.TEST_DATA = os.path.join(data_dir, "test_subset.csv")

    # Update cache paths to point to the new working dir
    Config.VOCAB_WORDS_PATH = os.path.join(demo_dir, "vocab_words.json")
    Config.VOCAB_CHARS_PATH = os.path.join(demo_dir, "vocab_chars.json")
    Config.VOCAB_CLASSES_PATH = os.path.join(demo_dir, "vocab_classes.json")
    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_features.npy")
    Config.VAL_FEATURES_PATH = os.path.join(demo_dir, "val_features.npy")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "test_features.npy")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(demo_dir, "knowledge_base.parquet")
    Config.TAGGER_MODEL_PATH = os.path.join(demo_dir, "tagger_demo.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(demo_dir, "seq2seq_demo.pth")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Hyperparameters
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PATIENCE = 1

    set_seed(42)

    # 3. Feature Extraction Verification
    print("\n--- Verifying Feature Extraction ---")
    extractor = RegexFeatureExtractor()
    test_tokens = ["123", "hello", "$5.00"]
    features = extractor.extract(test_tokens)

    # Check shape: (3 tokens, N patterns)
    expected_feats = len(Config.REGEX_PATTERNS)
    assert features.shape == (
        3,
        expected_feats,
    ), f"Expected shape (3, {expected_feats}), got {features.shape}"

    # Check logic: "123" should match pattern 0 (^\d+$)
    # Note: Pattern order in Config.REGEX_PATTERNS is [r"^\d+$", ...]
    assert features[0, 0] == 1, "Token '123' failed to match regex pattern 0 (digits)"
    assert features[1, 0] == 0, "Token 'hello' incorrectly matched regex pattern 0"
    print("Feature extraction verified.")

    # 4. Vocabularies and Knowledge Base
    print("\n--- Building Vocabularies & KB ---")
    vocab_words, vocab_chars, vocab_classes = build_vocabularies(load_cached_data=False)
    print(
        f"Vocabs built. Words: {len(vocab_words)}, Chars: {len(vocab_chars)}, Classes: {len(vocab_classes)}"
    )

    kb = KnowledgeBase()
    kb.build(load_cached_data=False)
    print("Knowledge Base built.")

    # 5. Dataset Verification
    print("\n--- Verifying Datasets ---")
    # Tagger Dataset
    tagger_ds = TaggerDataset("train", load_cached_data=False, limit=10)
    assert len(tagger_ds) > 0, "TaggerDataset is empty"

    # Get one sample
    word_idx, char_indices, explicit_feat, class_idx = tagger_ds[0]
    # Check shapes
    # word_idx: (SeqLen,)
    # char_indices: (SeqLen, CharLen)
    # explicit_feat: (SeqLen, NumFeats)
    # class_idx: (SeqLen,)
    seq_len = word_idx.shape[0]
    assert char_indices.shape[0] == seq_len
    assert explicit_feat.shape[0] == seq_len
    assert class_idx.shape[0] == seq_len
    print(f"TaggerDataset sample verified. Seq Len: {seq_len}")

    # Fallback Dataset
    fallback_ds = FallbackDataset("train", load_cached_data=False, limit=10)
    if len(fallback_ds) > 0:
        src, tgt, cls = fallback_ds[0]
        assert src.dim() == 1
        assert tgt.dim() == 1
        assert cls.dim() == 0  # scalar
        print("FallbackDataset sample verified.")
    else:
        print(
            "FallbackDataset empty (no changed tokens in subset), skipping item check."
        )

    # 6. Model Verification
    print("\n--- Verifying Models ---")
    device = Config.DEVICE

    # Tagger Model
    tagger_model = MorphEnhancedTagger(
        vocab_size=len(vocab_words),
        num_classes=len(vocab_classes),
        num_chars=len(vocab_chars),
        num_explicit_features=expected_feats,
    ).to(device)

    # Create a batch
    loader = DataLoader(tagger_ds, batch_size=2, collate_fn=collate_fn_tagger)
    batch = next(iter(loader))
    word_idxs, char_idxs, explicit_feats, class_idxs = [b.to(device) for b in batch]

    # Forward pass
    logits = tagger_model(word_idxs, char_idxs, explicit_feats)
    # Shape: (Batch, Seq, NumClasses)
    assert logits.shape == (2, word_idxs.shape[1], len(vocab_classes))
    print("Tagger forward pass successful.")

    # Fallback Model
    fallback_model = Seq2SeqFallback(
        char_vocab_size=len(vocab_chars), num_classes=len(vocab_classes)
    ).to(device)

    if len(fallback_ds) > 0:
        fb_loader = DataLoader(
            fallback_ds, batch_size=2, collate_fn=collate_fn_fallback
        )
        batch_fb = next(iter(fb_loader))
        src, tgt, cls = [b.to(device) for b in batch_fb]

        # Forward pass (Training)
        output = fallback_model(src, tgt, cls)
        # Shape: (Batch, MaxLen, Vocab)
        assert output.shape[0] == 2
        assert output.shape[2] == len(vocab_chars)
        print("Fallback forward pass successful.")

        # Inference Generation
        gen_out = fallback_model.generate(src, cls, max_len=20)
        assert gen_out.shape == (2, 20)
        print("Fallback generation successful.")

    # 7. Engine Execution (Training Loop)
    print("\n--- Running Engine Training Loops ---")
    engine = Engine()

    # Train Tagger
    # Using limit=20 to ensure it finishes instantly
    engine.train_tagger(epochs=1, limit=20)
    assert os.path.exists(
        Config.TAGGER_MODEL_PATH
    ), "Tagger model checkpoint not found after training"

    # Train Fallback
    engine.train_fallback(epochs=1, limit=20)
    assert os.path.exists(
        Config.SEQ2SEQ_MODEL_PATH
    ), "Fallback model checkpoint not found after training"

    # 8. Submission Generation
    print("\n--- Generating Submission ---")
    engine.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"
    print(f"Submission generated with {len(df_sub)} rows.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
