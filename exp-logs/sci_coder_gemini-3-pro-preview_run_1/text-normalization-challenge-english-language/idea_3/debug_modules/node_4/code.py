import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil
import tqdm

# 1. Monkey-patch tqdm to suppress progress bars before importing library modules
# Cite debug_lesson_3: Do Not Replace sys.modules Entries with MagicMock in PyTorch 2.x
tqdm.tqdm = lambda x, *args, **kwargs: x

# 2. Import Library Modules
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import (
    get_tagger_dataloaders,
    get_seq2seq_dataloaders,
    get_knowledge_base,
    Vocabulary,
)
from library.models import BiLSTMTagger, Seq2SeqNormalizer
from library.trainer import train_tagger, train_seq2seq


def setup_demo_config():
    """
    Overrides Config parameters to ensure the demo runs quickly and
    uses a temporary working directory.
    """
    print("Setting up demo configuration...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 2000  # Small sample for fast execution

    # Set a specific working directory for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update paths dependent on WORKING_DIR
    Config.TRAIN_GROUPED_PATH = os.path.join(
        Config.WORKING_DIR, "train_grouped.parquet"
    )
    Config.VAL_GROUPED_PATH = os.path.join(Config.WORKING_DIR, "val_grouped.parquet")
    Config.TEST_GROUPED_PATH = os.path.join(Config.WORKING_DIR, "test_grouped.parquet")
    Config.VOCAB_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "vocab_tokens.parquet")
    Config.VOCAB_CHARS_PATH = os.path.join(Config.WORKING_DIR, "vocab_chars.parquet")
    Config.VOCAB_CLASSES_PATH = os.path.join(
        Config.WORKING_DIR, "vocab_classes.parquet"
    )
    Config.KNOWLEDGE_BASE_PATH = os.path.join(
        Config.WORKING_DIR, "knowledge_base.parquet"
    )
    Config.TAGGER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(
        Config.WORKING_DIR, "seq2seq_best_model.pth"
    )

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.TAGGER_EMBEDDING_DIM = 32
    Config.TAGGER_HIDDEN_DIM = 64
    Config.TAGGER_LAYERS = 1
    Config.SEQ2SEQ_EMBED_DIM = 32
    Config.SEQ2SEQ_HIDDEN_DIM = 64
    Config.MAX_VOCAB_SIZE = 1000

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")


def verify_data_pipeline():
    print("\n=== Verifying Data Pipeline ===")

    # Get Dataloaders
    # This triggers data loading, grouping, vocab creation, and caching
    train_loader, val_loader, vocab_tokens, vocab_chars, vocab_classes = (
        get_tagger_dataloaders(load_cached_data=False)
    )

    print(f"Vocab Tokens Size: {len(vocab_tokens)}")
    print(f"Vocab Chars Size: {len(vocab_chars)}")
    print(f"Vocab Classes Size: {len(vocab_classes)}")

    # Assertions
    if len(vocab_tokens) == 0:
        raise AssertionError("Token vocabulary is empty.")
    if len(vocab_classes) == 0:
        raise AssertionError("Class vocabulary is empty.")

    # Check a batch
    batch = next(iter(train_loader))
    word_ids, char_ids, class_ids = batch

    print(
        f"Batch Shapes -> Word: {word_ids.shape}, Char: {char_ids.shape}, Class: {class_ids.shape}"
    )

    # Assert shapes
    # word_ids: (batch, seq_len)
    if word_ids.dim() != 2:
        raise AssertionError(f"Expected word_ids to be 2D, got {word_ids.dim()}")
    # char_ids: (batch, seq_len, char_len)
    if char_ids.dim() != 3:
        raise AssertionError(f"Expected char_ids to be 3D, got {char_ids.dim()}")
    # class_ids: (batch, seq_len)
    if class_ids.dim() != 2:
        raise AssertionError(f"Expected class_ids to be 2D, got {class_ids.dim()}")

    print("Data Pipeline verification passed.")
    return vocab_tokens, vocab_chars, vocab_classes


def verify_model_architecture(vocab_tokens, vocab_chars, vocab_classes):
    print("\n=== Verifying Model Architecture ===")

    # 1. Tagger Model
    tagger = BiLSTMTagger(
        token_vocab_size=len(vocab_tokens),
        char_vocab_size=len(vocab_chars),
        num_classes=len(vocab_classes),
    ).to(Config.DEVICE)

    # Create dummy input
    batch_size = 4
    seq_len = 10
    char_len = Config.MAX_CHAR_LEN

    dummy_word_ids = torch.randint(0, len(vocab_tokens), (batch_size, seq_len)).to(
        Config.DEVICE
    )
    dummy_char_ids = torch.randint(
        0, len(vocab_chars), (batch_size, seq_len, char_len)
    ).to(Config.DEVICE)

    # Forward pass
    logits = tagger(dummy_word_ids, dummy_char_ids)
    print(f"Tagger Output Shape: {logits.shape}")

    if logits.shape != (batch_size, seq_len, len(vocab_classes)):
        raise AssertionError(
            f"Expected Tagger output shape {(batch_size, seq_len, len(vocab_classes))}, got {logits.shape}"
        )

    # 2. Seq2Seq Model
    seq2seq = Seq2SeqNormalizer(char_vocab_size=len(vocab_chars)).to(Config.DEVICE)

    tgt_len = 15
    dummy_src = torch.randint(0, len(vocab_chars), (batch_size, seq_len)).to(
        Config.DEVICE
    )
    dummy_tgt = torch.randint(0, len(vocab_chars), (batch_size, tgt_len)).to(
        Config.DEVICE
    )

    # Forward pass (Training mode)
    outputs = seq2seq(dummy_src, dummy_tgt)
    print(f"Seq2Seq Output Shape: {outputs.shape}")

    if outputs.shape != (batch_size, tgt_len, len(vocab_chars)):
        raise AssertionError(
            f"Expected Seq2Seq output shape {(batch_size, tgt_len, len(vocab_chars))}, got {outputs.shape}"
        )

    print("Model Architecture verification passed.")


def run_training_demo():
    print("\n=== Running Training Demo ===")

    # 1. Train Tagger
    # We rely on the library function but we've set Config.EPOCHS=1 and DEBUG=True
    print("Training Tagger...")
    train_tagger(load_cached_data=True)

    if not os.path.exists(Config.TAGGER_MODEL_PATH):
        raise AssertionError(
            f"Tagger model file not found at {Config.TAGGER_MODEL_PATH}"
        )
    print("Tagger training completed and model saved.")

    # 2. Train Seq2Seq
    print("Training Seq2Seq...")
    train_seq2seq(load_cached_data=True)

    if not os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
        raise AssertionError(
            f"Seq2Seq model file not found at {Config.SEQ2SEQ_MODEL_PATH}"
        )
    print("Seq2Seq training completed and model saved.")


def verify_inference(vocab_tokens, vocab_chars, vocab_classes):
    print("\n=== Verifying Inference Logic ===")

    # Load Models
    tagger = BiLSTMTagger(len(vocab_tokens), len(vocab_chars), len(vocab_classes)).to(
        Config.DEVICE
    )
    load_checkpoint(Config.TAGGER_MODEL_PATH, tagger, device=Config.DEVICE)
    tagger.eval()

    seq2seq = Seq2SeqNormalizer(len(vocab_chars)).to(Config.DEVICE)
    load_checkpoint(Config.SEQ2SEQ_MODEL_PATH, seq2seq, device=Config.DEVICE)
    seq2seq.eval()

    # Simulate input: "The year is 2023"
    tokens = ["The", "year", "is", "2023"]

    # Prepare Input
    word_ids = torch.tensor([vocab_tokens.lookup_indices(tokens)], dtype=torch.long).to(
        Config.DEVICE
    )

    char_ids_list = []
    for t in tokens:
        c_ids = vocab_chars.lookup_indices(list(t))
        c_ids = c_ids[: Config.MAX_CHAR_LEN]
        pad_len = Config.MAX_CHAR_LEN - len(c_ids)
        c_ids = c_ids + [vocab_chars.stoi["<pad>"]] * pad_len
        char_ids_list.append(c_ids)

    char_ids = torch.tensor([char_ids_list], dtype=torch.long).to(Config.DEVICE)

    # 1. Tagger Prediction
    with torch.no_grad():
        logits = tagger(word_ids, char_ids)
        preds = torch.argmax(logits, dim=-1)  # (1, seq_len)

    pred_classes = [vocab_classes.lookup_token(idx.item()) for idx in preds[0]]
    print(f"Input Tokens: {tokens}")
    print(f"Predicted Classes: {pred_classes}")

    if len(pred_classes) != len(tokens):
        raise AssertionError(
            "Number of predicted classes does not match number of tokens."
        )

    # 2. Seq2Seq Prediction (Simulating normalization for '2023')
    target_token = "2023"
    src_ids = vocab_chars.lookup_indices(list(target_token))
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(Config.DEVICE)

    sos_idx = vocab_chars.stoi["<sos>"]
    eos_idx = vocab_chars.stoi["<eos>"]

    with torch.no_grad():
        # predict returns (batch, seq_len)
        output_indices = seq2seq.predict(src_tensor, sos_idx, eos_idx)

    # Decode
    out_flat = output_indices[0].tolist()  # Take first item in batch
    normalized_chars = []
    for idx in out_flat:
        if idx == eos_idx:
            break
        normalized_chars.append(vocab_chars.lookup_token(idx))

    normalized_text = "".join(normalized_chars)
    print(f"Seq2Seq Normalization for '{target_token}': '{normalized_text}'")

    # Basic check - output should be string
    if not isinstance(normalized_text, str):
        raise AssertionError("Normalized text is not a string.")

    print("Inference logic verification passed.")


def verify_knowledge_base():
    print("\n=== Verifying Knowledge Base ===")

    kb = get_knowledge_base(load_cached_data=True)

    # Add a dummy entry to test retrieval logic
    test_key = ("test_token", "PLAIN")
    kb.lookup[test_key] = "test_token"

    result = kb.get("test_token", "PLAIN")
    print(f"KB Lookup ('test_token', 'PLAIN') -> {result}")

    if result != "test_token":
        raise AssertionError("Knowledge Base lookup failed.")

    print("Knowledge Base verification passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Verify Data
    vocab_tokens, vocab_chars, vocab_classes = verify_data_pipeline()

    # 3. Verify Model Architecture
    verify_model_architecture(vocab_tokens, vocab_chars, vocab_classes)

    # 4. Run Training
    run_training_demo()

    # 5. Verify Inference
    verify_inference(vocab_tokens, vocab_chars, vocab_classes)

    # 6. Verify Knowledge Base
    verify_knowledge_base()

    print("\nAll demonstrations and verifications completed successfully.")
