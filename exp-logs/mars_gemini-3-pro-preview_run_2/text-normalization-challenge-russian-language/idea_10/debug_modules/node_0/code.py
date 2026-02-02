import os
import sys
import pandas as pd
import torch
import shutil
import numpy as np

# Import provided library modules
from library import config
from library import utils
from library import hfbb
from library import tokenizers
from library import dataset
from library import model as model_lib
from library import trainer
from library import inference


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo by overriding config paths
    and creating small sample datasets.
    """
    print("=== Setting up Demo Environment ===")

    # 1. Define Demo Paths in ./working
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # 2. Create Sample Data
    # We create a small synthetic dataset to ensure we have specific cases
    # (Plain, Numbers, Dates) to test the pipeline effectively.

    # Sample Training Data
    train_data = {
        "sentence_id": [0, 0, 0, 1, 1, 1, 2, 2],
        "token_id": [0, 1, 2, 0, 1, 2, 0, 1],
        "class": [
            "PLAIN",
            "PLAIN",
            "PUNCT",
            "PLAIN",
            "CARDINAL",
            "PLAIN",
            "DATE",
            "PLAIN",
        ],
        "before": ["Hello", "world", ".", "I", "123", "apples", "2012", "year"],
        "after": [
            "Hello",
            "world",
            ".",
            "I",
            "one hundred twenty three",
            "apples",
            "twenty twelve",
            "year",
        ],
    }
    df_train = pd.DataFrame(train_data)
    demo_train_path = os.path.join(demo_dir, "mini_train.csv")
    df_train.to_csv(demo_train_path, index=False)

    # Sample Validation Data
    val_data = {
        "sentence_id": [10, 10, 10],
        "token_id": [0, 1, 2],
        "class": ["PLAIN", "CARDINAL", "PLAIN"],
        "before": ["Number", "42", "is"],
        "after": ["Number", "forty two", "is"],
    }
    df_val = pd.DataFrame(val_data)
    demo_val_path = os.path.join(demo_dir, "mini_val.csv")
    df_val.to_csv(demo_val_path, index=False)

    # Sample Test Data
    test_data = {
        "sentence_id": [100] * 3,
        "token_id": [0, 1, 2],
        "before": ["Test", "99", "end"],
    }
    df_test = pd.DataFrame(test_data)
    demo_test_path = os.path.join(demo_dir, "mini_test.csv")
    df_test.to_csv(demo_test_path, index=False)

    # 3. Monkey-patch config module
    # This redirects the library to use our demo files and folders
    print("Overriding config parameters...")

    config.TRAIN_FILE = demo_train_path
    config.VAL_FILE = demo_val_path
    config.TEST_FILE = demo_test_path

    config.WORKING_DIR = demo_dir
    config.HFBB_CACHE_DIR = os.path.join(demo_dir, "hfbb_cache")
    config.TRANSFORMER_CACHE_DIR = os.path.join(demo_dir, "data_cache")
    config.TOKENIZER_DIR = os.path.join(demo_dir, "tokenizers")
    config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Ensure dirs exist
    for d in [
        config.HFBB_CACHE_DIR,
        config.TRANSFORMER_CACHE_DIR,
        config.TOKENIZER_DIR,
        config.CHECKPOINT_DIR,
        config.SUBMISSION_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # Update derived file paths
    config.UNIGRAM_PATH = os.path.join(config.HFBB_CACHE_DIR, "unigram.parquet")
    config.BIGRAM_PREV_PATH = os.path.join(config.HFBB_CACHE_DIR, "bigram_prev.parquet")
    config.BIGRAM_NEXT_PATH = os.path.join(config.HFBB_CACHE_DIR, "bigram_next.parquet")
    config.TRIGRAM_PATH = os.path.join(config.HFBB_CACHE_DIR, "trigram.parquet")

    config.PROCESSED_TRAIN_PATH = os.path.join(
        config.TRANSFORMER_CACHE_DIR, "train_proc.parquet"
    )
    config.PROCESSED_VAL_PATH = os.path.join(
        config.TRANSFORMER_CACHE_DIR, "val_proc.parquet"
    )

    config.CHAR_VOCAB_PATH = os.path.join(config.TOKENIZER_DIR, "char_vocab.json")
    config.BPE_MODEL_PREFIX = os.path.join(config.TOKENIZER_DIR, "bpe_demo")
    config.BPE_MODEL_PATH = config.BPE_MODEL_PREFIX + ".model"

    config.BEST_MODEL_PATH = os.path.join(
        config.CHECKPOINT_DIR, "transformer_demo_best.pth"
    )
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Reduce Hyperparameters for Speed
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4
    config.VOCAB_SIZE = 100  # Small vocab for small data
    config.DEBUG = True
    config.MAX_TRAIN_SAMPLES = 50

    print("Environment setup complete.")


def demonstrate_utils():
    print("\n=== Demonstrating Utils ===")
    utils.set_seed(42)

    # Test is_semiotic
    # "123" contains digits -> True
    assert utils.is_semiotic("123") == True, "123 should be semiotic"
    # "Hello" is plain text -> False
    assert utils.is_semiotic("Hello") == False, "Hello should not be semiotic"
    # "2012" with explicit class DATE -> True
    assert (
        utils.is_semiotic("2012", token_class="DATE") == True
    ), "DATE class should be semiotic"

    print("Utils verification passed.")


def demonstrate_hfbb():
    print("\n=== Demonstrating HFBB (Stats Engine) ===")

    # Initialize engine
    engine = hfbb.HFBB()

    # Build stats from our mini_train.csv
    # load_cached_data=False forces re-computation
    engine.build_stats(load_cached_data=False)

    # Verify internal dictionaries are populated
    # We expect unigrams for "Hello", "world", "123", etc.
    assert "123" in engine.unigram, "HFBB should contain unigram stats for '123'"

    # Query the engine
    # Context: I 123 apples
    # Prev="I", Curr="123", Next="apples"
    pred, conf, level = engine.query("I", "123", "apples")

    print(f"Query '123' -> Pred: '{pred}', Conf: {conf}, Level: {level}")

    # In our training data: "123" -> "one hundred twenty three"
    # It might be found via Trigram, Bigram, or Unigram depending on context overlap
    assert pred == "one hundred twenty three", f"Expected normalization, got {pred}"

    print("HFBB verification passed.")
    return engine


def demonstrate_tokenizers():
    print("\n=== Demonstrating Tokenizers ===")

    # Build tokenizers from scratch using mini_train.csv
    char_tok, bpe_tok = tokenizers.build_tokenizers(load_cached_data=False)

    # Verify Char Tokenizer
    text = "123"
    ids = char_tok.encode(text)
    decoded = char_tok.decode(ids)
    print(f"Char Tokenizer: '{text}' -> {ids} -> '{decoded}'")
    assert decoded == text, "Char tokenizer decode mismatch"

    # Verify BPE Tokenizer
    # Note: BPE training on tiny data is unstable/weird, but should function
    tgt_text = "one hundred"
    ids_bpe = bpe_tok.encode(tgt_text)
    decoded_bpe = bpe_tok.decode(ids_bpe)
    print(f"BPE Tokenizer: '{tgt_text}' -> {ids_bpe} -> '{decoded_bpe}'")
    # SentencePiece normalization might change spacing slightly or not, usually exact match for simple ASCII
    assert decoded_bpe == tgt_text, "BPE tokenizer decode mismatch"

    print("Tokenizer verification passed.")
    return char_tok, bpe_tok


def demonstrate_dataset(char_tok, bpe_tok):
    print("\n=== Demonstrating Dataset Pipeline ===")

    # Prepare data (process raw CSV -> Parquet -> Dataset objects)
    train_ds, val_ds, _, _ = dataset.prepare_data(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")

    # Check item structure
    if len(train_ds) > 0:
        sample = train_ds[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Check shapes
        # Input ids should be padded/truncated to MAX_INPUT_LEN
        assert sample["input_ids"].shape[0] == config.MAX_INPUT_LEN
        # Labels should be padded/truncated to MAX_OUTPUT_LEN
        assert sample["labels"].shape[0] == config.MAX_OUTPUT_LEN

        print("Dataset item shapes verified.")

    return train_ds, val_ds


def demonstrate_model(char_tok, bpe_tok):
    print("\n=== Demonstrating SemioticTransformer ===")

    model = model_lib.SemioticTransformer(
        src_vocab_size=char_tok.vocab_size,
        tgt_vocab_size=bpe_tok.vocab_size,
        d_model=32,  # Reduced for demo
        nhead=2,  # Reduced for demo
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
    ).to(config.DEVICE)

    # Create dummy batch
    batch_size = 2
    src = torch.randint(0, char_tok.vocab_size, (batch_size, config.MAX_INPUT_LEN)).to(
        config.DEVICE
    )
    tgt = torch.randint(0, bpe_tok.vocab_size, (batch_size, config.MAX_OUTPUT_LEN)).to(
        config.DEVICE
    )

    # Generate mask
    tgt_mask = model.generate_square_subsequent_mask(config.MAX_OUTPUT_LEN).to(
        config.DEVICE
    )

    # Forward pass
    output = model(src, tgt, tgt_mask=tgt_mask)

    # Check Output Shape: [batch_size, seq_len, vocab_size]
    expected_shape = (batch_size, config.MAX_OUTPUT_LEN, bpe_tok.vocab_size)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"

    print("Model forward pass verified.")
    return model


def demonstrate_training(train_ds, val_ds, char_tok, bpe_tok):
    print("\n=== Demonstrating Training Loop ===")

    # We use the trainer module provided.
    # Note: trainer.train_model initializes its own model internally based on config.
    # Since we modified config, it will use our reduced hyperparameters.

    trained_model = trainer.train_model(train_ds, val_ds, char_tok, bpe_tok)

    # Verify checkpoint creation
    assert os.path.exists(config.BEST_MODEL_PATH), "Model checkpoint was not saved."
    print(f"Training complete. Checkpoint saved at {config.BEST_MODEL_PATH}")


def demonstrate_inference():
    print("\n=== Demonstrating Inference/Submission ===")

    # Run the full inference pipeline
    # load_cached_data=True allows it to pick up the tokenizers and HFBB stats we just built
    inference.generate_submission(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Check columns
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission missing required columns"
    assert (
        len(df_sub) == 3
    ), f"Expected 3 predictions (from mini_test.csv), got {len(df_sub)}"

    print("Inference pipeline verified.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Utils
    demonstrate_utils()

    # 3. HFBB
    hfbb_engine = demonstrate_hfbb()

    # 4. Tokenizers
    char_tok, bpe_tok = demonstrate_tokenizers()

    # 5. Dataset
    train_ds, val_ds = demonstrate_dataset(char_tok, bpe_tok)

    # 6. Model (Unit Test)
    demonstrate_model(char_tok, bpe_tok)

    # 7. Training
    demonstrate_training(train_ds, val_ds, char_tok, bpe_tok)

    # 8. Inference
    demonstrate_inference()

    print("\n=== All Demonstrations Completed Successfully ===")
