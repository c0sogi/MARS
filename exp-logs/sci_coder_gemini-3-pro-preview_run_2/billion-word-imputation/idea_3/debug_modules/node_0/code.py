import os
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.vocabulary import WordVocabulary
from library.dataset import InfillingDataset, collate_fn
from library.model import DualHeadTransformer
from library.utils import insert_word_in_sentence, set_seed, calculate_metrics
from library.train import train_model
from library.predict import generate_submission


def setup_demo_environment():
    """
    Creates a small subset of the data in a temporary working directory
    to ensure the demonstration runs quickly (under 1 hour).
    """
    print(">>> Setting up demo environment...")

    # Define demo directory
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Define paths for small datasets
    demo_train_path = os.path.join(demo_dir, "train_filtered_debug_200.parquet")
    demo_val_path = os.path.join(demo_dir, "val_filtered_debug_200.parquet")
    demo_test_path = os.path.join(demo_dir, "test_filtered_debug_200.parquet")

    # Load a tiny fraction of the real data
    # Note: Loading the full parquet into memory is fast enough (few GBs) given the 220GB RAM available.
    print("Loading source data to create subsets...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH).head(200)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH).head(50)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH).head(50)

    # Save subsets
    df_train.to_parquet(demo_train_path)
    df_val.to_parquet(demo_val_path)
    df_test.to_parquet(demo_test_path)

    print(f"Created demo datasets in {demo_dir}")

    # OVERRIDE CONFIGURATION FOR DEMO
    # We monkey-patch the Config class attributes to point to our small files
    Config.TRAIN_DATA_PATH = demo_train_path
    Config.VAL_DATA_PATH = demo_val_path
    Config.TEST_DATA_PATH = demo_test_path
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Update derived paths
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.TARGET_VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.json")
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_cache.parquet")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_cache.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VAL_BATCH_SIZE = 8
    Config.TARGET_VOCAB_SIZE = 1000  # Small vocab for demo
    Config.DEBUG = True  # Ensure dataset class knows we are debugging (though we manually sliced files)


def verify_vocabulary():
    """Demonstrates and verifies Vocabulary creation and usage."""
    print("\n>>> Verifying Vocabulary...")

    vocab = WordVocabulary()

    # Build from the small demo corpus
    vocab.build_from_corpus(
        Config.TRAIN_DATA_PATH,
        vocab_size=Config.TARGET_VOCAB_SIZE,
        save_path=Config.TARGET_VOCAB_PATH,
        load_cached=False,
    )

    # Assertions
    assert (
        len(vocab) > 2
    ), "Vocabulary should contain at least special tokens and some words."
    assert vocab.pad_token in vocab.token2id, "PAD token missing."
    assert vocab.unk_token in vocab.token2id, "UNK token missing."

    # Test mapping
    test_word = list(vocab.token2id.keys())[-1]  # Get a real word
    idx = vocab.token_to_id(test_word)
    recovered_word = vocab.id_to_token(idx)

    assert (
        test_word == recovered_word
    ), f"Vocab mapping failed: {test_word} != {recovered_word}"
    print(f"Vocabulary verified. Size: {len(vocab)}")
    return vocab


def verify_dataset_logic(vocab):
    """Demonstrates and verifies the InfillingDataset logic (masking and labeling)."""
    print("\n>>> Verifying Dataset Logic...")

    # Initialize dataset
    ds = InfillingDataset(split="train", vocabulary=vocab, load_cached=False)

    # Fetch one sample
    sample = ds[0]

    # Check keys
    required_keys = ["input_ids", "attention_mask", "loc_label", "word_label"]
    for k in required_keys:
        assert k in sample, f"Missing key in dataset sample: {k}"

    # Check shapes
    seq_len = Config.MAX_SEQ_LEN
    assert sample["input_ids"].shape == (
        seq_len,
    ), f"Input IDs shape mismatch: {sample['input_ids'].shape}"
    assert sample["loc_label"].shape == (
        seq_len,
    ), f"Loc label shape mismatch: {sample['loc_label'].shape}"

    # Verify Logic:
    # 1. loc_label should be a binary vector with exactly one 1.0 (unless sentence was too short/skipped)
    # Note: In rare cases of extremely short sentences, it might be all zeros, but our demo data is standard.
    if sample["loc_label"].sum().item() == 1.0:
        loc_idx = torch.argmax(sample["loc_label"]).item()
        print(f"Sample verified. Target location index: {loc_idx}")
        print(f"Target word ID: {sample['word_label'].item()}")
    else:
        print("Warning: Sample had no valid location label (possibly too short).")

    # Verify Collate Function
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, seq_len), "Batch collation shape mismatch."
    print("Dataset and DataLoader verified.")


def verify_model_forward_pass(vocab):
    """Demonstrates model instantiation and a forward pass."""
    print("\n>>> Verifying Model Forward Pass...")

    model = DualHeadTransformer(vocab_size=len(vocab))
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy batch
    seq_len = Config.MAX_SEQ_LEN
    dummy_input = torch.randint(0, 100, (2, seq_len)).to(Config.DEVICE)
    dummy_mask = torch.ones((2, seq_len)).to(Config.DEVICE)

    with torch.no_grad():
        loc_logits, word_logits = model(dummy_input, dummy_mask)

    # Check output shapes
    # loc_logits: (Batch, Seq_Len)
    assert loc_logits.shape == (
        2,
        seq_len,
    ), f"Loc logits shape incorrect: {loc_logits.shape}"

    # word_logits: (Batch, Seq_Len, Vocab_Size)
    assert word_logits.shape == (
        2,
        seq_len,
        len(vocab),
    ), f"Word logits shape incorrect: {word_logits.shape}"

    print("Model forward pass verified.")
    return model


def verify_reconstruction_logic(vocab):
    """Verifies the utility function that inserts the word back into the sentence."""
    print("\n>>> Verifying Sentence Reconstruction...")

    # We need a tokenizer to get offsets. We can grab it from a dataset instance.
    ds = InfillingDataset(split="test", vocabulary=vocab, load_cached=False)
    tokenizer = ds.tokenizer

    # Test Case
    original_sentence = "The quick brown fox over the lazy dog ."
    # We want to insert "jumps" after "fox".
    # "The quick brown fox" -> "The quick brown fox jumps over the lazy dog ."

    # We need to find the token index for "fox".
    # Tokenizer output depends on the model (DistilRoberta).
    # " fox" is usually one token.
    encoding = tokenizer(
        original_sentence, return_offsets_mapping=True, add_special_tokens=True
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    # Find index of 'fox' (or 'Ġfox')
    fox_idx = -1
    for i, t in enumerate(tokens):
        if "fox" in t:
            fox_idx = i
            break

    assert fox_idx != -1, "Could not find 'fox' in tokens for verification."

    word_to_insert = "jumps"

    reconstructed = insert_word_in_sentence(
        original_sentence, word_to_insert, fox_idx, tokenizer
    )

    print(f"Original: '{original_sentence}'")
    print(f"Insertion: '{word_to_insert}' at token {fox_idx} ('{tokens[fox_idx]}')")
    print(f"Result:   '{reconstructed}'")

    # Basic check: the word should be in the string now
    assert (
        word_to_insert in reconstructed
    ), "Inserted word not found in reconstructed string."

    print("Reconstruction logic verified.")


def run_training_and_inference_demo():
    """Runs the actual training and prediction functions provided in the library."""
    print("\n>>> Running Training Demo...")

    # Run training (uses the parameters set in setup_demo_environment)
    train_model(debug=True)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training completed and checkpoint verified.")

    print("\n>>> Running Inference Demo...")
    generate_submission(debug=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "sentence" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    print(f"Submission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Setup Environment (Create small data subsets)
    setup_demo_environment()

    # 2. Set Seeds
    set_seed(Config.SEED)

    # 3. Verify Components
    vocab = verify_vocabulary()
    verify_dataset_logic(vocab)
    verify_model_forward_pass(vocab)
    verify_reconstruction_logic(vocab)

    # 4. Run Pipeline
    run_training_and_inference_demo()

    print("\n>>> All demonstrations and verifications passed successfully.")
