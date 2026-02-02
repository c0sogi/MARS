import os
import pandas as pd
import torch
import numpy as np
import shutil
from pathlib import Path

# Import library components
from library.config import Config
from library.tokenizer import CharTokenizer
from library.data_manager import build_ngram_stats, NormalizationDataset
from library.neural_model import TransformerSeq2Seq
from library.trainer import run_training, seed_everything
from library.hybrid_system import HybridPredictor


def create_dummy_data(base_dir):
    """
    Creates small dummy CSV files for training, validation, and testing
    to allow the demo to run quickly without processing the full dataset.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Schema: sentence_id, token_id, class, before, after
    # We include some digits to trigger the neural model routing logic.
    train_data = [
        [0, 0, "PLAIN", "I", "I"],
        [0, 1, "PLAIN", "have", "have"],
        [0, 2, "CARDINAL", "2", "two"],
        [0, 3, "PLAIN", "cats", "cats"],
        [0, 4, "PUNCT", ".", "."],
        [1, 0, "PLAIN", "It", "It"],
        [1, 1, "PLAIN", "is", "is"],
        [1, 2, "TIME", "5", "five"],
        [1, 3, "PLAIN", "pm", "p m"],
        [2, 0, "PLAIN", "Room", "Room"],
        [2, 1, "CARDINAL", "101", "one zero one"],
    ]

    val_data = [
        [3, 0, "PLAIN", "See", "See"],
        [3, 1, "PLAIN", "you", "you"],
        [3, 2, "PLAIN", "at", "at"],
        [3, 3, "DIGIT", "9", "nine"],
    ]

    # Test data does not have 'after' or 'class'
    test_data = [
        [0, 0, "I"],
        [0, 1, "have"],
        [0, 2, "2"],
        [0, 3, "cats"],
        [1, 0, "Room"],
        [1, 1, "101"],
    ]

    df_train = pd.DataFrame(
        train_data, columns=["sentence_id", "token_id", "class", "before", "after"]
    )
    df_val = pd.DataFrame(
        val_data, columns=["sentence_id", "token_id", "class", "before", "after"]
    )
    df_test = pd.DataFrame(test_data, columns=["sentence_id", "token_id", "before"])

    train_path = os.path.join(base_dir, "mini_train.csv")
    val_path = os.path.join(base_dir, "mini_val.csv")
    test_path = os.path.join(base_dir, "mini_test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Created dummy datasets at {base_dir}")
    return train_path, val_path, test_path


def configure_demo_environment(working_dir, train_path, val_path, test_path):
    """
    Overrides Config attributes to use the temporary directory and dummy data,
    and sets lightweight model hyperparameters for speed.
    """
    print("Configuring environment for demo execution...")

    # Paths
    Config.WORKING_DIR = working_dir
    Config.INPUT_DIR = working_dir  # Just for consistency in this demo
    Config.TRAIN_FILE = train_path
    Config.VAL_FILE = val_path
    Config.TEST_FILE = test_path

    # Artifact paths
    Config.NGRAM_STATS_PATH = os.path.join(working_dir, "ngram_stats.npy")
    Config.TOKENIZER_PATH = os.path.join(working_dir, "char_tokenizer.json")
    Config.MODEL_CHECKPOINT = os.path.join(working_dir, "neural_normalizer_demo.pt")
    Config.SUBMISSION_PATH = os.path.join(working_dir, "submission.csv")

    # Model Hyperparameters (Tiny model for instant training)
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 64
    Config.N_LAYERS = 2
    Config.N_HEADS = 2
    Config.DROPOUT = 0.0

    # Training Hyperparameters
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.DEBUG_SUBSET_SIZE = None  # Use full mini dataset

    # Ensure working dir exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.print_summary()


def demo_tokenizer():
    print("\n--- 1. Tokenizer Demonstration ---")
    tokenizer = CharTokenizer()

    # Build vocabulary from the mini train file
    # load_cached=False forces rebuild
    tokenizer.build_vocab(Config.TRAIN_FILE, load_cached=False)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test Encoding
    sample_text = "cat"
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f"Encoded '{sample_text}': {encoded}")

    # Test Decoding
    decoded = tokenizer.decode(encoded, remove_special_tokens=True)
    print(f"Decoded: '{decoded}'")

    # Validation
    assert (
        decoded == sample_text
    ), f"Decoding failed: expected {sample_text}, got {decoded}"
    assert tokenizer.pad_token_id is not None
    print("Tokenizer logic verified.")
    return tokenizer


def demo_ngram_stats():
    print("\n--- 2. N-gram Statistics Demonstration ---")
    # Build stats from scratch
    stats = build_ngram_stats(Config.TRAIN_FILE, load_cached=False)

    # Verify structure
    assert "unigram" in stats
    assert "bigram" in stats
    assert "trigram" in stats

    # Check if known mapping exists
    # In dummy data: "2" -> "two"
    if "2" in stats["unigram"]:
        print(f"Unigram check: '2' maps to '{stats['unigram']['2']}'")
        assert stats["unigram"]["2"] == "two"

    print("N-gram stats logic verified.")


def demo_dataset_loading(tokenizer):
    print("\n--- 3. Dataset & DataLoader Demonstration ---")
    # Initialize dataset
    # This dataset filters for tokens containing digits (based on Config.DIGIT_REGEX)
    ds = NormalizationDataset(
        data_path=Config.TRAIN_FILE,
        tokenizer=tokenizer,
        max_len=Config.MAX_INPUT_LEN,
        context_window=Config.CONTEXT_WINDOW,
        mode="train",
        load_cached=False,
    )

    print(f"Dataset size (digit tokens only): {len(ds)}")

    if len(ds) > 0:
        sample = ds[0]
        print("Sample keys:", sample.keys())
        print(f"Original Before: {sample['original_before']}")
        print(f"Original After: {sample['original_after']}")
        print(f"Source Tensor Shape: {sample['src'].shape}")

        # Verify tensor types
        assert isinstance(sample["src"], torch.Tensor)
        assert isinstance(sample["tgt"], torch.Tensor)
    else:
        print("Warning: No digit tokens found in dummy data for dataset demo.")

    print("Dataset logic verified.")


def demo_training():
    print("\n--- 4. Neural Model Training Demonstration ---")
    # run_training orchestrates the whole process:
    # 1. Builds vocab
    # 2. Creates datasets
    # 3. Initializes model & optimizer
    # 4. Runs training loop
    model = run_training(load_cached_data=True)  # Use cached vocab from step 1

    # Verify model checkpoint exists
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not saved!"
    print(f"Model successfully trained and saved to {Config.MODEL_CHECKPOINT}")

    # Basic forward pass check
    model.eval()
    dummy_src = torch.randint(0, 10, (1, 10)).to(Config.DEVICE)
    dummy_tgt = torch.randint(0, 10, (1, 10)).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_src, dummy_tgt)
    assert output.shape == (1, 10, model.fc_out.out_features)
    print("Model forward pass verified.")


def demo_hybrid_inference():
    print("\n--- 5. Hybrid System Inference Demonstration ---")

    # Initialize predictor
    # This loads N-grams, Tokenizer, and the Neural Model we just trained
    predictor = HybridPredictor(load_cached_data=True)

    # Generate submission
    predictor.generate_submission(
        test_file=Config.TEST_FILE, submission_path=Config.SUBMISSION_PATH
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print("Head of submission:")
    print(df_sub.head())

    # Check columns
    assert "id" in df_sub.columns
    assert "after" in df_sub.columns

    # Check row count matches test file
    df_test = pd.read_csv(Config.TEST_FILE)
    assert len(df_sub) == len(
        df_test
    ), f"Submission length mismatch: {len(df_sub)} vs {len(df_test)}"

    print("Hybrid inference logic verified.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    # Define working directory for this demo
    DEMO_DIR = "./working/demo_execution"

    # 1. Create Dummy Data
    train_p, val_p, test_p = create_dummy_data(DEMO_DIR)

    # 2. Configure Environment
    configure_demo_environment(DEMO_DIR, train_p, val_p, test_p)

    # 3. Run Demonstrations
    try:
        # Tokenizer
        tokenizer = demo_tokenizer()

        # N-grams
        demo_ngram_stats()

        # Dataset
        demo_dataset_loading(tokenizer)

        # Training
        demo_training()

        # Inference
        demo_hybrid_inference()

        print("\nAll demonstrations completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        raise e
