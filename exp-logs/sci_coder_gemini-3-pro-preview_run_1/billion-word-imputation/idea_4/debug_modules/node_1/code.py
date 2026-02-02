import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.vocab import get_vocab, Vocabulary
from library.dataset import get_dataloaders, GapTokenDataset
from library.model import InterleavedTransformer
from library.engine import fit_model, generate_submission


def run_demo():
    # 1. Setup and Configuration Override
    print("--- Setting up Demo Configuration ---")

    # Define a separate working directory for the demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch Config to use small settings and local paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = os.path.join(demo_dir, "train_small.csv")
    Config.VAL_METADATA_PATH = os.path.join(demo_dir, "val_small.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_dir, "test_small.csv")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.VOCAB_PATH = os.path.join(demo_dir, "vocab.npy")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Reduce compute requirements for speed
    Config.VOCAB_SIZE = 1000
    Config.MAX_SEQ_LEN = 32  # Short sequences
    Config.EMBED_DIM = 64
    Config.HIDDEN_DIM = 128
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.DEBUG = False  # We are manually controlling data size via file creation

    # Re-setup directories based on new Config
    Config.setup()

    # Set seed
    set_seed(42)

    # 2. Create Synthetic Data
    print("--- Creating Synthetic Data ---")

    # Create ~20 sentences for training
    train_sentences = [
        "The quick brown fox jumps over the lazy dog .",
        "Machine learning is fascinating and powerful .",
        "Python is a great programming language for data science .",
        "The weather today is sunny with a chance of rain .",
        "Artificial intelligence will change the world significantly .",
    ] * 4  # Duplicate to get enough batches

    val_sentences = [
        "Validation data is used to tune hyperparameters .",
        "Deep learning models require significant compute resources .",
    ] * 5

    # Test sentences have one word removed (simulated here)
    test_sentences = [
        "The quick brown jumps over the lazy dog .",  # missing 'fox'
        "Machine learning is and powerful .",  # missing 'fascinating'
    ] * 5

    # Create DataFrames
    df_train = pd.DataFrame(
        {"id": range(len(train_sentences)), "sentence": train_sentences}
    )
    df_val = pd.DataFrame({"id": range(len(val_sentences)), "sentence": val_sentences})
    df_test = pd.DataFrame(
        {"id": range(len(test_sentences)), "sentence": test_sentences}
    )

    # Save to the paths Config expects
    df_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    df_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    df_test.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(
        f"Created train ({len(df_train)}), val ({len(df_val)}), test ({len(df_test)}) samples."
    )

    # 3. Vocabulary Demonstration
    print("\n--- Demonstrating Vocabulary ---")

    # Build vocabulary
    vocab = get_vocab(
        load_cached_data=False,
        vocab_path=Config.VOCAB_PATH,
        train_metadata_path=Config.TRAIN_METADATA_PATH,
    )

    # Verify special tokens
    assert vocab.stoi[Config.PAD_TOKEN] == Config.PAD_IDX
    assert vocab.stoi[Config.UNK_TOKEN] == Config.UNK_IDX
    assert vocab.stoi[Config.GAP_TOKEN] == Config.GAP_IDX

    # Verify encoding/decoding
    sample_text = "The quick brown fox"
    encoded = vocab.encode(sample_text)
    decoded = vocab.decode(encoded)

    print(f"Original: '{sample_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")

    # Basic logic check: decoded string should contain the words (ignoring case/unk if vocab small)
    # Since our vocab is built from this text, it should match exactly (except maybe casing if logic changed, but here it splits by space)
    assert len(encoded) == 4
    assert isinstance(encoded, list)
    assert isinstance(decoded, str)

    # 4. Dataset & DataLoader Demonstration
    print("\n--- Demonstrating Dataset & DataLoader ---")

    # Get loaders (force reload to ignore any previous cache)
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab, load_cached_data=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = {
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "target_word_id",
        "target_gap_idx",
        "original_ids",
    }
    assert set(batch.keys()) == expected_keys

    input_ids = batch["input_ids"]
    token_type_ids = batch["token_type_ids"]

    print(f"Batch Input Shape: {input_ids.shape}")

    # Verify Shapes
    # Shape should be (Batch_Size, Seq_Len)
    assert input_ids.shape[0] == Config.BATCH_SIZE
    assert token_type_ids.shape == input_ids.shape

    # Verify Interleaved Structure
    # Config.GAP_IDX is 2.
    # The sequence should start with GAP (2).
    # Odd indices (0, 2, 4...) should be GAP tokens in the interleaved format.
    # Note: Pad tokens might disrupt this pattern at the end, but the start should be consistent.
    first_seq = input_ids[0]
    assert (
        first_seq[0] == Config.GAP_IDX
    ), f"Expected first token to be GAP ({Config.GAP_IDX}), got {first_seq[0]}"
    assert token_type_ids[0][0] == 1, "Expected token_type_id 1 for GAP"

    # 5. Model Demonstration
    print("\n--- Demonstrating Model ---")

    model = InterleavedTransformer()
    model.to(Config.DEVICE)

    # Forward pass
    input_ids = input_ids.to(Config.DEVICE)
    token_type_ids = token_type_ids.to(Config.DEVICE)
    attention_mask = batch["attention_mask"].to(Config.DEVICE)

    loc_logits, id_logits = model(input_ids, token_type_ids, attention_mask)

    print(f"Loc Logits Shape: {loc_logits.shape}")
    print(f"ID Logits Shape: {id_logits.shape}")

    # Verify Output Shapes
    # Loc: (B, L, 1)
    assert loc_logits.shape == (Config.BATCH_SIZE, input_ids.shape[1], 1)
    # ID: (B, L, VocabSize + 3)
    # Note: Model initializes embedding with vocab_size + 3
    expected_vocab_dim = Config.VOCAB_SIZE + 3
    assert id_logits.shape == (
        Config.BATCH_SIZE,
        input_ids.shape[1],
        expected_vocab_dim,
    )

    # 6. Training Loop Demonstration
    print("\n--- Demonstrating Training Loop ---")

    # Run fit_model (Runs 1 epoch as configured)
    fit_model(model, train_loader, val_loader)

    # Verify model file created
    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint not found after training."
    print("Training completed and model saved.")

    # 7. Inference Demonstration
    print("\n--- Demonstrating Inference ---")

    # Generate submission
    generate_submission(model, test_loader, vocab, output_file=Config.SUBMISSION_FILE)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not generated."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission Head:\n{df_sub.head()}")

    # Verify submission content
    assert "id" in df_sub.columns
    assert "sentence" in df_sub.columns
    assert len(df_sub) == len(df_test)

    # Check that the predicted sentence is different from the input (something was inserted)
    # Though with a random model trained for 1 step, quality is low, logic should hold.
    # We just check it's a string.
    assert isinstance(df_sub.iloc[0]["sentence"], str)

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
