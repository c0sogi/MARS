import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library is in path
sys.path.append(".")

# Import Config first to modify it for the demonstration
from library.config import Config

# --- Modify Config for Fast Demonstration ---
print("Configuring for fast demonstration...")
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
Config.EPOCHS = 1  # Only 1 epoch
Config.BATCH_SIZE = 4  # Small batch size
Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in simple script
Config.WORKING_DIR = "./working/demo"  # Separate working directory
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Import library modules after Config modification
from library.tokenizer import Tokenizer
from library.dataset import ChemicalDataset
from library.model import CRNN
from library.engine import fit, predict_and_submit


def main():
    # Set seed for reproducibility in this script
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # 1. Test Tokenizer
    # -------------------------------------------------------------------------
    print("\n=== Testing Tokenizer ===")
    tokenizer = Tokenizer()

    # Test Encoding
    sample_text = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    encoded_seq = tokenizer.text_to_sequence(sample_text)
    print(f"Original Text: {sample_text}")
    print(f"Encoded Sequence Shape: {encoded_seq.shape}")

    # Validate basic encoding
    assert len(encoded_seq) == len(sample_text), "Encoded sequence length mismatch."

    # Test CTC Decoding Logic (Collapse repeats, handle blanks)
    # We construct a sequence manually: [C, C, Blank, C] -> Should decode to "CC"
    # 'C' is in VOCAB.
    idx_C = Config.CHAR2IDX["C"]
    blank = Config.BLANK_IDX

    # Sequence: C, C, Blank, C
    test_seq_indices = torch.tensor([idx_C, idx_C, blank, idx_C], dtype=torch.long)
    decoded_text = tokenizer.sequence_to_text(test_seq_indices)

    print(f"Test Sequence Indices: {test_seq_indices.tolist()}")
    print(f"Decoded Text: {decoded_text}")

    assert (
        decoded_text == "CC"
    ), f"Tokenizer CTC decoding failed. Expected 'CC', got '{decoded_text}'"
    print("Tokenizer verified.")

    # -------------------------------------------------------------------------
    # 2. Test Dataset
    # -------------------------------------------------------------------------
    print("\n=== Testing Dataset ===")
    # Initialize dataset in train mode (loads metadata, samples subset due to DEBUG=True)
    # load_cached_data=False ensures we test the CSV loading logic
    train_dataset = ChemicalDataset(mode="train", load_cached_data=False)

    print(f"Train Dataset Length (Debug): {len(train_dataset)}")
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_dataset)}"

    # Fetch one sample
    image, label_seq, label_len = train_dataset[0]

    print(f"Image Tensor Shape: {image.shape}")
    print(f"Label Sequence Shape: {label_seq.shape}")
    print(f"Label Length: {label_len}")

    # Verify shapes
    # Image should be (1, 128, 2048) based on Config
    assert image.shape == (
        1,
        Config.IMAGE_HEIGHT,
        Config.IMAGE_WIDTH,
    ), f"Image shape mismatch. Expected {(1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)}, got {image.shape}"
    assert isinstance(label_seq, torch.Tensor), "Label sequence is not a Tensor."
    print("Dataset verified.")

    # -------------------------------------------------------------------------
    # 3. Test Model Architecture
    # -------------------------------------------------------------------------
    print("\n=== Testing Model Architecture ===")
    model = CRNN().to(Config.DEVICE)

    # Create a dummy batch of images: (Batch=2, Channels=1, H=128, W=2048)
    dummy_input = torch.randn(2, 1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH).to(
        Config.DEVICE
    )

    # Run forward pass
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")

    # Expected Output: (Batch, Sequence_Length, Num_Classes)
    # Based on architecture:
    # Width 2048 -> Pool1(stride 2) -> 1024. Subsequent pools are stride (2,1), so Width stays 1024.
    expected_seq_len = 1024
    assert output.shape == (
        2,
        expected_seq_len,
        Config.NUM_CLASSES,
    ), f"Model output mismatch. Expected {(2, expected_seq_len, Config.NUM_CLASSES)}, got {output.shape}"
    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 4. Test Training Engine
    # -------------------------------------------------------------------------
    print("\n=== Running Training Loop (1 Epoch) ===")
    # We pass epochs=1 explicitly because the default argument in 'fit' is evaluated at import time
    # and might hold the original Config value if not careful.
    best_model_path = fit(epochs=1, load_cached_data=False)

    print(f"Training complete. Best model saved to: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model file was not created."

    # -------------------------------------------------------------------------
    # 5. Test Inference and Submission
    # -------------------------------------------------------------------------
    print("\n=== Running Inference ===")
    # This uses the 'test' mode of the dataset and generates a submission CSV
    predict_and_submit(best_model_path)

    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(submission_path), "Submission file was not created."

    # Validate submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission DataFrame Shape: {df_sub.shape}")
    print("First 3 rows of submission:")
    print(df_sub.head(3))

    # In debug mode, test dataset is also sampled to DEBUG_SAMPLE_SIZE
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
