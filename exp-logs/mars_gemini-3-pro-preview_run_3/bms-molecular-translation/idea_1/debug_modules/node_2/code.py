import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path to import library modules correctly
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.tokenizer import InchiTokenizer
from library.dataset import get_dataloaders
from library.model import ShowAndTell
from library.train import train_model
from library.predict import generate_submission


def run_demonstration():
    print("=== InChI Prediction Pipeline Demonstration ===\n")

    # --- 1. Configuration Override for Speed ---
    print("[1] Configuring hyperparameters for rapid execution...")
    # We modify the Config class attributes directly to affect all downstream modules
    Config.DEBUG_SIZE = 32  # Use only 32 samples for training/testing
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.ENCODER_PRETRAINED = (
        False  # Disable downloading weights for speed/offline safety in demo
    )

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Hyperparameters set: Debug Size=32, Batch Size=8, Epochs=1")

    # --- 2. Tokenizer Verification ---
    print("\n[2] Verifying Tokenizer...")
    tokenizer = InchiTokenizer()
    sample_inchi = "InChI=1S/H2O/h1H2"
    print(f"    Original Text: {sample_inchi}")

    # Convert to sequence
    seq = tokenizer.text_to_sequence(
        sample_inchi, max_len=Config.MAX_TEXT_LEN, padding=True
    )
    print(f"    Tokenized Sequence (first 10): {seq[:10].tolist()}...")

    # Decode back to text
    decoded_text = tokenizer.sequence_to_text(seq)
    print(f"    Decoded Text:  {decoded_text}")

    assert (
        decoded_text == sample_inchi
    ), f"Tokenizer round-trip failed! Got {decoded_text}, expected {sample_inchi}"
    print("    Tokenizer logic verified.")

    # --- 3. Data Loading Verification ---
    print("\n[3] Verifying Data Loaders...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        debug=True, debug_size=Config.DEBUG_SIZE
    )

    # Fetch a single batch
    images, captions = next(iter(train_loader))

    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Caption Batch Shape: {captions.shape}")

    # Assertions
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    expected_cap_shape = (Config.BATCH_SIZE, Config.MAX_TEXT_LEN)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        captions.shape == expected_cap_shape
    ), f"Caption shape mismatch. Expected {expected_cap_shape}, got {captions.shape}"
    print("    Data loader shapes verified.")

    # --- 4. Model Architecture Verification ---
    print("\n[4] Verifying Model Forward Pass...")
    vocab_size = len(tokenizer)
    model = ShowAndTell(vocab_size=vocab_size)
    model = model.to(Config.DEVICE)

    images = images.to(Config.DEVICE)
    captions = captions.to(Config.DEVICE)

    # Perform forward pass
    # The model expects captions to include <SOS> and <EOS>, and internally slices for input
    logits = model(images, captions)

    print(f"    Logits Shape: {logits.shape}")

    # Expected output shape: [Batch, Sequence Length - 1, Vocab Size]
    # Sequence Length - 1 because we predict the next token for every token except the last one
    expected_out_seq_len = Config.MAX_TEXT_LEN - 1
    expected_logits_shape = (Config.BATCH_SIZE, expected_out_seq_len, vocab_size)

    assert (
        logits.shape == expected_logits_shape
    ), f"Logits shape mismatch. Expected {expected_logits_shape}, got {logits.shape}"
    print("    Model forward pass verified.")

    # --- 5. Training Loop Simulation ---
    print("\n[5] Simulating Training Loop (1 Epoch)...")
    # train_model handles the full loop: train_one_epoch -> validate -> save_checkpoint
    trained_model = train_model(debug=True, epochs=Config.NUM_EPOCHS)
    print("    Training simulation completed.")

    # --- 6. Inference and Submission Generation ---
    print("\n[6] Generating Submission on Test Set...")
    # generate_submission loads the best model (or uses the one provided) and predicts on test_loader
    generate_submission(model=trained_model, debug=True)

    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file)
        print(f"    Submission file generated at: {submission_file}")
        print(f"    Rows in submission: {len(df_sub)}")
        print("    First 3 rows:")
        print(df_sub.head(3))

        # Verify submission format
        assert (
            "image_id" in df_sub.columns and "InChI" in df_sub.columns
        ), "Submission missing required columns"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
