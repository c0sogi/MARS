import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

# Force unload library modules to ensure fresh import (Cite debug_lesson_1)
# This handles cases where the script is run in a persistent interpreter session.
for module_name in list(sys.modules.keys()):
    if module_name.startswith("library"):
        del sys.modules[module_name]

from library.config import Config
from library.utils import seed_everything, compute_levenshtein
from library.tokenizer import Tokenizer
from library.dataset import (
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
)
from library.model import ShowAttendTell
from library.train import train_one_epoch, validate


def run_demo():
    print("===============================================================")
    print("   Spatial Attention-Guided Recurrent Network Pipeline Demo    ")
    print("===============================================================")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration for Fast Demonstration...")
    seed_everything(42)

    # Override Config parameters to ensure the script runs quickly (within minutes)
    # and uses a very small subset of data.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Use only 16 samples for dataloaders
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.NUM_WORKERS = (
        0  # Disable multiprocessing to avoid overhead/errors in this script
    )

    # Use CUDA if available, else CPU
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ---------------------------------------------------------
    # 2. Tokenizer Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Tokenizer...")
    # Initialize tokenizer. load_cached_data=False ensures we test the build logic from metadata.
    tokenizer = Tokenizer(load_cached_data=False)
    vocab_size = len(tokenizer)
    print(f"    Vocabulary Size: {vocab_size}")

    # Test encoding and decoding
    sample_inchi = "InChI=1S/H2O/h1H2"
    encoded_seq = tokenizer.text_to_sequence(sample_inchi, padding=False)
    decoded_text = tokenizer.sequence_to_text(encoded_seq)

    print(f"    Original: {sample_inchi}")
    print(f"    Encoded:  {encoded_seq}")
    print(f"    Decoded:  {decoded_text}")

    # Assertions
    assert isinstance(
        encoded_seq, torch.Tensor
    ), "Tokenizer output must be a torch.Tensor"
    assert (
        encoded_seq[0] == tokenizer.stoi[Config.SOS_TOKEN]
    ), "Sequence must start with SOS"
    assert (
        encoded_seq[-1] == tokenizer.stoi[Config.EOS_TOKEN]
    ), "Sequence must end with EOS"
    # Basic content check (ignoring potential special token artifacts if any)
    assert "H2O" in decoded_text, "Decoded text lost content information"
    print("    -> Tokenizer logic verified.")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoader...")
    # Get training dataloader with debug subset
    train_loader = get_train_dataloader(
        tokenizer, batch_size=Config.BATCH_SIZE, debug=True, load_cached_data=False
    )

    # Fetch a single batch
    try:
        images, captions, lengths = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader returned no data. Check metadata availability.")

    print(f"    Images Tensor Shape:   {images.shape}")
    print(f"    Captions Tensor Shape: {captions.shape}")
    print(f"    Lengths: {lengths}")

    # Assertions
    expected_image_shape = (Config.BATCH_SIZE, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        images.shape == expected_image_shape
    ), f"Expected image shape {expected_image_shape}, got {images.shape}"
    assert captions.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQUENCE_LENGTH,
    ), "Incorrect caption shape"
    print("    -> Dataset and DataLoader shapes verified.")

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = ShowAttendTell(vocab_size=vocab_size).to(device)

    # Move batch to device
    images = images.to(device)
    captions = captions.to(device)

    # Test Forward Pass (Training Mode - with teacher forcing)
    # We set teacher_forcing_ratio=1.0 to ensure deterministic path usage for shape check
    outputs = model(images, captions, teacher_forcing_ratio=1.0)
    print(f"    Training Output Shape: {outputs.shape}")

    # Assertions
    # Output should be (Batch, Max_Len, Vocab)
    expected_out_shape = (Config.BATCH_SIZE, Config.MAX_SEQUENCE_LENGTH, vocab_size)
    assert (
        outputs.shape == expected_out_shape
    ), f"Expected output shape {expected_out_shape}, got {outputs.shape}"

    # Test Forward Pass (Inference Mode - greedy decoding)
    model.eval()
    with torch.no_grad():
        outputs_inf = model(images, captions=None)
    print(f"    Inference Output Shape: {outputs_inf.shape}")
    assert outputs_inf.shape == expected_out_shape, "Inference output shape mismatch"
    print("    -> Model forward pass verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Training Loop (1 Epoch)...")
    # Setup simple optimizer and loss
    pad_idx = tokenizer.stoi[Config.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training for one epoch on the tiny debug dataset
    train_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, device, epoch=0, tokenizer=tokenizer
    )
    print(f"    Epoch 0 Train Loss: {train_loss:.4f}")

    # Assertions
    assert train_loss > 0, "Training loss should be positive"
    assert not np.isnan(train_loss), "Training loss returned NaN"
    print("    -> Training loop executed successfully.")

    # ---------------------------------------------------------
    # 6. Validation Loop Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Validation Loop...")
    val_loader = get_val_dataloader(
        tokenizer, batch_size=Config.BATCH_SIZE, debug=True, load_cached_data=False
    )

    val_loss, val_lev = validate(val_loader, model, criterion, device, tokenizer)
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation Levenshtein Distance: {val_lev:.4f}")

    # Assertions
    assert val_loss > 0, "Validation loss should be positive"
    assert val_lev >= 0, "Levenshtein distance cannot be negative"
    print("    -> Validation loop executed successfully.")

    # ---------------------------------------------------------
    # 7. Inference & Submission Verification
    # ---------------------------------------------------------
    print("\n[7] Verifying Inference on Test Data...")
    test_loader = get_test_dataloader(
        tokenizer, batch_size=Config.BATCH_SIZE, debug=True, load_cached_data=False
    )

    # Get one batch of test data
    test_images, test_ids = next(iter(test_loader))
    test_images = test_images.to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        # Run model
        outputs = model(test_images, captions=None)
        # Greedy decode
        predicted_indices = torch.argmax(outputs, dim=2)

        for idx in range(test_images.size(0)):
            seq = predicted_indices[idx].cpu().tolist()
            text = tokenizer.sequence_to_text(seq)
            predictions.append(text)

    print(f"    Generated {len(predictions)} predictions.")
    print(f"    Sample Prediction: {predictions[0]}")

    # Assertions
    assert len(predictions) == Config.BATCH_SIZE, "Prediction count mismatch"
    assert isinstance(predictions[0], str), "Prediction must be a string"
    print("    -> Inference logic verified.")

    print("\n===============================================================")
    print("   Demonstration Completed Successfully")
    print("===============================================================")


if __name__ == "__main__":
    run_demo()
