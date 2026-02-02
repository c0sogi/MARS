import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import time

# Import library components
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_dataloaders
from library.model import GFCN
from library.train import train_one_epoch, validate, generate_submission
from library.utils import save_checkpoint, load_checkpoint, compute_levenshtein


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("Starting Demo Script...")
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 1. Configure for Speed/Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.npy")

    # Create demo directories
    Config.create_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Verify Tokenizer
    # ---------------------------------------------------------
    print("\n[2] Verifying Tokenizer...")
    # Force rebuild of vocab for the demo to ensure it works with the subset logic if needed,
    # but here we rely on the full metadata existing. We'll use load_cached_data=False
    # to test the build process or True if we assume cache might exist.
    # Let's try building it to verify logic.
    tokenizer = Tokenizer(load_cached_data=False)

    test_str = "InChI=1S/H2O/h1H2"
    encoded = tokenizer.encode(test_str)
    print(f"Encoded '{test_str}': {encoded}")

    assert len(encoded) == len(test_str), "Encoding length mismatch"
    assert isinstance(encoded, torch.Tensor), "Encoded output should be a tensor"

    # Test Decoding
    # Create dummy logits: (Batch=1, Time=5, Classes=vocab_len)
    # We want to decode to indices corresponding to 'C', 'C' (collapsed), 'H'
    # Let's assume 'C' is index 5, 'H' is index 10 (just examples, we look them up)
    if "C" in tokenizer.char_to_idx and "H" in tokenizer.char_to_idx:
        idx_c = tokenizer.char_to_idx["C"]
        idx_h = tokenizer.char_to_idx["H"]
        blank = 0

        # Sequence: C, blank, C, C, H -> Decodes to CCH (CTC collapse C,C -> C? No, C, blank, C -> CC)
        # Wait, C, C -> C. C, blank, C -> CC.
        # Let's try: C, C, H -> CH
        vocab_size = len(tokenizer)
        dummy_logits = torch.zeros(1, 3, vocab_size)
        # Time 0: C
        dummy_logits[0, 0, idx_c] = 10.0
        # Time 1: C
        dummy_logits[0, 1, idx_c] = 10.0
        # Time 2: H
        dummy_logits[0, 2, idx_h] = 10.0

        decoded = tokenizer.decode_greedy(dummy_logits)
        print(f"Dummy Decode Result: {decoded[0]}")
        assert (
            decoded[0] == "CH"
        ), f"Expected 'CH' from greedy decode of C-C-H, got {decoded[0]}"

    # ---------------------------------------------------------
    # 3. Verify Dataset and Loaders
    # ---------------------------------------------------------
    print("\n[3] Verifying DataLoaders...")
    # We use load_cached_data=False to force loading from CSVs and sampling
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch
    images, labels, label_lengths = next(iter(train_loader))

    print(f"Image Batch Shape: {images.shape}")
    print(f"Label Batch Shape: {labels.shape}")
    print(f"Label Lengths: {label_lengths}")

    # Assertions
    assert images.ndim == 4, "Images should be 4D (B, C, H, W)"
    assert images.shape[1] == 1, "Images should be 1 channel (grayscale)"
    assert (
        images.shape[2] == Config.IMAGE_HEIGHT
    ), f"Height should be {Config.IMAGE_HEIGHT}"
    # Width is padded to MAX_WIDTH (2560) in dataset.py
    assert images.shape[3] == 2560, "Width should be padded to 2560"
    assert labels.ndim == 2, "Labels should be 2D (B, MaxLen)"
    assert (
        len(label_lengths) == Config.BATCH_SIZE
    ), "Label lengths should match batch size"

    # ---------------------------------------------------------
    # 4. Verify Model
    # ---------------------------------------------------------
    print("\n[4] Verifying Model...")
    model = GFCN(num_classes=len(tokenizer)).to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    logits = model(images)
    print(f"Logits Shape: {logits.shape}")

    # Assertions
    # Output should be (B, T, C)
    # T depends on width downsampling. ResNet18 modified:
    # Conv1 (stride 2) -> MaxPool (stride 2) -> Layer1 (1) -> Layer2 (stride 2,1) -> Layer3 (2,1) -> Layer4 (2,1)
    # Vertical total stride: 2*2*1*2*2*2 = 32. 256 / 32 = 8.
    # Horizontal total stride: 2*2*1*1*1*1 = 4. 2560 / 4 = 640.
    # So features are (B, 512, 8, 640).
    # Then max pool over height -> (B, 512, 640).
    # Then head -> (B, C, 640) -> Permute -> (B, 640, C).

    expected_time_steps = 2560 // 4
    assert logits.shape[0] == Config.BATCH_SIZE
    assert (
        logits.shape[1] == expected_time_steps
    ), f"Expected time steps {expected_time_steps}, got {logits.shape[1]}"
    assert logits.shape[2] == len(tokenizer), "Last dimension should match vocab size"

    # ---------------------------------------------------------
    # 5. Verify Training Step
    # ---------------------------------------------------------
    print("\n[5] Verifying Training Step...")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch (it's short because of DEBUG sampling)
    loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=0)
    print(f"Training Step Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # ---------------------------------------------------------
    # 6. Verify Validation Step
    # ---------------------------------------------------------
    print("\n[6] Verifying Validation Step...")
    val_loss, val_metric = validate(model, val_loader, criterion, tokenizer, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Levenshtein: {val_metric:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert val_metric >= 0, "Levenshtein distance cannot be negative"

    # ---------------------------------------------------------
    # 7. Verify Checkpointing and Inference
    # ---------------------------------------------------------
    print("\n[7] Verifying Checkpointing and Inference...")

    # Save dummy best model
    save_checkpoint(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": val_metric,
        },
        is_best=True,
    )

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created"

    # Load model
    model_infer = GFCN(num_classes=len(tokenizer)).to(device)
    epoch, best_score = load_checkpoint(model_infer, filename=Config.BEST_MODEL_PATH)
    print(f"Loaded checkpoint: Epoch {epoch}, Score {best_score:.4f}")
    assert epoch == 1

    # Generate submission on test loader
    # We iterate manually briefly to check logic, then call the function
    print("Running submission generation...")
    generate_submission(model_infer, test_loader, tokenizer, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"First few rows:\n{df_sub.head()}")

    assert "image_id" in df_sub.columns
    assert "InChI" in df_sub.columns
    assert len(df_sub) > 0

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
