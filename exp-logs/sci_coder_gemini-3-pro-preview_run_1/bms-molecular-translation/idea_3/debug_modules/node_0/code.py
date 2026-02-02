import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms, collate_fn
from library.model import VisualTransformer
from library.train import train_one_epoch
from library.utils import compute_levenshtein


def run_demonstration():
    print("=== Starting Demonstration of InChi Library Components ===\n")

    # 1. Configuration Setup
    # Override Config for a fast demonstration run
    print("1. Setting up Configuration...")
    Config.setup()
    Config.DEBUG = True
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EPOCHS = 1
    Config.MAX_LEN = 50  # Reduce sequence length for speed

    # Ensure device is set correctly
    device = torch.device(Config.DEVICE)
    print(f"   Device: {device}")
    print("   Configuration setup complete.\n")

    # 2. Tokenizer Initialization and Verification
    print("2. Initializing Tokenizer...")
    # This will load from cache or build from the training metadata
    tokenizer = Tokenizer(load_cached_data=True, debug=True)

    print(f"   Vocabulary size: {len(tokenizer)}")

    # Verify text to sequence conversion
    sample_text = "InChI=1S/H2O/h1H2"
    seq = tokenizer.text_to_sequence(sample_text, max_len=Config.MAX_LEN, padding=True)
    print(f"   Sample text: {sample_text}")
    print(f"   Encoded sequence length: {len(seq)}")

    # Verify sequence to text conversion
    decoded_text = tokenizer.sequence_to_text(seq)
    print(f"   Decoded text: {decoded_text}")

    assert (
        len(seq) == Config.MAX_LEN
    ), "Sequence length does not match MAX_LEN with padding"
    assert decoded_text == sample_text, "Decoded text does not match original"
    assert seq[0] == tokenizer.sos_token_id, "Sequence must start with SOS"
    print("   Tokenizer logic verified.\n")

    # 3. Dataset and DataLoader
    print("3. Creating Dataset and DataLoader...")
    # Load a tiny subset of the training metadata
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata file not found at {Config.TRAIN_METADATA}")

    df_train = pd.read_csv(Config.TRAIN_METADATA, nrows=10)
    print(f"   Loaded {len(df_train)} rows from training metadata.")

    # Instantiate dataset
    train_dataset = InChiDataset(
        df=df_train, tokenizer=tokenizer, transform=get_transforms("train")
    )

    # Verify __getitem__
    sample_item = train_dataset[0]
    print(f"   Sample image shape: {sample_item['image'].shape}")
    print(f"   Sample sequence shape: {sample_item['seq'].shape}")

    assert sample_item["image"].shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image dimensions"
    assert sample_item["seq"].shape[0] == Config.MAX_LEN, "Incorrect sequence length"

    # Instantiate DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    batch = next(iter(train_loader))
    print(f"   Batch image shape: {batch['image'].shape}")
    print(f"   Batch sequence shape: {batch['seq'].shape}")
    print("   Dataset and DataLoader verified.\n")

    # 4. Model Initialization and Forward Pass
    print("4. Initializing Model...")
    model = VisualTransformer(vocab_size=len(tokenizer))
    model = model.to(device)
    print("   Model moved to device.")

    # Run a forward pass with the batch
    images = batch["image"].to(device)
    seqs = batch["seq"].to(device)

    # Input to decoder excludes the last token (EOS/PAD)
    decoder_input = seqs[:, :-1]

    print("   Running forward pass...")
    # We use pad_idx=0 as per the tokenizer standard in this library
    logits = model(images, decoder_input, pad_idx=tokenizer.pad_token_id)

    print(f"   Output logits shape: {logits.shape}")
    expected_seq_len = Config.MAX_LEN - 1
    assert logits.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
        len(tokenizer),
    ), f"Expected output shape ({Config.BATCH_SIZE}, {expected_seq_len}, {len(tokenizer)}), got {logits.shape}"
    print("   Forward pass verified.\n")

    # 5. Training Step Simulation
    print("5. Simulating Training Step...")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch (which is just a few steps since we loaded 10 rows with batch size 4)
    avg_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, device, epoch=0, scheduler=None
    )

    print(f"   Training step complete. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Loss is NaN"
    print("   Training logic verified.\n")

    # 6. Metric Verification
    print("6. Verifying Levenshtein Metric...")
    str1 = "InChI=1S/H2O/h1H2"
    str2 = "InChI=1S/H2O/h1H2"  # Exact match
    str3 = "InChI=1S/H3O/h1H3"  # 2 edits

    score_exact = compute_levenshtein([str1], [str2])
    score_diff = compute_levenshtein([str1], [str3])

    print(f"   Distance ('{str1}', '{str2}'): {score_exact}")
    print(f"   Distance ('{str1}', '{str3}'): {score_diff}")

    assert score_exact == 0.0, "Levenshtein distance for identical strings should be 0"
    assert score_diff > 0.0, "Levenshtein distance for different strings should be > 0"
    print("   Metric calculation verified.\n")

    print("=== Demonstration Complete: All Components Functioning Correctly ===")


if __name__ == "__main__":
    run_demonstration()
