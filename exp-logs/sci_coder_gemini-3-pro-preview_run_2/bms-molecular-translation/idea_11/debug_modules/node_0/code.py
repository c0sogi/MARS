import sys
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure library is importable
sys.path.append(".")

from library.config import Config
from library.tokenizer import InChITokenizer
from library.dataset import InChIDataset, get_transforms, collate_fn
from library.model import AnisotropicResNetTransformer
from library.engine import train_one_epoch, validate
from library.utils import seed_everything


def run_demo():
    # Set seed for reproducibility
    seed_everything(42)

    print("=== Starting Demonstration ===\n")

    # 1. Setup Config for Demo
    # We enable debug mode to set low defaults, then override specific paths for this run
    Config.setup(debug=True, batch_size=4, epochs=1)

    # Create a tiny subset of metadata for speed
    # We read the first 10 rows of the existing train metadata
    full_train_path = "./metadata/train.csv"
    if os.path.exists(full_train_path):
        df_full = pd.read_csv(full_train_path, nrows=10)
        subset_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
        df_full.to_csv(subset_path, index=False)

        # Override Config paths to use the subset
        Config.TRAIN_METADATA = subset_path
        Config.VAL_METADATA = subset_path  # Use same for validation in demo
        Config.VOCAB_PATH = os.path.join(
            Config.WORKING_DIR, "demo_vocab.npy"
        )  # New vocab path
        print(
            f"Created temporary metadata subset at {subset_path} with {len(df_full)} samples."
        )
    else:
        raise FileNotFoundError(
            f"Metadata not found at {full_train_path}. Cannot run demo."
        )

    # 2. Tokenizer
    print("\n--- Initializing Tokenizer ---")
    # Force build from the small metadata subset by ensuring cache doesn't exist or forcing reload
    if os.path.exists(Config.VOCAB_PATH):
        os.remove(Config.VOCAB_PATH)

    tokenizer = InChITokenizer(load_cached_data=False)
    print(f"Vocabulary size: {len(tokenizer)}")

    # Verify tokenizer logic
    test_str = "InChI=1S/H2O/h1H2"
    # Note: Characters in test_str must be in the vocab built from the 10 samples.
    # 'InChI=1S/' is standard, likely present.
    seq = tokenizer.text_to_sequence(test_str)
    decoded = tokenizer.sequence_to_text(seq)
    print(f"Test Encoding: {test_str} -> {seq}")
    print(f"Test Decoding: {seq} -> {decoded}")

    assert isinstance(seq, list), "Tokenizer output should be a list"
    assert isinstance(decoded, str), "Tokenizer decoding should return a string"
    assert len(seq) > 2, "Sequence should contain at least SOS, EOS and some chars"

    # 3. Dataset and DataLoader
    print("\n--- Initializing Dataset and DataLoader ---")
    transforms = get_transforms("train")
    dataset = InChIDataset(
        metadata_path=Config.TRAIN_METADATA,
        tokenizer=tokenizer,
        transform=transforms,
        phase="train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 0 for simple debug execution
        collate_fn=collate_fn,
    )

    # Verify Batch Structure
    images, labels, lengths = next(iter(dataloader))
    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.ndim == 4, "Images should be [B, C, H, W]"
    assert images.shape[1] == 1, "Images should be grayscale (1 channel)"
    assert (
        images.shape[2] == Config.IMAGE_HEIGHT
    ), f"Height should be {Config.IMAGE_HEIGHT}"
    assert labels.ndim == 2, "Labels should be [B, SeqLen]"
    assert labels.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    # 4. Model
    print("\n--- Initializing Model ---")
    model = AnisotropicResNetTransformer(vocab_size=len(tokenizer))
    model.to(Config.DEVICE)

    # Forward pass (Training mode: with targets)
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)

    print("Running forward pass (training mode)...")
    logits = model(images, labels)
    print(f"Logits Shape: {logits.shape}")  # Expected: [B, SeqLen, Vocab]

    assert logits.shape[0] == Config.BATCH_SIZE
    assert logits.shape[1] == labels.shape[1]
    assert logits.shape[2] == len(tokenizer)

    # Forward pass (Inference mode: without targets)
    print("Running forward pass (inference mode)...")
    memory = model(images)
    print(f"Memory Shape: {memory.shape}")  # Expected: [H*W, B, Dim]

    assert memory.shape[1] == Config.BATCH_SIZE
    assert memory.shape[2] == Config.DECODER_DIM

    # 5. Training Loop (Engine)
    print("\n--- Testing Training Engine (1 Epoch) ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # train_one_epoch returns the average loss
    train_loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        device=Config.DEVICE,
        epoch=1,
    )
    print(f"Training Loop Completed. Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float)

    # 6. Validation Loop (Engine)
    print("\n--- Testing Validation Engine ---")
    # We use the same dataloader for validation in this demo
    val_metrics = validate(
        model=model, dataloader=dataloader, tokenizer=tokenizer, device=Config.DEVICE
    )
    print(f"Validation Metrics: {val_metrics}")

    assert "val_loss" in val_metrics
    assert "val_score" in val_metrics
    assert isinstance(val_metrics["val_score"], float)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
