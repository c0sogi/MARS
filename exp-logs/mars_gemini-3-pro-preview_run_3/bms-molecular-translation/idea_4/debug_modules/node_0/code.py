import os
import sys
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the current directory is in the path to import library modules correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms
from library.model import Seq2Seq
from library.trainer import Trainer, generate_submission
from library.utils import seed_everything


def run_demo():
    print("=== Starting InChI Library Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up configuration...")
    seed_everything(Config.SEED)

    # Override Config for a quick demo run
    Config.DEBUG = True
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    # We will use a small subset of the data
    DEMO_SAMPLE_SIZE = 32

    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Tokenizer Demonstration
    print("\n[2] initializing Tokenizer...")
    tokenizer = Tokenizer()

    # Fit on texts (this will load from metadata/train_metadata.csv or cache)
    # Note: This might take a moment to read the full CSV if cache doesn't exist,
    # but it's necessary for the model vocabulary.
    tokenizer.fit_on_texts(load_cached_data=True)

    # Validation of Tokenizer
    test_inchi = "InChI=1S/H2O/h1H2"
    seq = tokenizer.text_to_sequence(test_inchi)
    decoded = tokenizer.sequence_to_text(seq)

    print(f"Original: {test_inchi}")
    print(f"Sequence: {seq[:10]}... (len={len(seq)})")
    print(f"Decoded:  {decoded}")

    # Basic assertion logic (decoded might differ slightly due to special tokens handling in sequence_to_text)
    # The tokenizer adds SOS/EOS and padding. sequence_to_text stops at EOS.
    assert "H2O" in decoded, "Tokenizer decoding failed to preserve core content."
    print("Tokenizer logic verified.")

    # 3. Dataset and DataLoader Demonstration
    print("\n[3] Preparing Datasets...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subset for speed
    train_subset = train_df.iloc[:DEMO_SAMPLE_SIZE].copy()
    val_subset = val_df.iloc[:DEMO_SAMPLE_SIZE].copy()
    test_subset = test_df.iloc[:DEMO_SAMPLE_SIZE].copy()

    print(f"Training on {len(train_subset)} samples.")

    # Instantiate Datasets
    train_dataset = InChiDataset(
        train_subset, tokenizer, transform=get_transforms("train"), mode="train"
    )
    val_dataset = InChiDataset(
        val_subset, tokenizer, transform=get_transforms("valid"), mode="valid"
    )
    test_dataset = InChiDataset(
        test_subset, tokenizer, transform=get_transforms("test"), mode="test"
    )

    # Verify Train Item
    img, label, seq_len = train_dataset[0]
    print(f"Train Image Shape: {img.shape}")
    print(f"Train Label Shape: {label.shape}")
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image shape"
    assert label.shape == (Config.MAX_TEXT_LEN,), "Incorrect label shape"

    # Verify Test Item
    img_test, img_id = test_dataset[0]
    assert isinstance(img_id, str), "Image ID should be a string"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("DataLoaders created.")

    # 4. Model Demonstration
    print("\n[4] Initializing Model...")
    model = Seq2Seq(tokenizer_len=len(tokenizer))
    model.to(Config.DEVICE)

    # Verify Forward Pass
    dummy_imgs, dummy_labels, _ = next(iter(train_loader))
    dummy_imgs = dummy_imgs.to(Config.DEVICE)
    dummy_labels = dummy_labels.to(Config.DEVICE)

    print("Running forward pass check...")
    outputs = model(dummy_imgs, dummy_labels)
    # Output shape: [Batch, Seq_Len, Vocab_Size]
    expected_shape = (Config.BATCH_SIZE, Config.MAX_TEXT_LEN, len(tokenizer))
    print(f"Output Shape: {outputs.shape}")

    assert (
        outputs.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {outputs.shape}"
    print("Forward pass successful.")

    # Verify Predict Method
    print("Running prediction check...")
    preds = model.predict(dummy_imgs, tokenizer)
    assert len(preds) == Config.BATCH_SIZE, "Prediction count mismatch"
    assert isinstance(preds[0], str), "Prediction should be a string"
    print(f"Sample Prediction: {preds[0]}")

    # 5. Trainer Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    trainer = Trainer(model, tokenizer, optimizer, scheduler)

    # Run fit
    # Config.DEBUG is True, so it will only run a few steps per epoch
    trainer.fit(train_loader, val_loader, epochs=1)
    print("Training loop completed.")

    # 6. Submission Demonstration
    print("\n[6] Generating Submission...")
    # Define a temporary output path for the demo
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(model, test_loader, tokenizer, output_path=demo_submission_path)

    if os.path.exists(demo_submission_path):
        sub_df = pd.read_csv(demo_submission_path)
        print(f"Submission generated with {len(sub_df)} rows.")
        print(sub_df.head())
        assert len(sub_df) == DEMO_SAMPLE_SIZE, "Submission row count mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
