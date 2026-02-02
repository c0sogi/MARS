import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_levenshtein
from library.tokenizer import InChITokenizer
from library.dataset import get_dataloaders
from library.model import ViT2InChI
from library.trainer import train_one_epoch, validate, predict_and_submit


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print(">>> Setting up environment...")
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Tokenizer Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Testing Tokenizer...")
    tokenizer = InChITokenizer()

    # Test string
    sample_inchi = "InChI=1S/H2O/h1H2"
    print(f"Original Text: {sample_inchi}")

    # Encode
    seq = tokenizer.text_to_sequence(sample_inchi)
    print(f"Encoded Sequence Shape: {seq.shape}")
    print(f"Encoded Sequence (first 10 tokens): {seq[:10]}")

    # Decode
    decoded_text = tokenizer.sequence_to_text(seq)
    print(f"Decoded Text: {decoded_text}")

    # Validations
    assert (
        decoded_text == sample_inchi
    ), f"Decoding mismatch: {decoded_text} != {sample_inchi}"
    assert seq[0] == Config.SOS_IDX, "Sequence must start with SOS token"
    # Note: The tokenizer pads to MAX_TEXT_LEN. We check if the token after the text is EOS.
    # Length of text tokens = len(sample_inchi). SOS is at 0. Text is at 1..len. EOS should be at len+1.
    assert (
        seq[len(sample_inchi) + 1] == Config.EOS_IDX
    ), "Sequence must end with EOS token after text"
    print("Tokenizer validation passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Loading Data (Debug Mode)...")
    # We use debug=True to load a tiny subset of the data for speed
    batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        debug=True,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for simple sequential debugging
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")  # Expected: (B, 3, 224, 224)
    print(f"Target batch shape: {targets.shape}")  # Expected: (B, 410)

    assert images.shape == (
        batch_size,
        Config.IN_CHANS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image batch shape"
    assert targets.shape == (
        batch_size,
        Config.MAX_TEXT_LEN,
    ), "Incorrect target batch shape"
    print("Data loading validation passed.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")
    model = ViT2InChI().to(device)

    # Prepare inputs for forward pass (Teacher Forcing)
    images = images.to(device)
    targets = targets.to(device)

    # In training, the decoder input is the target sequence without the last token (EOS/PAD)
    # The ground truth for loss calculation is the target sequence without the first token (SOS)
    input_ids = targets[:, :-1]

    print("Running forward pass...")
    logits = model(images, input_ids)
    print(f"Logits shape: {logits.shape}")  # Expected: (B, MAX_TEXT_LEN-1, VOCAB_SIZE)

    expected_seq_len = Config.MAX_TEXT_LEN - 1
    assert logits.shape == (
        batch_size,
        expected_seq_len,
        Config.VOCAB_SIZE,
    ), f"Logits shape mismatch. Expected {(batch_size, expected_seq_len, Config.VOCAB_SIZE)}, got {logits.shape}"
    print("Model forward pass validation passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Simulating Training Loop (1 Epoch)...")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for 1 epoch
    # Note: Since we are using debug loaders, this will be very fast
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=1,
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    print("Running Validation...")
    val_loss, val_score = validate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        tokenizer=tokenizer,
    )
    print(f"Epoch 1 Val Loss: {val_loss:.4f}")
    print(f"Epoch 1 Val Levenshtein Distance: {val_score:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Generating Submission (Test Set)...")

    submission_file = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=device,
        tokenizer=tokenizer,
        save_path=submission_file,
    )

    # Verify submission file
    if os.path.exists(submission_file):
        df_sub = pd.read_csv(submission_file)
        print(f"Submission saved to: {submission_file}")
        print(f"Submission shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())

        assert list(df_sub.columns) == [
            "image_id",
            "InChI",
        ], "Submission columns mismatch"
        assert len(df_sub) == len(
            test_loader.dataset
        ), "Submission length mismatch with test dataset"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    main()
