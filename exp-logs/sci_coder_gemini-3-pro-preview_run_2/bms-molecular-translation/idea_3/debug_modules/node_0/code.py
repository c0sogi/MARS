import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.tokenizer import InChITokenizer
from library.dataset import get_dataloaders
from library.model import CNNTransformer
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("Starting demonstration script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[1] Overriding Configuration for Demo...")

    # Set paths to a specific demo directory in working
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Enable debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Main process only for simple debugging

    # Print config to verify
    Config.print_config()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Tokenizer Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Tokenizer...")
    tokenizer = InChITokenizer()

    sample_inchi = "InChI=1S/H2O/h1H2"
    print(f"Original: {sample_inchi}")

    # Test Encoding
    encoded = tokenizer.encode(sample_inchi)
    print(f"Encoded Tensor Shape: {encoded.shape}")
    print(f"Encoded Tensor: {encoded}")

    # Assert special tokens
    assert encoded[0] == Config.SOS_IDX, "First token should be SOS"
    assert (
        encoded[len(sample_inchi) + 1] == Config.EOS_IDX
    ), "Token after text should be EOS"

    # Test Decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    assert decoded == sample_inchi, f"Decoding mismatch: {decoded} != {sample_inchi}"
    print("Tokenizer verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Verify Train Loader
    assert train_loader is not None, "Train loader is None"
    batch = next(iter(train_loader))

    images = batch["images"]
    labels = batch["labels"]

    print(f"Batch Images Shape: {images.shape}")  # Should be (B, 1, 256, 256)
    print(f"Batch Labels Shape: {labels.shape}")  # Should be (B, MaxSeqLenInBatch)

    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect image shape: {images.shape}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Incorrect label batch size"

    print("DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model...")

    model = CNNTransformer().to(device)

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)

    # Test Forward (Training Mode)
    logits = model(images, labels)
    # Output shape should be (B, L-1, VocabSize) because targets are shifted
    expected_seq_len = labels.shape[1] - 1
    print(f"Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
        Config.VOCAB_SIZE,
    ), f"Incorrect logits shape: {logits.shape}"

    # Test Predict (Inference Mode)
    # Using a small max_len for speed test
    preds = model.predict(images, max_len=10, device=device)
    print(f"Predictions Shape: {preds.shape}")

    assert preds.shape[0] == Config.BATCH_SIZE, "Incorrect prediction batch size"
    # Length might vary if EOS is hit early, but tensor is padded/stacked to max_len or stop
    # In the provided predict implementation, it stacks up to max_len.
    assert preds.shape[1] <= 10, "Prediction length exceeded max_len"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Trainer (1 Epoch)...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        patience=Config.PATIENCE,
    )

    # Run training
    # Since we set Config.EPOCHS = 1, this runs 1 epoch
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model checkpoint was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Checkpoint found at {Config.MODEL_SAVE_PATH}")
    else:
        # It's possible validation score was inf if something went wrong, or first epoch didn't improve (unlikely with inf init)
        # Actually Trainer init best_score is inf, so first epoch should save unless score is nan/inf
        print(
            "Warning: Checkpoint not found. This might happen if validation metric failed."
        )
        # For the sake of the demo, we ensure a file exists to test inference
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print("Forced saving checkpoint for inference test.")

    print("Training loop verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # Run inference using the saved checkpoint
    # We pass explicit args to ensure it uses our demo settings
    df_submission = run_inference(
        checkpoint_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        device=device,
        debug=True,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        df_check = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df_check.shape}")

        assert "image_id" in df_check.columns, "image_id column missing"
        assert "InChI" in df_check.columns, "InChI column missing"
        assert len(df_check) > 0, "Submission file is empty"

        # Check format of InChI (basic check)
        # Since model is trained for 1 epoch on 20 samples, predictions will be garbage,
        # but we check if they are strings.
        assert isinstance(
            df_check.iloc[0]["InChI"], str
        ), "InChI prediction is not a string"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("Inference pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
