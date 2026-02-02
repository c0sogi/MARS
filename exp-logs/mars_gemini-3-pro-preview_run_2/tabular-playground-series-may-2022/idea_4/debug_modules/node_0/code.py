import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import GatedWideMLP
from library.engine import train_model, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Ensure directories exist
    Config.setup()

    print(f"Device: {Config.DEVICE}")
    print("Step 1: Setup complete.")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\nStep 2: Loading Data...")

    # Load data loaders
    # We use the default batch size from Config, but could override if needed
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))

    # Check keys
    required_keys = {"continuous", "tokens", "target"}
    assert required_keys.issubset(
        sample_batch.keys()
    ), f"Batch missing keys. Expected {required_keys}, got {sample_batch.keys()}"

    # Check dimensions
    # Continuous: (Batch, 30)
    # Tokens: (Batch, 10)
    # Target: (Batch,)
    cont_shape = sample_batch["continuous"].shape
    tok_shape = sample_batch["tokens"].shape
    target_shape = sample_batch["target"].shape

    assert cont_shape[1] == 30, f"Expected 30 continuous features, got {cont_shape[1]}"
    assert tok_shape[1] == 10, f"Expected 10 token features, got {tok_shape[1]}"
    assert len(target_shape) == 1, f"Targets should be 1D, got {target_shape}"

    print(f"Data loaded successfully. Vocab size: {vocab_size}")
    print(f"Batch shapes verified: Continuous {cont_shape}, Tokens {tok_shape}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # --------------------------------------------------------------------------
    print("\nStep 3: Initializing Model...")

    model = GatedWideMLP(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT,
    ).to(Config.DEVICE)

    # Verify Model Output Shape with a dummy forward pass
    model.eval()
    with torch.no_grad():
        # Move sample batch to device
        dummy_cont = sample_batch["continuous"].to(Config.DEVICE)
        dummy_tok = sample_batch["tokens"].to(Config.DEVICE)

        output = model(dummy_cont, dummy_tok)

        # Output should be (Batch, 1) and values between 0 and 1 (Sigmoid)
        assert output.shape == (
            dummy_cont.size(0),
            1,
        ), f"Model output shape mismatch. Expected {(dummy_cont.size(0), 1)}, got {output.shape}"
        assert (
            output.min() >= 0 and output.max() <= 1
        ), "Model output values out of probability range [0, 1]"

    print("Model initialized and forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\nStep 4: Starting Training...")

    # Define Optimizer
    # Using AdamW as is common for modern MLP architectures
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run training
    # Limiting to 2 epochs for demonstration speed
    DEMO_EPOCHS = 2

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        epochs=DEMO_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    # Verify model file was created
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved after training."
    print(f"Training finished. Model saved to {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission
    # --------------------------------------------------------------------------
    print("\nStep 5: Generating Predictions...")

    # Load best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # Predict
    predictions = predict(model, test_loader, Config.DEVICE)

    # Verify predictions
    expected_test_samples = 100000
    assert (
        len(predictions) == expected_test_samples
    ), f"Prediction count mismatch. Expected {expected_test_samples}, got {len(predictions)}"

    # Create Submission DataFrame
    print("Creating submission file...")
    submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    submission[Config.TARGET_COL] = predictions

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
