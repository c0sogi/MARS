import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.dataset import get_dataloaders, set_seed
from library.model import GatedStemHybridNet
from library.trainer import Trainer


def main():
    # 1. Configuration and Setup
    print("Initializing configuration...")
    # Use a specific working directory for this execution
    WORKING_DIR = "./working/demo_execution"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Hyperparameters optimized for a quick demonstration
    BATCH_SIZE = 256
    MAX_SAMPLES = 5000  # Subsample to ensure quick runtime
    EPOCHS = 2
    LEARNING_RATE = 1e-3

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility
    set_seed(42)

    # 2. Data Loading
    print("\n--- Step 1: Loading Data ---")
    # We use a specific cache directory to avoid conflicts
    cache_dir = os.path.join(WORKING_DIR, "cache")

    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE,
        load_cached_data=False,  # Force processing to demonstrate the pipeline
        max_samples=MAX_SAMPLES,
        data_dir="./input",
        metadata_dir="./metadata",
        cache_dir=cache_dir,
    )

    # Verification: Check Data Loaders
    print("Verifying data loaders...")
    try:
        # Fetch one batch to verify shapes
        x_cont_batch, x_seq_batch, y_batch = next(iter(train_loader))

        # Continuous data should be (Batch, 30)
        assert x_cont_batch.shape == (
            BATCH_SIZE,
            30,
        ), f"Expected continuous shape ({BATCH_SIZE}, 30), got {x_cont_batch.shape}"

        # Sequence data should be (Batch, 10)
        assert x_seq_batch.shape == (
            BATCH_SIZE,
            10,
        ), f"Expected sequence shape ({BATCH_SIZE}, 10), got {x_seq_batch.shape}"

        # Targets should be (Batch,)
        assert y_batch.shape == (
            BATCH_SIZE,
        ), f"Expected target shape ({BATCH_SIZE},), got {y_batch.shape}"

        print(f"Data verification passed. Train batches: {len(train_loader)}")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Model Initialization
    print("\n--- Step 2: Initializing Model ---")
    model = GatedStemHybridNet(
        continuous_dim=30,
        vocab_size=30,  # Safe upper bound for 'A'-'Z'
        seq_len=10,
        embed_dim=32,
        backbone_dropout=0.2,
    )
    model.to(device)

    # Verification: Forward Pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        dummy_cont = x_cont_batch.to(device)
        dummy_seq = x_seq_batch.to(device)
        output = model(dummy_cont, dummy_seq)

        assert output.shape == (
            BATCH_SIZE,
            1,
        ), f"Expected output shape ({BATCH_SIZE}, 1), got {output.shape}"
    print("Model forward pass successful.")

    # 4. Training
    print("\n--- Step 3: Training Loop ---")
    trainer = Trainer(
        model=model, device=device, learning_rate=LEARNING_RATE, step_size=1, gamma=0.5
    )

    # Fit the model
    # This will save 'best_model.pth' in the specified directory
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, checkpoint_dir=WORKING_DIR)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not created at {checkpoint_path}")
    print(f"Training finished. Checkpoint verified at {checkpoint_path}")

    # 5. Prediction
    print("\n--- Step 4: Inference ---")
    # Predict using the best model found during training
    predictions = trainer.predict(test_loader, checkpoint_path=checkpoint_path)

    # Verification: Prediction Shape
    # Note: test_loader might drop the last batch if configured, but here drop_last is default False for test
    # However, get_dataloaders doesn't subsample test set based on max_samples logic for training,
    # but let's check the length against test_ids.
    assert len(predictions) == len(
        test_ids
    ), f"Mismatch: {len(predictions)} predictions for {len(test_ids)} test IDs."

    print(f"Generated {len(predictions)} predictions.")

    # 6. Submission Generation
    print("\n--- Step 5: Generating Submission ---")
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    # Final check
    if os.path.exists(submission_path):
        print("\nSUCCESS: Pipeline completed and submission file generated.")
    else:
        raise FileNotFoundError("Failed to generate submission file.")


if __name__ == "__main__":
    main()
