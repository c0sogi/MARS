import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import process_data, get_dataloaders
from library.model import DualViewResFunnel
from library.train import Trainer


def run_demonstration():
    print("================================================================")
    print("   Dual-View Post-Norm SwiGLU-ResFunnel Network Demonstration   ")
    print("================================================================")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment for Demo Run...")

    # Override Config for a fast, isolated execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"

    # Ensure paths are updated based on the new directories
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORKING_DIR, "processed_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Runtime optimizations for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.DEBUG = True  # Enable debug mode
    Config.DEBUG_SAMPLE_SIZE = 2048  # Use a small subset (2 batches of 1024)
    Config.BATCH_SIZE = 1024  # Large batch size for speed
    Config.NUM_WORKERS = 2  # Reduced workers for small data

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG} (Samples: {Config.DEBUG_SAMPLE_SIZE})")

    # --------------------------------------------------------------------------
    # 2. Data Processing Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Processing Data...")

    # Force reprocessing to verify the pipeline
    data_dict = process_data(load_cached_data=False)

    # Verify Data Shapes
    print("Verifying processed data shapes...")
    X_cat = data_dict["X_cat_train"]
    X_cont = data_dict["X_cont_train"]
    y = data_dict["y_train"]

    # Check dimensions
    # Categorical: (N, Sequence_Length)
    assert (
        X_cat.ndim == 2 and X_cat.shape[1] == Config.SEQUENCE_LENGTH
    ), f"Expected X_cat shape (_, {Config.SEQUENCE_LENGTH}), got {X_cat.shape}"

    # Continuous: (N, Num_Cont_Features)
    assert (
        X_cont.ndim == 2 and X_cont.shape[1] == Config.NUM_CONT_FEATURES
    ), f"Expected X_cont shape (_, {Config.NUM_CONT_FEATURES}), got {X_cont.shape}"

    # Target: (N,)
    assert y.ndim == 1, f"Expected y shape (N,), got {y.shape}"

    print("Data processing integrity check passed.")

    # --------------------------------------------------------------------------
    # 3. DataLoader Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Initializing DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Fetch one batch to verify
    cat_batch, cont_batch, target_batch = next(iter(train_loader))

    print(
        f"Batch Shapes -> Cat: {cat_batch.shape}, Cont: {cont_batch.shape}, Target: {target_batch.shape}"
    )

    assert cat_batch.shape == (Config.BATCH_SIZE, Config.SEQUENCE_LENGTH)
    assert cont_batch.shape == (Config.BATCH_SIZE, Config.NUM_CONT_FEATURES)
    assert target_batch.shape == (Config.BATCH_SIZE,)

    print("DataLoader verification passed.")

    # --------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Instantiating Model and Checking Forward Pass...")

    model = DualViewResFunnel(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        seq_len=Config.SEQUENCE_LENGTH,
        num_cont=Config.NUM_CONT_FEATURES,
        # Using default architecture params from Config
    ).to(Config.DEVICE)

    model.eval()
    with torch.no_grad():
        # Move inputs to device
        cat_in = cat_batch.to(Config.DEVICE)
        cont_in = cont_batch.to(Config.DEVICE)

        # Forward pass
        output = model(cat_in, cont_in)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output out of probability range [0, 1]"

    print("Model forward pass verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[Step 5] Executing Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Fit the model
    # This uses the modified Config (EPOCHS=1, DEBUG=True)
    test_loader_returned = trainer.fit()

    # Verify model checkpoint exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint was not saved at {Config.MODEL_SAVE_PATH}"
        )

    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 6. Inference Execution
    # --------------------------------------------------------------------------
    print("\n[Step 6] Running Inference...")

    trainer.predict(test_loader_returned)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not generated at {Config.SUBMISSION_PATH}"
        )

    # Load and check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # In debug mode, the submission should have DEBUG_SAMPLE_SIZE rows
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count {len(df_sub)} does not match debug sample size {Config.DEBUG_SAMPLE_SIZE}"

    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing"

    print("\n[Success] Demonstration completed successfully.")


if __name__ == "__main__":
    # Filter warnings to keep output clean
    warnings.filterwarnings("ignore")
    run_demonstration()
