import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import RNATokenizer, get_dataloaders, get_test_dataloader
from library.model import RNA_Net
from library.train import run_training
from library.inference import run_inference, predict, create_submission


def main():
    print("Initializing demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # We override the Config parameters to ensure the demo runs quickly.
    # Note: Default arguments in functions are evaluated at import time,
    # so we must pass these modified values explicitly to the functions later.
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Define a specific working directory for this execution to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "data_cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure clean working directory for demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Demonstrate RNATokenizer
    print("\n--- Testing RNATokenizer ---")
    tokenizer = RNATokenizer()
    # Create dummy sequence data (length 107)
    dummy_seq = "AGUC" * 20 + "AGUC"[:27]
    dummy_struct = "()." * 35 + "()."[:2]
    dummy_loop = "SMIBHEX" * 15 + "SM"

    tokenized = tokenizer.tokenize(dummy_seq, dummy_struct, dummy_loop)

    # Verification
    assert tokenized.shape == (
        107,
        3,
    ), f"Expected shape (107, 3), got {tokenized.shape}"
    assert tokenized.dtype == np.int64, "Expected dtype int64"
    print("Tokenizer output shape verified: (107, 3)")

    # 3. Demonstrate Data Loading
    print("\n--- Testing Data Loading ---")
    # This will trigger preprocessing and caching since the cache dir is new.
    # We pass batch_size explicitly.
    train_loader, val_loader = get_dataloaders(
        load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch
    inputs, targets = next(iter(train_loader))

    # Verification
    # Inputs: (Batch, Seq_Len, 3)
    # Targets: (Batch, Seq_Len, 5)
    assert inputs.ndim == 3
    assert inputs.shape[1] == Config.SEQ_LEN
    assert inputs.shape[2] == 3
    assert targets.ndim == 3
    assert targets.shape[1] == Config.SEQ_LEN
    assert targets.shape[2] == Config.NUM_TARGETS

    print(f"Train batch inputs shape: {inputs.shape}")
    print(f"Train batch targets shape: {targets.shape}")

    # 4. Demonstrate Model Architecture
    print("\n--- Testing Model Architecture ---")
    model = RNA_Net().to(device)

    # Forward pass with the fetched batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)

    # Verification
    # Output shape should be (Batch, Seq_Len, Num_Targets)
    assert (
        outputs.shape == targets.shape
    ), f"Output shape {outputs.shape} mismatch with targets {targets.shape}"
    print(f"Model output shape verified: {outputs.shape}")

    # 5. Demonstrate Loss Function
    print("\n--- Testing Loss Function ---")
    criterion = MCRMSELoss()
    loss = criterion(outputs, targets)

    # Verification
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Calculated MCRMSE Loss: {loss.item():.4f}")

    # 6. Demonstrate Training Loop
    print("\n--- Running Training Loop (1 Epoch) ---")
    # We use the run_training function from library.train
    # We must pass the modified Config values explicitly
    run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file was not saved at {Config.MODEL_PATH}"
    print("Training completed and model saved.")

    # 7. Demonstrate Inference
    print("\n--- Running Inference ---")
    # Run inference using the saved model
    run_inference(
        model_path=Config.MODEL_PATH,
        output_path=Config.SUBMISSION_FILE,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Use cache if available (test data processed inside function)
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check submission format
    # Expected rows: Number of test samples * Seq_Len
    # Test set has 240 samples. 240 * 107 = 25680 rows.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Columns mismatch. Got {list(df_sub.columns)}"

    # Check content of first ID
    first_id_seqpos = df_sub.iloc[0]["id_seqpos"]
    assert "_0" in first_id_seqpos, "First row ID should end with _0"

    print("Submission format verified.")
    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
