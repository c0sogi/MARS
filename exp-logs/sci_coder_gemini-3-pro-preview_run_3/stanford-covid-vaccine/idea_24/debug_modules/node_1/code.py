import os
import shutil
import torch
import numpy as np
import pandas as pd
import time

# Import from the provided library
from library.config import Config
from library.utils import set_seed, mcrmse_loss, get_scored_metrics
from library.dataset import get_loader
from library.model import RNAModel
from library.train import run_training, generate_submission, train_epoch, validate


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # ------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = DEMO_DIR  # Cache locally to demo dir
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce training intensity for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Use 0 for simple debugging/demo stability

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")

    # ------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Load Train Loader
    # This will trigger preprocessing and caching in the demo directory
    train_loader = get_loader(
        "train", shuffle=True, load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch
    batch = next(iter(train_loader))

    inputs = batch["sequence"]
    adj = batch["adjacency"]
    mask = batch["mask"]
    targets = batch["target"]

    # Assertions for shapes
    # Inputs: (Batch, Seq_Len, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {inputs.shape}"

    # Adjacency: (Batch, Seq_Len)
    assert adj.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Adjacency shape mismatch: {adj.shape}"

    # Targets: (Batch, Seq_Scored, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch: {targets.shape}"

    print("    Data shapes verified successfully.")
    print(f"    Input: {inputs.shape}, Target: {targets.shape}")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = RNAModel(Config).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    adj = adj.to(device)
    mask = mask.to(device)
    targets = targets.to(device)

    # Forward Pass
    preds = model(inputs, adj, mask)

    # Check Output Shape: (Batch, Seq_Len, Num_Targets)
    # Note: Model outputs predictions for full length (107), even though we only score 68
    expected_out_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        preds.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {preds.shape}"

    print("    Forward pass successful.")
    print(f"    Output shape: {preds.shape}")

    # ------------------------------------------------------------------------
    # 4. Loss Function Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Loss Calculation...")

    # Slice predictions to match scored length for loss calculation
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]

    # Calculate MCRMSE Loss
    loss = mcrmse_loss(targets, preds_sliced)

    # Calculate Scored Metrics (subset of columns)
    score = get_scored_metrics(targets, preds_sliced)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert isinstance(score, float), "Score should be a float"

    print(f"    Loss: {loss.item():.4f}")
    print(f"    Score (Scored Columns): {score:.4f}")

    # ------------------------------------------------------------------------
    # 5. Training Loop Execution
    # ------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # We use the provided run_training function which handles the loop, validation, and saving
    # Since we set Config.EPOCHS = 1, this will run quickly.
    start_time = time.time()
    run_training()
    end_time = time.time()

    print(f"    Training finished in {end_time - start_time:.2f} seconds.")

    # Verify model checkpoint was created
    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint not found after training."
    print(f"    Checkpoint confirmed at: {Config.MODEL_PATH}")

    # ------------------------------------------------------------------------
    # 6. Inference & Submission Verification
    # ------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Expected rows: 240 test samples * 107 positions = 25680
    # Expected columns: id_seqpos + 5 targets = 6
    expected_rows = 240 * 107
    expected_cols = 6

    assert (
        df_sub.shape[0] == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {df_sub.shape[0]}"
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Submission column count mismatch. Expected {expected_cols}, got {df_sub.shape[1]}"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("    Submission format verified.")

    # ------------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------------
    print("\n[7] Cleaning up...")
    # Optional: Remove the demo directory to save space
    # shutil.rmtree(DEMO_DIR)
    print(f"    Demo artifacts remain in {DEMO_DIR}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
