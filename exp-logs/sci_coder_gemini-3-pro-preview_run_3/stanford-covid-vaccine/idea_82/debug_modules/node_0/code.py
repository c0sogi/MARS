import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import set_seed, MCRMSELoss, parse_structure_to_adj
from library.data import get_dataloaders
from library.layers import RNAModel
from library.engine import train_fn, eval_fn, inference_fn
from library.model import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Override for Speed and Demo
    print("Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 100  # Small subset for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Verify Utility Logic (Adjacency Parsing)
    print("\n=== Verifying Utility Functions ===")
    test_structure = "..((...)).."
    # Expected: indices 0,1,5,6,9,10 are -1 (unpaired).
    # 2 pairs with 8, 3 pairs with 7.
    adj = parse_structure_to_adj(test_structure)

    assert len(adj) == len(test_structure), "Adjacency length mismatch"
    assert adj[2] == 8 and adj[8] == 2, "Pairing logic incorrect for outer bracket"
    assert adj[3] == 7 and adj[7] == 3, "Pairing logic incorrect for inner bracket"
    assert adj[0] == -1, "Unpaired logic incorrect"
    print("parse_structure_to_adj logic verified.")

    # 3. Data Pipeline
    print("\n=== Initializing Data Pipeline ===")
    # Force reprocessing (load_cached_data=False) to verify data processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    inputs = sample_batch["inputs"].to(device)
    neighbor_indices = sample_batch["neighbor_indices"].to(device)
    pair_masks = sample_batch["pair_masks"].to(device)
    targets = sample_batch["targets"].to(device)

    print(f"Batch shapes verified:")
    print(
        f"  Inputs: {inputs.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, {Config.INPUT_CHANNELS})"
    )
    print(
        f"  Targets: {targets.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, 5)"
    )

    assert inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.INPUT_CHANNELS)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, 5)
    assert neighbor_indices.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert pair_masks.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)

    # 4. Model Initialization & Forward Pass
    print("\n=== Initializing Model ===")
    model = RNAModel().to(device)

    print("Running forward pass check...")
    with torch.no_grad():
        outputs = model(inputs, neighbor_indices, pair_masks)

    print(f"  Output shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"
    print("Model forward pass verified.")

    # 5. Training Loop Execution
    print("\n=== Executing Training Loop ===")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = MCRMSELoss()

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )
        val_score = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.4f}"
        )

        assert np.isfinite(train_loss), "Training loss is not finite"
        assert np.isfinite(val_score), "Validation score is not finite"

    # Save dummy best model for inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("Training loop completed and model saved.")

    # 6. Inference & Submission
    print("\n=== Running Inference ===")
    # Load model state (simulating best model loading)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    preds, ids = inference_fn(model, test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    # Test set size in debug mode might be smaller or full depending on implementation of get_dataloaders
    # But shape should be (N_samples, 107, 5)
    assert len(preds.shape) == 3
    assert preds.shape[1] == Config.SEQ_LENGTH
    assert preds.shape[2] == Config.NUM_TARGETS
    assert len(ids) == preds.shape[0]

    print("Generating submission file...")
    generate_submission(preds, ids, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Column mismatch. Got {df_sub.columns}"

    # Check row count: N_samples * Seq_Length
    expected_rows = len(ids) * Config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
