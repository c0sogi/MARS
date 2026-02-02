import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import MCSDBiGRU
from library.engine import train_fn, eval_fn


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("Starting RNA Degradation Prediction Demo...")

    # 1. Setup and Configuration Override
    set_seed(Config.SEED)

    # Define a demo-specific working directory
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Reduce compute requirements for demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = (
        0  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
    )

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("\nLoading data subsets...")
    # Load only 40 samples to ensure speed
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing from scratch for demo
        max_samples=40,
    )

    # Verify Data Integrity
    print("Verifying data integrity...")
    sample_batch = next(iter(train_loader))
    features = sample_batch["features"]
    pair_indices = sample_batch["pair_indices"]
    targets = sample_batch["targets"]

    # Check shapes
    # Features: (B, L, 14)
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Incorrect feature shape: {features.shape}"
    # Pair Indices: (B, L)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect pair_indices shape: {pair_indices.shape}"
    # Targets: (B, L, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Incorrect targets shape: {targets.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization
    print("\nInitializing MCSDBiGRU model...")
    device = torch.device(Config.DEVICE)
    model = MCSDBiGRU().to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = model(features.to(device), pair_indices.to(device))
        assert dummy_out.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            Config.NUM_TARGETS,
        ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {dummy_out.shape}"
    print("Model forward pass verified.")

    # 4. Training Loop
    print("\nStarting training loop...")
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, device)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss (MCRMSE): {train_loss:.4f}"
        )

        # Simple validation check
        val_metric = eval_fn(model, val_loader, device)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Val MCRMSE: {val_metric:.4f}")

    # Save the model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

    # 5. Inference and Submission Generation
    print("\nRunning inference on test set...")
    model.eval()

    test_ids = []
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            f = batch["features"].to(device)
            p = batch["pair_indices"].to(device)
            ids = batch["id"]  # List of IDs

            outputs = model(f, p)  # (B, SeqLen, 5)

            test_preds.append(outputs.cpu().numpy())
            test_ids.extend(ids)

    test_preds = np.concatenate(test_preds, axis=0)  # (N_samples, SeqLen, 5)

    print(f"Inference complete. Predictions shape: {test_preds.shape}")

    # 6. Formatting Submission
    print("Generating submission file...")

    # We need to flatten the predictions: one row per sequence position
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_pred = test_preds[i]  # (SeqLen, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_pred[seq_pos].tolist()

            # Create dictionary for DataFrame
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Verify submission format
    expected_cols = ["id_seqpos"] + target_cols
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"
    assert (
        len(submission_df) == len(test_ids) * Config.SEQ_LEN
    ), "Submission row count mismatch"

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission file saved to {Config.SUBMISSION_PATH}")
    print(f"Submission head:\n{submission_df.head(3)}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
