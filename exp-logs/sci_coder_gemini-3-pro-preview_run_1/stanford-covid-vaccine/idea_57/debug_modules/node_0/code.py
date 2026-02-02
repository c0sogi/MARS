import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import process_dataframe, RNADataset, get_dataloaders
from library.model import VectorScaledWideStreamBiGRU
from library.train import train_epoch, validate


def run_demo():
    print("Initializing Demo...")

    # =========================================================================
    # 1. Setup Environment and Overrides for Demo Speed
    # =========================================================================
    DEMO_DIR = "./demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    print(f"Created temporary demo directory: {DEMO_DIR}")

    # Override Config to use the demo directory and small settings
    # This limits the runtime significantly for demonstration purposes
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    # Define paths for mini datasets
    original_train_path = "./metadata/train.parquet"
    original_val_path = "./metadata/val.parquet"
    original_test_path = "./metadata/test.parquet"

    mini_train_path = os.path.join(DEMO_DIR, "mini_train.parquet")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.parquet")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.parquet")

    # Create mini datasets (10 samples each) to ensure speed
    print("Creating mini datasets...")
    pd.read_parquet(original_train_path).head(10).to_parquet(mini_train_path)
    pd.read_parquet(original_val_path).head(10).to_parquet(mini_val_path)
    pd.read_parquet(original_test_path).head(10).to_parquet(mini_test_path)

    # Point Config to these new mini datasets
    Config.TRAIN_DATA_PATH = mini_train_path
    Config.VAL_DATA_PATH = mini_val_path
    Config.TEST_DATA_PATH = mini_test_path

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Demonstrate Data Processing Logic
    # =========================================================================
    print("\n--- Testing Data Processing ---")

    # Load the mini dataframe manually to test process_dataframe
    df_mini = pd.read_parquet(mini_train_path)

    # Call process_dataframe
    # This function tokenizes sequences and parses structures into distance matrices
    X_seq, X_loop, X_dist, Y, ids = process_dataframe(df_mini, mode="train")

    # Verification of shapes
    # We expect 10 samples, sequence length 107
    assert len(X_seq) == 10, "Should have 10 sequences"
    assert X_seq.shape == (
        10,
        Config.SEQ_LENGTH,
    ), f"X_seq shape mismatch: {X_seq.shape}"
    assert X_loop.shape == (
        10,
        Config.SEQ_LENGTH,
    ), f"X_loop shape mismatch: {X_loop.shape}"
    assert X_dist.shape == (
        10,
        Config.SEQ_LENGTH,
    ), f"X_dist shape mismatch: {X_dist.shape}"
    # Y shape: (Samples, Scored_Length, Num_Targets) -> (10, 68, 3)
    assert Y.shape == (
        10,
        Config.SCORED_LENGTH,
        Config.NUM_TARGETS,
    ), f"Y shape mismatch: {Y.shape}"

    print("process_dataframe logic verified.")

    # =========================================================================
    # 3. Demonstrate Dataset and DataLoader
    # =========================================================================
    print("\n--- Testing Dataset and DataLoader ---")

    # Instantiate Dataset manually
    dataset = RNADataset(X_seq, X_loop, X_dist, Y, ids)

    # Fetch one item to verify __getitem__ logic
    item = dataset[0]

    # Verify Item Keys
    assert "seq" in item
    assert "loop" in item
    assert "pair_enc" in item
    assert "target" in item
    assert "mask" in item

    # Verify Tensor Shapes
    # seq: (L,)
    assert item["seq"].shape == (Config.SEQ_LENGTH,)
    # pair_enc: (L, Embed_Dim_Pair) - computed on the fly via sinusoidal encoding
    assert item["pair_enc"].shape == (Config.SEQ_LENGTH, Config.EMBED_DIM_PAIR)
    # target: (L, Num_Targets) - Note: Dataset pads targets from 68 to 107
    assert item["target"].shape == (Config.SEQ_LENGTH, Config.NUM_TARGETS)

    print("RNADataset item structure verified.")

    # Use the library function get_dataloaders to test the full loading pipeline
    # This will trigger the caching logic in get_dataloaders using our mini files
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    assert len(train_loader.dataset) == 10
    assert len(val_loader.dataset) == 10
    print("get_dataloaders pipeline verified.")

    # =========================================================================
    # 4. Demonstrate Model Architecture
    # =========================================================================
    print("\n--- Testing Model Architecture ---")

    device = Config.DEVICE
    model = VectorScaledWideStreamBiGRU().to(device)

    # Create a dummy batch from the loader
    batch = next(iter(train_loader))
    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    pair_enc = batch["pair_enc"].to(device)

    # Forward Pass
    preds = model(seq, loop, pair_enc)

    # Verify Output Shape: (Batch, Seq_Len, Num_Targets)
    # Note: train_loader drops last if not full batch, but 10 // 4 = 2 full batches.
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)

    assert (
        preds.shape == expected_shape
    ), f"Model output shape mismatch. Got {preds.shape}, expected {expected_shape}"
    print("Model forward pass verified.")

    # =========================================================================
    # 5. Demonstrate Metric Calculation
    # =========================================================================
    print("\n--- Testing MCRMSE Metric ---")

    # Case 1: Perfect prediction (Error should be 0.0)
    y_true_perfect = np.random.rand(5, Config.SCORED_LENGTH, 3)
    y_pred_perfect = y_true_perfect.copy()
    score_perfect = calculate_mcrmse(y_true_perfect, y_pred_perfect)
    assert np.isclose(score_perfect, 0.0), "Perfect prediction should have 0 error"

    # Case 2: Known error
    # Target 0: error 1.0 -> RMSE 1.0
    # Target 1: error 2.0 -> RMSE 2.0
    # Target 2: error 0.0 -> RMSE 0.0
    # Mean RMSE = (1+2+0)/3 = 1.0
    y_true_known = np.zeros((1, 1, 3))
    y_pred_known = np.array([[[1.0, 2.0, 0.0]]])
    score_known = calculate_mcrmse(y_true_known, y_pred_known)
    assert np.isclose(score_known, 1.0), f"Expected 1.0, got {score_known}"

    print("MCRMSE calculation verified.")

    # =========================================================================
    # 6. Demonstrate Training Loop (Integration)
    # =========================================================================
    print("\n--- Testing Training Loop ---")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    # Run one training epoch
    # This tests the gradient flow, loss calculation, and optimizer step
    initial_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch 1 Loss: {initial_loss:.4f}")

    assert not np.isnan(initial_loss), "Training loss is NaN"
    assert initial_loss > 0, "Training loss should be positive"

    # Run validation
    val_score = validate(model, val_loader, device)
    print(f"Validation MCRMSE: {val_score:.4f}")

    assert not np.isnan(val_score), "Validation score is NaN"

    # =========================================================================
    # 7. Demonstrate Prediction Generation (Submission Format)
    # =========================================================================
    print("\n--- Testing Prediction Generation ---")

    # We use the test loader to simulate generating a submission
    model.eval()
    test_preds = []
    test_ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_enc = batch["pair_enc"].to(device)
            ids_batch = batch["id"]  # List of strings

            preds = model(seq, loop, pair_enc)  # (B, 107, 3)

            # Move to CPU
            preds = preds.cpu().numpy()

            test_preds.append(preds)
            test_ids.extend(ids_batch)

    test_preds = np.concatenate(test_preds, axis=0)

    # Verify shapes
    # 10 test samples, 107 length, 3 targets
    assert test_preds.shape == (10, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    assert len(test_ids) == 10

    print("Prediction generation verified.")

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
