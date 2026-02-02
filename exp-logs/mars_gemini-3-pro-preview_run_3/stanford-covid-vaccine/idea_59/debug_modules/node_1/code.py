import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse_loss, mcrmse_metric
from library.data import process_data, RNADataset, SEQ_MAP, STRUCT_MAP, LOOP_MAP
from library.model import HC_DBR_BiGRU
from library.train import train_one_epoch, validate, generate_submission


def run_demo():
    print("==== RNA Degradation Prediction Demo ====")

    # 1. Setup and Configuration Override
    # We override config parameters to make this run fast for demonstration purposes.
    seed_everything(Config.SEED)

    # Create a specific working directory for this demo
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and params
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading (Subset for Speed)
    print("\n[Step 1] Loading and Processing Data Subsets...")

    # Load raw metadata
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA}"
        )

    full_train_df = pd.read_parquet(Config.TRAIN_METADATA)
    full_test_df = pd.read_parquet(Config.TEST_METADATA)

    # Take small subsets: 20 for train, 10 for val, 20 for test
    demo_train_df = full_train_df.iloc[:20].reset_index(drop=True)
    demo_val_df = full_train_df.iloc[20:30].reset_index(drop=True)
    demo_test_df = full_test_df.iloc[:20].reset_index(drop=True)

    print(f"  Train subset shape: {demo_train_df.shape}")
    print(f"  Val subset shape:   {demo_val_df.shape}")
    print(f"  Test subset shape:  {demo_test_df.shape}")

    # Process data using library function
    # This converts text sequences to one-hot tensors and builds graph adjacency
    train_data = process_data(demo_train_df, mode="train")
    val_data = process_data(demo_val_df, mode="val")
    test_data = process_data(demo_test_df, mode="test")

    # Verification of Data Processing
    # Features shape: (N, 107, 14)
    assert train_data["features"].shape == (
        20,
        107,
        14,
    ), "Train features shape mismatch"
    # Targets shape: (N, 68, 5)
    assert train_data["targets"].shape == (20, 68, 5), "Train targets shape mismatch"
    # Pair indices shape: (N, 107)
    assert train_data["pair_indices"].shape == (
        20,
        107,
    ), "Train pair_indices shape mismatch"

    print("  Data processing verified successfully.")

    # Create Datasets and Loaders
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Initialization and Logic Check
    print("\n[Step 2] Initializing Model and Verifying Forward Pass...")

    model = HC_DBR_BiGRU().to(device)

    # Fetch one batch to verify
    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    targets = batch["target"].to(device)

    # Forward pass
    outputs = model(features, pair_indices, pair_mask)

    # Check Output Shape: (Batch, Seq_Len, Num_Targets) -> (4, 107, 5)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Got {outputs.shape}, expected {expected_shape}"

    # Check Loss Calculation
    loss = mcrmse_loss(outputs, targets)
    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"  Forward pass successful. Initial Loss: {loss.item():.4f}")

    # 4. Training Loop Demo
    print("\n[Step 3] Running Training Loop (2 Epochs)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metric = validate(model, val_loader, device)
        print(
            f"  Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val MCRMSE = {val_metric:.4f}"
        )

    # Save the "best" model (just the current one for demo)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("  Model saved.")

    # 5. Inference and Submission Generation
    print("\n[Step 4] Generating Submission...")

    # Reload model to verify loading logic
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission file loaded. Shape: {sub_df.shape}")

    # Expected rows: Num_Test_Samples * Seq_Len = 20 * 107 = 2140
    expected_rows = len(demo_test_df) * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Got {len(sub_df)}, expected {expected_rows}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check ID format (id_xxxxx_0)
    sample_id_seqpos = sub_df.iloc[0]["id_seqpos"]
    assert "_0" in sample_id_seqpos, "id_seqpos format seems incorrect"

    print("  Submission format verified successfully.")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
