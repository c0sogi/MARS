import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import process_data, get_dataloaders
from library.model import SustainedDepthHybridNet, PreActGLUBlock, TransformerStream
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    print("\n[Step 1] Setting up environment and overriding config for speed...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters to make this demo fast
    # We use a separate cache directory to avoid conflicts with other runs
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_run")
    Config.PROCESSED_DATA_PATH = os.path.join(
        Config.CACHE_DIR, "processed_data_demo.npz"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.CACHE_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.CACHE_DIR, "submission_demo.csv")

    # Ensure the directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Speed optimizations
    Config.BATCH_SIZE = 4096  # Larger batch size for faster iteration on A100
    Config.EPOCHS = 1  # Only run 1 epoch
    Config.NUM_WORKERS = 2  # Reduce overhead

    print(f"Working Directory: {Config.CACHE_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Processing and Loading...")

    # Test process_data (this creates the npz file)
    # We force reload to demonstrate processing logic
    data_dict = process_data(load_cached_data=False)

    # Assertions to verify data structure
    assert "train_cont" in data_dict, "Missing train_cont in processed data"
    assert "train_seq" in data_dict, "Missing train_seq in processed data"
    assert "train_target" in data_dict, "Missing train_target in processed data"

    # Check shapes
    n_train = data_dict["train_cont"].shape[0]
    print(f"Training samples: {n_train}")

    assert data_dict["train_cont"].shape == (
        n_train,
        Config.NUM_CONT_FEATURES,
    ), f"Incorrect continuous feature shape: {data_dict['train_cont'].shape}"
    assert data_dict["train_seq"].shape == (
        n_train,
        Config.SEQ_LEN,
    ), f"Incorrect sequence feature shape: {data_dict['train_seq'].shape}"

    # Test DataLoader generation
    loaders = get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True)
    train_loader = loaders["train"]

    # Fetch one batch to verify tensor properties
    batch = next(iter(train_loader))
    cont_batch = batch["cont"]
    seq_batch = batch["seq"]
    target_batch = batch["target"]

    print(
        f"Batch shapes - Cont: {cont_batch.shape}, Seq: {seq_batch.shape}, Target: {target_batch.shape}"
    )

    assert cont_batch.shape == (Config.BATCH_SIZE, Config.NUM_CONT_FEATURES)
    assert seq_batch.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert target_batch.shape == (Config.BATCH_SIZE,)
    assert cont_batch.dtype == torch.float32
    assert seq_batch.dtype == torch.long

    print("Data pipeline verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Component Verification (Unit Tests)
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Components...")

    device = Config.DEVICE

    # A. Test PreActGLUBlock
    # Case 1: Same dimensions (Identity skip)
    block_identity = PreActGLUBlock(
        in_features=32, out_features=32, dropout_rate=0.1
    ).to(device)
    dummy_input = torch.randn(10, 32).to(device)
    out = block_identity(dummy_input)
    assert out.shape == (
        10,
        32,
    ), f"PreActGLUBlock (Identity) output mismatch: {out.shape}"

    # Case 2: Dimension change (Projected skip)
    block_proj = PreActGLUBlock(in_features=32, out_features=64, dropout_rate=0.1).to(
        device
    )
    out = block_proj(dummy_input)
    assert out.shape == (
        10,
        64,
    ), f"PreActGLUBlock (Projection) output mismatch: {out.shape}"

    print("PreActGLUBlock verified.")

    # B. Test TransformerStream
    transformer = TransformerStream(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=16,
        seq_len=Config.SEQ_LEN,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        activation="relu",
    ).to(device)

    # Input: (Batch, Seq_Len)
    dummy_seq = torch.randint(0, Config.VOCAB_SIZE, (5, Config.SEQ_LEN)).to(device)
    out_trans = transformer(dummy_seq)

    # Expected output: (Batch, Seq_Len * Embed_Dim) -> (5, 10 * 16) = (5, 160)
    expected_dim = Config.SEQ_LEN * 16
    assert out_trans.shape == (
        5,
        expected_dim,
    ), f"TransformerStream output mismatch. Got {out_trans.shape}, expected (5, {expected_dim})"

    print("TransformerStream verified.")

    # --------------------------------------------------------------------------
    # 4. Full Model Integration Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Full Model Integration...")

    model = SustainedDepthHybridNet().to(device)

    # Use the real batch fetched earlier
    cont_batch = cont_batch.to(device)
    seq_batch = seq_batch.to(device)

    # Forward pass
    logits = model(cont_batch, seq_batch)

    # Check output shape (Batch_Size,)
    assert logits.shape == (
        Config.BATCH_SIZE,
    ), f"Model output shape mismatch. Got {logits.shape}, expected ({Config.BATCH_SIZE},)"

    # Check Backward pass (Gradient flow)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, target_batch.to(device))
    loss.backward()

    # Check if gradients exist for a parameter (e.g., head)
    assert model.head.weight.grad is not None, "Gradients not computed for model head."

    print("Full model forward/backward pass verified.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[Step 5] Executing Training Loop (1 Epoch)...")

    # Run the training function provided in the library
    # We use the parameters overridden in Config
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        device=Config.DEVICE,
        save_path=Config.MODEL_SAVE_PATH,
        submission_path=Config.SUBMISSION_PATH,
    )

    # Verify outputs exist
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Should have 100,000 rows + header
    assert (
        len(sub_df) == 100000
    ), f"Submission row count mismatch. Expected 100000, got {len(sub_df)}"
    assert list(sub_df.columns) == [
        "id",
        "target",
    ], f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check value range
    assert (
        sub_df["target"].min() >= 0.0 and sub_df["target"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
