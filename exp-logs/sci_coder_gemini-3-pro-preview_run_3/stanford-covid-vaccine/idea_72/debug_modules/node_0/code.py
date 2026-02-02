import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_val_metric
from library.data import get_dataloaders, get_structure_adj
from library.model import RNAModel
from library.engine import train_fn, eval_fn, predict_fn, generate_submission


# ==========================================
# 1. Configuration for Fast Demonstration
# ==========================================
class DemoConfig(Config):
    """
    Override standard configuration to run a lightweight, fast demonstration.
    """

    # Reduced Model Complexity
    CNN_FILTERS = 16
    HIDDEN_DIM = 32
    NUM_LAYERS = 1
    FFN_DIM = 64
    DROPOUT = 0.0

    # Reduced Training Parameters
    BATCH_SIZE = 16
    EPOCHS = 1
    LEARNING_RATE = 1e-3

    # Paths (Redirect output to working directory)
    WORKING_DIR = "./working/demo_execution"
    SUBMISSION_DIR = "./working/demo_execution"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_demo.csv")

    # Ensure cache uses the demo working dir to avoid conflicts
    @staticmethod
    def setup_directories():
        os.makedirs(DemoConfig.WORKING_DIR, exist_ok=True)
        os.makedirs(DemoConfig.SUBMISSION_DIR, exist_ok=True)
        # Patch the Config.WORKING_DIR used by data.py caching
        Config.WORKING_DIR = os.path.join(DemoConfig.WORKING_DIR, "cache")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)


def run_demo():
    print("=== Starting RNA Degradation Model Demonstration ===")

    # Setup
    set_seed(DemoConfig.SEED)
    DemoConfig.setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================
    # 2. Verify Data Loading Logic
    # ==========================================
    print("\n[1/5] Verifying Data Loading...")

    # Load dataloaders
    # We use the provided get_dataloaders function.
    # Note: This reads from metadata/train.parquet, etc.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=DemoConfig.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple script execution
        load_cached_data=False,  # Force re-processing to verify logic
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    bpp_indices = batch["bpp_indices"]
    bpp_masks = batch["bpp_masks"]
    targets = batch["targets"]

    # Assertions
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        DemoConfig.BATCH_SIZE,
        107,
        14,
    ), f"Input shape mismatch: {inputs.shape}"

    # BPP Indices: (Batch, Seq_Len=107)
    assert bpp_indices.shape == (
        DemoConfig.BATCH_SIZE,
        107,
    ), f"BPP Indices shape mismatch: {bpp_indices.shape}"

    # Targets: (Batch, Pred_Len=68, Num_Targets=5)
    assert targets.shape == (
        DemoConfig.BATCH_SIZE,
        68,
        5,
    ), f"Targets shape mismatch: {targets.shape}"

    print("Data shapes verified successfully.")

    # Verify Structure Adjacency Logic explicitly
    # Test on a simple structure: "(...)" -> 0 pairs with 4
    test_struct = "(...)"
    adj, mask = get_structure_adj(test_struct)
    assert adj[0] == 4 and adj[4] == 0, "Adjacency logic failed for paired bases"
    assert adj[1] == 1, "Adjacency logic failed for unpaired base (should be self-loop)"
    assert mask[0] == 1.0 and mask[1] == 0.0, "Mask logic failed"
    print("Structure parsing logic verified.")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[2/5] Verifying Model Architecture...")

    model = RNAModel(config=DemoConfig).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    bpp_indices = bpp_indices.to(device)
    bpp_masks = bpp_masks.to(device)
    targets = targets.to(device)

    # Forward Pass
    outputs = model(inputs, bpp_indices, bpp_masks)

    # Output shape should be (Batch, 107, 5)
    # Even though we only score the first 68, the model outputs for full sequence length
    assert outputs.shape == (
        DemoConfig.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape mismatch: {outputs.shape}"

    print("Model forward pass verified.")

    # ==========================================
    # 4. Verify Loss Function
    # ==========================================
    print("\n[3/5] Verifying MCRMSE Loss...")

    criterion = MCRMSELoss()
    loss = criterion(outputs, targets)

    # Loss should be a scalar
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"Loss calculation verified. Initial Loss: {loss.item():.4f}")

    # ==========================================
    # 5. Run Training Loop (1 Epoch)
    # ==========================================
    print("\n[4/5] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=DemoConfig.LEARNING_RATE)

    # Run training function
    # train_fn iterates the whole loader. Since we used the full parquet,
    # it has 1728 samples / 16 batch = 108 steps. This is fast enough for demo.
    epoch_loss = train_fn(model, train_loader, optimizer, criterion, device)
    print(f"Train Loop Completed. Avg Loss: {epoch_loss:.4f}")

    # Run validation function
    val_score = eval_fn(model, val_loader, device)
    print(f"Validation Loop Completed. MCRMSE Score: {val_score:.4f}")

    # Save model (simulating checkpointing)
    model_path = os.path.join(DemoConfig.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\n[5/5] Running Inference & Generating Submission...")

    # Load model
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    # Predict
    test_preds = predict_fn(model, test_loader, device)

    # Verify predictions dictionary
    assert len(test_preds) > 0, "No predictions generated"
    sample_id = list(test_preds.keys())[0]
    assert test_preds[sample_id].shape == (
        107,
        5,
    ), f"Prediction shape mismatch: {test_preds[sample_id].shape}"

    # Generate CSV
    generate_submission(test_preds, DemoConfig.SUBMISSION_PATH)

    # Verify file existence
    assert os.path.exists(DemoConfig.SUBMISSION_PATH), "Submission file not created"

    # Verify file content format
    df_sub = pd.read_csv(DemoConfig.SUBMISSION_PATH)
    expected_cols = ["id_seqpos"] + DemoConfig.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Submission generated successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
