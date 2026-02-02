import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss, mcrmse_metric
from library.layers import StabilizedGLUInteraction
from library.model import HighCapacityBiGRU
from library.data import get_dataloaders
from library.train import train_epoch, validate, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # =========================================================================
    # 1. Patch Configuration for Speed and Isolation
    # =========================================================================
    # We modify the Config class attributes directly to run a fast demo on a small subset.
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up demo configuration in {demo_dir}")

    # Enable Debug mode to use only a small subset of data (20 samples)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20

    # Redirect paths to the demo directory
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Update cache paths to use the demo directory to avoid conflicts
    Config.TRAIN_CACHE_PATH = os.path.join(demo_dir, "train_cache.npy")
    Config.VAL_CACHE_PATH = os.path.join(demo_dir, "val_cache.npy")
    Config.TEST_CACHE_PATH = os.path.join(demo_dir, "test_cache.npy")

    # Reduce training parameters for immediate completion
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set Seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device("cpu")  # Use CPU for deterministic demo behavior
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Demonstrate Data Loading
    # =========================================================================
    print("\n--- Testing Data Loading ---")
    # This will trigger processing of the first 20 rows of metadata and caching
    # load_cached_data=False forces reprocessing to ensure we use the debug subset
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch from training loader
    features, pair_indices, pair_masks, targets = next(iter(train_loader))

    print(f"Feature shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

    # Assertions to verify data shapes
    # Features: (Batch, Seq_Len, Input_Channels)
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Expected feature shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_CHANNELS)}, got {features.shape}"

    # Targets: (Batch, Seq_Len, Num_Targets)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Expected target shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {targets.shape}"

    assert pair_indices.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert pair_masks.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)

    print("Data Loading verification passed.")

    # =========================================================================
    # 3. Demonstrate Layer Logic (StabilizedGLUInteraction)
    # =========================================================================
    print("\n--- Testing StabilizedGLUInteraction Layer ---")
    hidden_dim = 32
    layer = StabilizedGLUInteraction(hidden_dim=hidden_dim, dropout=0.0)

    # Create dummy inputs
    dummy_x = torch.randn(Config.BATCH_SIZE, Config.SEQ_LEN, hidden_dim)
    dummy_indices = pair_indices  # Reuse from loader
    dummy_mask = pair_masks  # Reuse from loader

    # Forward pass
    layer_out = layer(dummy_x, dummy_indices, dummy_mask)

    print(f"Layer output shape: {layer_out.shape}")

    # Assertions
    assert layer_out.shape == dummy_x.shape, "Layer output shape mismatch"
    assert not torch.isnan(layer_out).any(), "Layer output contains NaNs"

    print("Layer verification passed.")

    # =========================================================================
    # 4. Demonstrate Model Architecture
    # =========================================================================
    print("\n--- Testing HighCapacityBiGRU Model ---")
    model = HighCapacityBiGRU().to(device)

    # Move batch to device
    features = features.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = pair_masks.to(device)
    targets = targets.to(device)

    # Forward pass with real batch data
    outputs = model(features, pair_indices, pair_masks)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch"

    print("Model verification passed.")

    # =========================================================================
    # 5. Demonstrate Loss and Metric
    # =========================================================================
    print("\n--- Testing Loss and Metric ---")

    # Loss (PyTorch)
    loss = mcrmse_loss(outputs, targets)
    print(f"Calculated Loss: {loss.item()}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Metric (NumPy) - Used for validation
    y_pred_np = outputs.detach().cpu().numpy()
    y_true_np = targets.detach().cpu().numpy()

    metric = mcrmse_metric(y_pred_np, y_true_np)
    print(f"Calculated Metric: {metric}")

    assert isinstance(metric, float) or isinstance(
        metric, np.floating
    ), "Metric should be a float"
    assert metric >= 0, "Metric should be non-negative"

    print("Loss and Metric verification passed.")

    # =========================================================================
    # 6. Demonstrate Training Loop (One Epoch)
    # =========================================================================
    print("\n--- Testing Training Loop (1 Epoch) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch of training
    avg_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch finished. Average Loss: {avg_loss:.6f}")

    assert avg_loss > 0, "Training loss should be positive"

    # =========================================================================
    # 7. Demonstrate Validation
    # =========================================================================
    print("\n--- Testing Validation ---")
    val_score = validate(model, val_loader, device)
    print(f"Validation Score: {val_score}")

    assert val_score >= 0, "Validation score should be non-negative"

    # =========================================================================
    # 8. Demonstrate Inference & Submission Generation
    # =========================================================================
    print("\n--- Testing Inference and Submission ---")

    # The inference function generates predictions and writes to Config.SUBMISSION_PATH
    inference(model, test_loader, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {list(sub_df.columns)}")

    # Verify row count: Num_Test_Samples (20 in debug) * Seq_Len (107)
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    print("Inference and Submission verification passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
