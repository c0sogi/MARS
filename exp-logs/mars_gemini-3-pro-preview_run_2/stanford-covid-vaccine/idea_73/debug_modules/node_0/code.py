import os
import sys
import torch
import numpy as np
import random
import shutil

# Add current directory to sys.path to ensure library imports work
sys.path.append(".")

# Import library modules
from library.config import Config
from library.data_utils import get_data
from library.dataset import get_dataloaders
from library.model import AS_DFRN
from library.loss import MCRMSELoss


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_loss_logic():
    """Verifies that MCRMSELoss correctly handles scored vs unscored columns."""
    print("Verifying MCRMSELoss logic...")

    # Instantiate loss
    # Default Config.SCORED_INDICES = [0, 1, 3]
    loss_fn = MCRMSELoss()

    # Create dummy data: Batch=1, Len=107, Channels=5
    inputs = torch.zeros(1, 107, 5)
    targets = torch.zeros(1, 107, 5)

    # Case 1: Perfect prediction -> Loss should be 0
    l_zero = loss_fn(inputs, targets)
    assert l_zero.item() == 0.0, "Loss should be 0.0 for perfect prediction"

    # Case 2: Error in a scored channel (Index 0: reactivity)
    # Target = 1.0, Input = 0.0 -> MSE=1.0, RMSE=1.0
    targets_scored = targets.clone()
    targets_scored[:, :, 0] = 1.0

    # MCRMSE = mean([RMSE_0, RMSE_1, RMSE_3]) = mean([1.0, 0.0, 0.0]) = 1/3
    l_scored = loss_fn(inputs, targets_scored)
    expected = 1.0 / 3.0
    assert (
        abs(l_scored.item() - expected) < 1e-5
    ), f"Expected {expected}, got {l_scored.item()}"

    # Case 3: Error in an unscored channel (Index 2: deg_pH10)
    # This should be ignored by the loss function
    targets_unscored = targets_scored.clone()
    targets_unscored[:, :, 2] = 100.0  # Large error
    l_unscored = loss_fn(inputs, targets_unscored)

    assert (
        abs(l_unscored.item() - expected) < 1e-5
    ), "Unscored channel incorrectly affected loss"

    print("MCRMSELoss logic verified.")


def main():
    # 1. Setup
    set_seed(42)
    print("Setting up demonstration...")

    # 2. Configure for Speed/Demo
    # We modify Config attributes in-place to control the execution environment
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script
    Config.WORKING_DIR = "./working/demo_execution"

    # Redirect cache paths to a temp directory to verify data processing logic
    # and avoid conflicts with existing cache files.
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "demo_train.npz")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "demo_val.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "demo_test.npz")

    # 3. Verify Loss Function
    test_loss_logic()

    # 4. Data Loading
    print("\nLoading DataLoaders (forcing re-computation)...")
    # load_cached_data=False ensures we test the parsing logic in data_utils.py
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a batch to verify shapes and content
    batch = next(iter(train_loader))
    inputs, pair_indices, targets, ids = batch

    print(f"\nBatch Shapes:")
    print(f"Inputs: {inputs.shape} (Expected: [{Config.BATCH_SIZE}, 107, 18])")
    print(f"Pairs: {pair_indices.shape} (Expected: [{Config.BATCH_SIZE}, 107])")
    print(f"Targets: {targets.shape} (Expected: [{Config.BATCH_SIZE}, 107, 5])")

    # Assertions
    assert inputs.shape == (Config.BATCH_SIZE, 107, 18)
    assert pair_indices.shape == (Config.BATCH_SIZE, 107)
    assert targets.shape == (Config.BATCH_SIZE, 107, 5)

    # Verify One-Hot Encoding Integrity
    # First 4 channels are Sequence (A, G, C, U). Sum should be 1.0.
    seq_sum = inputs[0, :, :4].sum(dim=1)
    assert torch.allclose(
        seq_sum, torch.ones_like(seq_sum)
    ), "Sequence one-hot encoding invalid"

    # 5. Model Initialization
    print("\nInitializing AS_DFRN Model...")
    model = AS_DFRN()
    device = Config.DEVICE
    model.to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params}")

    # 6. Training Simulation
    print("\nSimulating Training Step...")
    criterion = MCRMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    model.train()
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    targets = targets.to(device)

    # Forward Pass
    # The model returns predictions from two stages (Pass 1: Zero feedback, Pass 2: Dense feedback)
    y_pred_1, y_pred_2 = model(inputs, pair_indices)

    assert y_pred_1.shape == targets.shape
    assert y_pred_2.shape == targets.shape

    # Loss Calculation
    loss_1 = criterion(y_pred_1, targets)
    loss_2 = criterion(y_pred_2, targets)
    total_loss = loss_1 + loss_2

    print(f"Loss Pass 1: {loss_1.item():.4f}")
    print(f"Loss Pass 2: {loss_2.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

    # Backward Pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("Optimizer step completed.")

    # 7. Inference Simulation
    print("\nSimulating Inference on Test Set...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        t_in, t_pairs, t_targets, t_ids = test_batch

        t_in = t_in.to(device)
        t_pairs = t_pairs.to(device)

        # Inference
        p1, p2 = model(t_in, t_pairs)

        # Check output statistics
        print(f"Prediction Mean: {p2.mean().item():.4f}")
        print(f"Prediction Std: {p2.std().item():.4f}")

        # Verify no NaNs
        assert not torch.isnan(p2).any(), "NaNs detected in prediction"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
