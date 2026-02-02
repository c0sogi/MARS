import os
import torch
import numpy as np
import sys

# Import from the provided library files
from library.config import WORKING_DIR, TRAIN_METADATA_PATH, VAL_METADATA_PATH
from library.utils import (
    set_seed,
    normalize_range,
    rle_encode,
    dice_coeff,
    get_transforms,
)
from library.dataset import ContrailsDataset
from library.model import DilatedResNetUNet
from library.loss import HybridLoss
from library.train import train_model


def run_demo():
    print("Starting Library Usage Demo...")

    # 1. Setup and Reproducibility
    set_seed(42)
    print("Random seed set.")

    # 2. Verify Utilities
    print("\n--- Verifying Utilities ---")

    # Test normalize_range
    data = np.array([-10.0, 0.0, 10.0, 20.0])
    norm = normalize_range(data, 0.0, 10.0)
    expected_norm = np.array([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(norm, expected_norm, err_msg="normalize_range failed")
    print("normalize_range: OK")

    # Test rle_encode
    # Create a 3x3 mask.
    # Column-major indexing:
    # (0,0)=1, (1,0)=2, (2,0)=3
    # (0,1)=4, (1,1)=5, (2,1)=6
    # (0,2)=7, (1,2)=8, (2,2)=9
    # Let's mark pixels 4 and 5 (Column 1, rows 0 and 1).
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 1] = 1  # Pixel 4
    mask[1, 1] = 1  # Pixel 5

    encoded = rle_encode(mask)
    # Expected: Start at 4, length 2 -> "4 2"
    assert encoded == "4 2", f"rle_encode failed. Expected '4 2', got '{encoded}'"
    print("rle_encode: OK")

    # Test dice_coeff
    pred = torch.tensor([1.0, 1.0, 0.0])
    target = torch.tensor([1.0, 0.0, 0.0])
    # Intersection = 1. Union = 2 + 1 = 3. Dice = 2*1 / 3 = 0.666...
    score = dice_coeff(pred, target)
    assert abs(score.item() - 0.666666) < 1e-4, f"dice_coeff failed. Got {score.item()}"
    print("dice_coeff: OK")

    # 3. Verify Dataset
    print("\n--- Verifying Dataset ---")
    # Use debug_size to load only a few records
    ds = ContrailsDataset(TRAIN_METADATA_PATH, split="train", debug_size=10)
    print(f"Dataset initialized with {len(ds)} samples.")

    sample = ds[0]
    image = sample["image"]
    mask = sample["mask"]
    record_id = sample["record_id"]

    print(f"Sample Record ID: {record_id}")
    print(f"Image Shape: {image.shape}")  # Should be (6, 256, 256)
    print(f"Mask Shape: {mask.shape}")  # Should be (1, 256, 256)

    assert image.shape == (6, 256, 256), "Incorrect image shape"
    assert mask.shape == (1, 256, 256), "Incorrect mask shape"
    assert isinstance(image, torch.Tensor), "Image is not a tensor"
    print("Dataset loading: OK")

    # 4. Verify Model
    print("\n--- Verifying Model ---")
    model = DilatedResNetUNet()
    # Move model to CPU for this quick test to avoid VRAM overhead if not needed,
    # though config defaults to CUDA if available.
    device = torch.device("cpu")
    model.to(device)

    # Create a dummy batch
    input_tensor = image.unsqueeze(0).to(device)  # (1, 6, 256, 256)

    with torch.no_grad():
        output = model(input_tensor)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (1, 1, 256, 256), "Model output shape mismatch"
    print("Model forward pass: OK")

    # 5. Verify Loss
    print("\n--- Verifying Loss ---")
    criterion = HybridLoss()
    target_tensor = mask.unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # Ensure output requires grad for backward check simulation
    output.requires_grad = True
    loss = criterion(output, target_tensor)

    print(f"Loss Value: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # Verify gradients can be computed
    loss.backward()
    print("Loss backward pass: OK")

    # 6. Verify Training Loop
    print("\n--- Verifying Training Loop ---")
    # We run a very short training session: 1 epoch, batch size 2, 4 samples total
    # This tests the integration of Dataset, DataLoader, Model, Optimizer, and Loss.

    try:
        train_model(
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            debug_size=4,  # Only use 4 images for speed
            patience=1,
        )
        print("Training loop execution: OK")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

    # Check if model was saved
    saved_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(saved_model_path):
        print(f"Model checkpoint found at: {saved_model_path}")
    else:
        # It's possible no improvement happened in 1 epoch with random weights,
        # but the code should run without crashing.
        print(
            "Note: No best_model.pth found (validation might not have improved in 1 epoch), but execution finished."
        )

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
