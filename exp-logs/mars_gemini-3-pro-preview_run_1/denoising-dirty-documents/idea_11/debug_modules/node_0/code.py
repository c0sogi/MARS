import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import components from the provided library
from library.config import TRAIN_METADATA_PATH, VAL_METADATA_PATH, WORKING_DIR, DEVICE
from library.utils import (
    seed_everything,
    invert_signal,
    revert_signal,
    rmse_loss,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import load_and_cache_data, DenoisingDataset
from library.models import build_context_model, build_diversity_model
from library.engine import fit_model


def main():
    # 1. Setup
    print("--- 1. Setup ---")
    seed_everything(42)
    print(f"Device: {DEVICE}")

    # 2. Utility Verification
    print("\n--- 2. Verifying Utilities ---")
    # Test Signal Inversion (1.0 - x)
    dummy_tensor = torch.tensor([0.0, 0.5, 1.0])
    inverted = invert_signal(dummy_tensor)
    reverted = revert_signal(inverted)

    expected_inverted = torch.tensor([1.0, 0.5, 0.0])

    assert torch.allclose(inverted, expected_inverted), "invert_signal logic failed"
    assert torch.allclose(reverted, dummy_tensor), "revert_signal logic failed"
    print("Signal processing utilities verified.")

    # Test RMSE Loss
    pred = torch.tensor([0.2, 0.8])
    target = torch.tensor([0.8, 0.2])
    # MSE = ((0.6^2) + (0.6^2)) / 2 = (0.36 + 0.36) / 2 = 0.36
    # RMSE = sqrt(0.36) = 0.6
    loss = rmse_loss(pred, target)
    assert torch.isclose(
        loss, torch.tensor(0.6)
    ), f"RMSE calculation failed. Got {loss}, expected 0.6"
    print("RMSE metric verified.")

    # 3. Data Pipeline Demonstration
    print("\n--- 3. Data Pipeline ---")
    # Load a small subset of data to speed up the demo
    train_data = load_and_cache_data(
        TRAIN_METADATA_PATH, cache_name="demo_train_cache.npz", limit=10
    )
    val_data = load_and_cache_data(
        VAL_METADATA_PATH, cache_name="demo_val_cache.npz", limit=5
    )

    assert len(train_data) == 10, "Train data subset loading failed"
    assert len(val_data) == 5, "Val data subset loading failed"

    # Instantiate Datasets
    # Stream A uses 320x320 patches
    train_dataset = DenoisingDataset(
        train_data, patch_size=320, augment=True, mode="train"
    )
    val_dataset = DenoisingDataset(val_data, patch_size=None, augment=False, mode="val")

    # Verify Dataset Item
    sample = train_dataset[0]
    assert "noisy" in sample and "clean" in sample
    assert sample["noisy"].shape == (
        1,
        320,
        320,
    ), f"Unexpected shape: {sample['noisy'].shape}"
    # Check value range (should be [0, 1])
    assert sample["noisy"].min() >= 0.0 and sample["noisy"].max() <= 1.0
    print("Dataset shapes and value ranges verified.")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4. Model Instantiation
    print("\n--- 4. Model Instantiation ---")
    # Build Context Model (Stream A)
    model_ctx = build_context_model().to(DEVICE)

    # Build Diversity Model (Stream B)
    model_div = build_diversity_model().to(DEVICE)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 1, 320, 320).to(DEVICE)

    with torch.no_grad():
        out_ctx = model_ctx(dummy_input)
        out_div = model_div(dummy_input)

    assert out_ctx.shape == (2, 1, 320, 320), "Context model output shape mismatch"
    assert out_div.shape == (2, 1, 320, 320), "Diversity model output shape mismatch"
    print("Model architectures verified successfully.")

    # 5. Training Loop Demonstration
    print("\n--- 5. Training Loop (Engine) ---")
    optimizer = optim.Adam(model_ctx.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    save_path = os.path.join(WORKING_DIR, "demo_model.pth")

    # Run training for 1 epoch
    print("Running fit_model for 1 epoch...")
    best_rmse = fit_model(
        model=model_ctx,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=1,
        save_path=save_path,
    )

    print(f"Training complete. Best Validation RMSE: {best_rmse:.4f}")
    assert os.path.exists(save_path), "Model checkpoint was not created."

    # 6. Checkpoint Loading
    print("\n--- 6. Checkpoint Verification ---")
    # Create a new model instance to load weights into
    loaded_model = build_context_model().to(DEVICE)
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=1e-3)

    checkpoint = load_checkpoint(save_path, loaded_model, loaded_optimizer)

    assert "state_dict" in checkpoint
    assert "optimizer" in checkpoint
    assert checkpoint["epoch"] == 1
    assert checkpoint["val_rmse"] == best_rmse

    print("Checkpoint loaded successfully. All demonstrations passed.")


if __name__ == "__main__":
    main()
