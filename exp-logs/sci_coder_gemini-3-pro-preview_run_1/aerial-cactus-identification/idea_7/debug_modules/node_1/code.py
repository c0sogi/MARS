import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import CustomRepVGG, RepVGGBlock
from library.train import Trainer, set_seed
from library.utils import mixup_data


def demonstrate_task():
    print("=== Starting Cactus Identification Task Demonstration ===")

    # 1. Setup and Configuration Patching
    # We patch the Config class to optimize for a quick demo run
    print("\n[Step 1] Configuring environment...")

    # Create a specific directory for this demo to avoid conflicts
    demo_work_dir = "./working/demo_run_script"
    os.makedirs(demo_work_dir, exist_ok=True)

    # Patch Config paths
    Config.WORK_DIR = demo_work_dir
    Config.CACHE_TRAIN_IMGS = os.path.join(demo_work_dir, "cache_train_imgs.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_work_dir, "cache_train_labels.npy")
    Config.CACHE_VAL_IMGS = os.path.join(demo_work_dir, "cache_val_imgs.npy")
    Config.CACHE_VAL_LABELS = os.path.join(demo_work_dir, "cache_val_labels.npy")
    Config.CACHE_TEST_IMGS = os.path.join(demo_work_dir, "cache_test_imgs.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_work_dir, "cache_test_ids.npy")
    Config.BEST_MODEL_PATH = os.path.join(demo_work_dir, "best_model_demo.pth")
    Config.FINAL_SWA_MODEL_PATH = os.path.join(
        demo_work_dir, "final_swa_model_demo.pth"
    )
    Config.SUBMISSION_PATH = os.path.join(demo_work_dir, "submission_demo.csv")

    # Patch Hyperparameters for speed
    Config.EPOCHS = 2
    Config.SWA_START_EPOCH = 1  # Start SWA immediately after epoch 1
    Config.BATCH_SIZE = 64  # Smaller batch size for demo
    Config.NUM_WORKERS = 2

    set_seed(Config.SEED)
    print("Configuration patched. Random seed set.")

    # 2. Dataset Verification
    print("\n[Step 2] Verifying Dataset Loading...")
    # We use load_cached=True to leverage existing caches if available, or create new ones
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached=True
    )

    # Fetch one batch from train loader
    images, labels = next(iter(train_loader))

    print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), "Incorrect train image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect train label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized to [0, 1]"

    print("Dataset verification successful.")

    # 3. Model Architecture & Re-parameterization Verification
    print("\n[Step 3] Verifying Model and RepVGG Logic...")

    # Instantiate model
    model = CustomRepVGG(num_classes=1, deploy=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, 32, 32)

    # Check forward pass
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    # Verify RepVGG Block Fusion (Training -> Deploy)
    print("Testing RepVGG block fusion consistency...")
    block = RepVGGBlock(in_channels=32, out_channels=32, stride=1, deploy=False)
    block.eval()

    x = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        out_train = block(x)

    # Switch to deploy
    block.switch_to_deploy()
    assert block.deploy is True, "Block did not switch to deploy mode"
    assert hasattr(block, "rbr_reparam"), "Fused layer rbr_reparam not found"
    assert not hasattr(
        block, "rbr_dense"
    ), "Training branch rbr_dense should be removed"

    with torch.no_grad():
        out_deploy = block(x)

    # Check difference (should be negligible)
    diff = (out_train - out_deploy).abs().max().item()
    print(f"Max difference between Train and Deploy modes: {diff:.8f}")
    assert diff < 1e-4, "RepVGG fusion resulted in significant output deviation"

    print("Model verification successful.")

    # 4. Utility Verification (Mixup)
    print("\n[Step 4] Verifying Utilities (Mixup)...")
    mix_imgs, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2, use_cuda=False)

    assert mix_imgs.shape == images.shape, "Mixed images shape mismatch"
    assert y_a.shape == labels.shape, "Label A shape mismatch"
    assert y_b.shape == labels.shape, "Label B shape mismatch"
    assert 0 <= lam <= 1, "Lambda should be between 0 and 1"

    print("Mixup verification successful.")

    # 5. Training Loop Demonstration
    print("\n[Step 5] Running Training Loop (2 Epochs)...")

    # Move model to device
    device = Config.DEVICE
    model = model.to(device)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run fit
    # Epoch 1: Standard training
    # Epoch 2: SWA training (since SWA_START_EPOCH = 1)
    final_model = trainer.fit(epochs=Config.EPOCHS)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint not created"
    if Config.EPOCHS > Config.SWA_START_EPOCH:
        assert os.path.exists(
            Config.FINAL_SWA_MODEL_PATH
        ), "SWA model checkpoint not created"

    print("Training loop completed successfully.")

    # 6. Inference and Submission
    print("\n[Step 6] Generating Submission...")

    test_loader = get_test_dataloader(batch_size=Config.BATCH_SIZE, load_cached=True)

    # Ensure model is in eval mode
    final_model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for batch_imgs, batch_ids in test_loader:
            batch_imgs = batch_imgs.to(device)
            outputs = final_model(batch_imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            predictions.extend(probs)
            ids.extend(batch_ids)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "has_cactus": predictions})

    print(f"Generated {len(df_sub)} predictions.")
    print(df_sub.head())

    # Validate submission format
    assert len(df_sub) == 3325, f"Expected 3325 predictions, got {len(df_sub)}"
    assert "id" in df_sub.columns and "has_cactus" in df_sub.columns

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_task()
