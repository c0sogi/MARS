import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config, utils, data, model, train


def demo_utils_and_config():
    """
    Demonstrates utility functions and configuration settings.
    """
    print("\n=== Demo: Utils and Config ===")

    # 1. Set Seed
    print("Setting random seed...")
    utils.set_seed(config.SEED)

    # 2. Check Config Paths
    print(f"Input Directory: {config.INPUT_DIR}")
    print(f"Metadata Directory: {config.METADATA_DIR}")
    print(f"Working Directory: {config.WORK_DIR}")

    # Verify directories exist (created by config import)
    assert os.path.exists(config.WORK_DIR), "Working directory should exist"
    assert os.path.exists(config.CHECKPOINT_DIR), "Checkpoint directory should exist"

    # 3. Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Average should be (20 + 40) / 4 = 15
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("AverageMeter logic verified.")


def demo_data_pipeline():
    """
    Demonstrates the data loading pipeline: Dataset creation, Transforms, and DataLoaders.
    """
    print("\n=== Demo: Data Pipeline ===")

    # 1. Test IcebergDataset directly with dummy data
    print("Testing IcebergDataset...")
    N, C, H, W = 10, 3, 75, 75
    dummy_images = np.random.randn(N, C, H, W).astype(np.float32)
    dummy_angles = np.random.rand(N).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, size=(N)).astype(np.float32)

    dataset = data.IcebergDataset(dummy_images, dummy_angles, dummy_labels)

    # Check length
    assert len(dataset) == N

    # Check item retrieval
    img, ang, lbl = dataset[0]
    assert img.shape == (C, H, W), f"Image shape mismatch: {img.shape}"
    assert isinstance(img, torch.Tensor)
    assert isinstance(ang, torch.Tensor)
    assert isinstance(lbl, torch.Tensor)
    print("IcebergDataset basic functionality verified.")

    # 2. Test DataLoaders (using actual data)
    # Note: This will trigger data processing/caching if not present
    print("Loading actual DataLoaders (Fold 0)...")

    # Patch config to ensure we don't run out of memory or time if defaults were huge
    # (Defaults are fine: Batch 32)

    train_loader, val_loader = data.get_dataloaders(fold_idx=0, load_cached_data=True)

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))

    print(
        f"Batch shapes - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert images.dim() == 4
    assert images.size(1) == 3  # 3 channels
    assert images.size(2) == 75 and images.size(3) == 75
    assert angles.dim() == 1
    assert labels.dim() == 1
    assert images.size(0) == config.BATCH_SIZE

    print("DataLoaders verified.")
    return train_loader, val_loader


def demo_model_architecture(device):
    """
    Demonstrates the SPPCNN model instantiation and forward pass.
    """
    print("\n=== Demo: Model Architecture ===")

    # 1. Instantiate Model
    net = model.SPPCNN().to(device)
    print("SPPCNN model instantiated.")

    # 2. Create Dummy Input on Device
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_ang = torch.randn(batch_size).to(device)

    # 3. Forward Pass
    output = net(dummy_img, dummy_ang)

    print(f"Output shape: {output.shape}")

    # 4. Assertions
    assert output.dim() == 2
    assert output.size(0) == batch_size
    assert output.size(1) == 1  # Binary classification logits

    # Check that output is not NaN
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model forward pass verified.")
    return net


def demo_training_execution(train_loader, val_loader, device):
    """
    Demonstrates the training loop components and a short training run.
    """
    print("\n=== Demo: Training Execution ===")

    # 1. Setup
    # Patch config for speed: Reduce epochs to 1 for demonstration
    original_epochs = config.EPOCHS
    config.EPOCHS = 1
    print(f"Temporarily patched config.EPOCHS to {config.EPOCHS} for demo.")

    net = model.SPPCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 2. Run Train One Epoch
    print("Running training for one epoch...")
    train_loss = train.train_one_epoch(
        train_loader, net, criterion, optimizer, device, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float)

    # 3. Run Validation
    print("Running validation...")
    val_loss = train.validate(val_loader, net, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    assert isinstance(val_loss, float)

    # 4. Checkpoint Saving
    print("Testing checkpoint saving...")
    fold_idx = 0
    checkpoint_state = {
        "epoch": 1,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": val_loss,
        "fold_idx": fold_idx,
    }

    # Save
    utils.save_checkpoint(checkpoint_state, is_best=True, fold_idx=fold_idx)

    expected_path = os.path.join(
        config.CHECKPOINT_DIR, f"checkpoint_fold_{fold_idx}.pth"
    )
    expected_best_path = os.path.join(
        config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
    )

    assert os.path.exists(expected_path), "Checkpoint file not created."
    assert os.path.exists(expected_best_path), "Best model file not created."

    # 5. Checkpoint Loading
    print("Testing checkpoint loading...")
    loaded_checkpoint = utils.load_checkpoint(net, expected_path, optimizer, device)
    assert loaded_checkpoint["epoch"] == 1
    assert loaded_checkpoint["fold_idx"] == 0
    print("Checkpoint save/load verified.")

    # Restore config
    config.EPOCHS = original_epochs


def demo_test_inference(device):
    """
    Demonstrates generating predictions on the test set.
    """
    print("\n=== Demo: Test Inference ===")

    # 1. Get Test Loader
    test_loader, test_ids = data.get_test_dataloader(load_cached_data=True)

    # 2. Instantiate Model
    net = model.SPPCNN().to(device)
    net.eval()

    # 3. Run Inference on one batch
    images, angles = next(iter(test_loader))
    images = images.to(device)
    angles = angles.to(device)

    with torch.no_grad():
        logits = net(images, angles)
        probs = torch.sigmoid(logits)

    print(f"Test Batch Probabilities Shape: {probs.shape}")
    print(f"Sample Probabilities: {probs[:3].flatten().cpu().numpy()}")

    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range"
    assert len(test_ids) > 0, "Test IDs should not be empty"

    print("Inference verified.")


if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demo on device: {device}")

    try:
        # Run Demos
        demo_utils_and_config()
        train_loader, val_loader = demo_data_pipeline()
        demo_model_architecture(device)
        demo_training_execution(train_loader, val_loader, device)
        demo_test_inference(device)

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nDEMO FAILED: Assertion Error - {e}")
        exit(1)
    except Exception as e:
        print(f"\nDEMO FAILED: Exception - {e}")
        exit(1)
