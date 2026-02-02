import os
import torch
import numpy as np
import shutil
import glob
from torch.utils.data import DataLoader
import torch.optim as optim

# Import library components
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coeff, average_checkpoints
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.trainer import Trainer


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Configuration & Setup
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINTS_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTIONS_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Create directories
    Config.setup()

    # Set hyperparams for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 samples
    Config.START_SAVING_EPOCH = 1
    Config.SAVE_TOP_K = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed
    set_seed(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # 2. Verify Utils
    print("\n[2] Verifying Utils...")

    # Test RLE Encoding
    # Create a simple 4x4 mask
    # Pixels numbered top-to-bottom, left-to-right
    # 1 5 9  13
    # 2 6 10 14
    # 3 7 11 15
    # 4 8 12 16
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0, 0] = 1  # Pixel 1
    dummy_mask[1, 0] = 1  # Pixel 2
    # Run: start 1, length 2

    encoded = rle_encode(dummy_mask)
    print(f"RLE Encode Test: {encoded}")
    assert encoded == "1 2", f"Expected '1 2', got '{encoded}'"

    # Test Dice Coefficient
    pred_t = torch.tensor([1.0, 1.0, 0.0])
    target_t = torch.tensor([1.0, 1.0, 0.0])
    score = dice_coeff(pred_t, target_t)
    print(f"Dice Score (Perfect Match): {score.item():.4f}")
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)

    pred_bad = torch.tensor([0.0, 0.0, 1.0])
    score_bad = dice_coeff(pred_bad, target_t)
    print(f"Dice Score (No Match): {score_bad.item():.4f}")
    assert torch.isclose(score_bad, torch.tensor(0.0), atol=1e-4)

    # 3. Verify Dataset
    print("\n[3] Verifying Dataset...")

    # Initialize Train Dataset
    train_dataset = ContrailDataset(
        split="train", transform=get_transforms("train"), debug=True
    )
    print(f"Train Dataset Length (Debug): {len(train_dataset)}")
    assert len(train_dataset) == Config.DEBUG_SAMPLE_SIZE

    # Fetch one item
    img, mask = train_dataset[0]
    print(f"Image Shape: {img.shape}")  # Should be (6, 256, 256)
    print(f"Mask Shape: {mask.shape}")  # Should be (1, 256, 256)

    assert img.shape == (6, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32

    # 4. Verify Model
    print("\n[4] Verifying Model...")

    device = Config.DEVICE
    model = ConvNeXtUNet().to(device)

    # Dummy forward pass
    dummy_input = torch.randn(2, 6, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1, 256, 256)

    # 5. Verify Training Loop
    print("\n[5] Running Training Demo...")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Validation Dataset (reuse train logic for demo speed)
    val_dataset = ContrailDataset(
        split="validation", transform=get_transforms("validation"), debug=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Setup Training Components
    criterion = HybridLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
    )

    # Run Fit
    trainer.fit()

    # Check if checkpoint exists
    checkpoints = glob.glob(os.path.join(Config.CHECKPOINTS_DIR, "*.pth"))
    print(f"Checkpoints found: {len(checkpoints)}")
    assert len(checkpoints) > 0, "No checkpoints were saved!"

    # 6. Verify Checkpoint Averaging
    print("\n[6] Verifying Checkpoint Averaging...")
    if len(checkpoints) > 0:
        avg_path = os.path.join(Config.CHECKPOINTS_DIR, "avg_checkpoint.pth")
        avg_state = average_checkpoints(checkpoints, output_path=avg_path)
        print("Averaged checkpoint saved successfully.")
        assert os.path.exists(avg_path)

        # Verify loading averaged weights
        model.load_state_dict(avg_state)
        print("Model loaded with averaged weights.")

    # 7. Verify Inference / Test Dataset
    print("\n[7] Verifying Test Dataset...")
    test_dataset = ContrailDataset(
        split="test",
        transform=get_transforms("test"),
        debug=True,  # Need debug=True to limit size, though test set usually has no masks
    )

    test_img, record_id = test_dataset[0]
    print(f"Test Image Shape: {test_img.shape}")
    print(f"Record ID: {record_id}")

    assert isinstance(record_id, str)
    assert test_img.shape == (6, 256, 256)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
