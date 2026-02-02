import os
import torch
import pandas as pd
import numpy as np
import shutil
import time

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, AverageMeter, mixup_data, mixup_criterion
from library.dataset import PetDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, validate_one_epoch, predict
from library.train import run_training
from library.inference import predict_test_set


def main():
    print("Starting Demo Execution...")

    # =========================================================================
    # 1. Configuration Override for Speed and Demo Purposes
    # =========================================================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set parameters for a very fast run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 images
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.N_FOLDS = 2  # Use 2 folds to demonstrate CV loop
    Config.MODEL_ARCHS = ["resnet18"]  # Use a small, standard model
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.TTA_FLIP = False  # Disable TTA for speed

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Architecture: {Config.MODEL_ARCHS}")
    print(f"Debug Mode: {Config.DEBUG}")

    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n[2] Verifying Utility Functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=1)
    meter.update(val=20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("  AverageMeter: OK")

    # Test Mixup
    batch_size = 4
    channels = 3
    height = 224
    width = 224

    dummy_input = torch.randn(batch_size, channels, height, width).to(Config.DEVICE)
    dummy_target = torch.tensor([0.0, 1.0, 0.0, 1.0]).to(Config.DEVICE)

    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_input, dummy_target, alpha=1.0, device=Config.DEVICE
    )

    assert mixed_x.shape == dummy_input.shape, "Mixup output shape mismatch"
    assert y_a.shape == dummy_target.shape, "Mixup target A shape mismatch"
    assert y_b.shape == dummy_target.shape, "Mixup target B shape mismatch"
    assert 0 <= lam <= 1, "Mixup lambda out of range"
    print("  Mixup Logic: OK")

    # =========================================================================
    # 3. Verify Dataset and Transforms
    # =========================================================================
    print("\n[3] Verifying Dataset and Transforms...")

    # Load metadata (using the real metadata files provided in the environment)
    train_meta_df = pd.read_csv(Config.TRAIN_META)

    # Create a small subset manually to ensure we don't rely on Config.DEBUG logic inside Dataset yet
    subset_df = train_meta_df.head(10).copy()

    # Initialize Dataset
    dataset = PetDataset(subset_df, transforms=get_transforms("train"), mode="train")

    # Fetch one item
    img, label = dataset[0]

    # Verify shapes and types
    assert isinstance(img, torch.Tensor), "Dataset did not return a tensor image"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {img.shape}"
    assert isinstance(label, torch.Tensor), "Dataset did not return a tensor label"
    print(f"  Dataset Item Shape: {img.shape}, Label: {label}")
    print("  Dataset: OK")

    # =========================================================================
    # 4. Verify Model and Engine Components
    # =========================================================================
    print("\n[4] Verifying Model and Engine...")

    # Instantiate Model
    model = get_model("resnet18", pretrained=False, num_classes=1)
    model = model.to(Config.DEVICE)

    # Forward pass check
    with torch.no_grad():
        output = model(dummy_input)
        assert output.shape == (
            batch_size,
            1,
        ), f"Model output shape mismatch: {output.shape}"
    print("  Model Instantiation & Forward Pass: OK")

    # Simulate Train One Epoch (Short)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a dummy dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=Config.BATCH_SIZE)

    print("  Testing train_one_epoch...")
    loss = train_one_epoch(
        model, optimizer, train_loader, Config.DEVICE, epoch=0, max_steps=2
    )
    assert not np.isnan(loss), "Training loss returned NaN"
    print(f"  Train Loss: {loss:.4f}")

    # Simulate Validation
    print("  Testing validate_one_epoch...")
    val_loss = validate_one_epoch(model, train_loader, Config.DEVICE, max_steps=2)
    assert not np.isnan(val_loss), "Validation loss returned NaN"
    print(f"  Val Loss: {val_loss:.4f}")

    # Clean up memory
    del model, optimizer, train_loader, dataset
    torch.cuda.empty_cache()

    # =========================================================================
    # 5. Verify Full Training Pipeline (Integration Test)
    # =========================================================================
    print("\n[5] Running Full Training Pipeline (run_training)...")

    # We force `load_cached_data=False` to ensure folds are created from scratch in our demo working dir
    run_training(load_cached_data=False)

    # Verify Checkpoints exist
    expected_checkpoints = [
        f"resnet18_fold_0.pth",
        f"resnet18_fold_1.pth",
        f"best_resnet18_fold_0.pth",
        f"best_resnet18_fold_1.pth",
    ]

    for ckpt in expected_checkpoints:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt)
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print("  Training Pipeline: OK (Checkpoints created)")

    # =========================================================================
    # 6. Verify Inference Pipeline
    # =========================================================================
    print("\n[6] Running Inference Pipeline (predict_test_set)...")

    predict_test_set()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(sub_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(sub_df)}"
    assert (
        "id" in sub_df.columns and "label" in sub_df.columns
    ), "Submission columns mismatch"

    print("  Inference Pipeline: OK (Submission file created)")
    print(f"  Submission Head:\n{sub_df.head()}")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
