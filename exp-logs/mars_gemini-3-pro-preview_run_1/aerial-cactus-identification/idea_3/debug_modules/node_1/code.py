import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, mixup_data, CactusDataset
from library.model import SteerableCactusNet, EquivariantConv2d
from library.engine import train_engine, generate_submission_csv


def demo_data_pipeline(device):
    print("\n=== Demonstrating Data Pipeline ===")

    # Verify Dataset instantiation
    train_ds = CactusDataset(Config.TRAIN_METADATA_PATH, transform=None, is_test=False)
    print(f"Dataset initialized. Length (Debug): {len(train_ds)}")

    # Fetch a single item
    img, label = train_ds[0]
    print(f"Sample Image Shape: {img.shape}, Label: {label}")
    assert img.shape == (3, 32, 32), f"Expected (3, 32, 32), got {img.shape}"

    # Verify DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch a batch
    batch_imgs, batch_labels = next(iter(train_loader))
    batch_imgs = batch_imgs.to(device)
    batch_labels = batch_labels.to(device)

    print(f"Batch Image Shape: {batch_imgs.shape}")
    print(f"Batch Label Shape: {batch_labels.shape}")

    assert batch_imgs.shape == (Config.BATCH_SIZE, 3, 32, 32)
    assert batch_labels.shape == (Config.BATCH_SIZE,)

    # Verify Mixup
    print("Verifying Mixup augmentation...")
    mixed_x, y_a, y_b, lam = mixup_data(
        batch_imgs, batch_labels, alpha=0.2, device=device
    )
    assert mixed_x.shape == batch_imgs.shape
    assert y_a.shape == batch_labels.shape
    assert y_b.shape == batch_labels.shape
    assert 0 <= lam <= 1
    print("Mixup verification successful.")


def demo_model_components(device):
    print("\n=== Demonstrating Model Components ===")

    batch_size = 4
    h, w = 32, 32

    # 1. Verify EquivariantConv2d (Lifting)
    # Input: (B, 3, H, W) -> Output: (B, Out*8, H, W)
    in_channels = 3
    out_channels = 8
    lifting_layer = EquivariantConv2d(
        in_channels, out_channels, kernel_size=3, padding=1, type="lifting"
    ).to(device)

    dummy_input = torch.randn(batch_size, in_channels, h, w).to(device)
    output = lifting_layer(dummy_input)

    expected_shape = (batch_size, out_channels * 8, h, w)
    print(f"Lifting Layer Input: {dummy_input.shape} -> Output: {output.shape}")
    assert (
        output.shape == expected_shape
    ), f"Lifting layer failed. Expected {expected_shape}, got {output.shape}"

    # 2. Verify EquivariantConv2d (Group)
    # Input: (B, In*8, H, W) -> Output: (B, Out*8, H, W)
    group_in = out_channels  # logical channels
    group_out = 16
    group_layer = EquivariantConv2d(
        group_in, group_out, kernel_size=3, padding=1, type="group"
    ).to(device)

    # Input to group layer is the output of lifting layer
    group_output = group_layer(output)

    expected_group_shape = (batch_size, group_out * 8, h, w)
    print(f"Group Layer Input: {output.shape} -> Output: {group_output.shape}")
    assert (
        group_output.shape == expected_group_shape
    ), f"Group layer failed. Expected {expected_group_shape}, got {group_output.shape}"

    # 3. Verify Full Model
    print("Verifying Full SteerableCactusNet...")
    model = SteerableCactusNet().to(device)
    model_output = model(dummy_input)

    print(f"Model Output Shape: {model_output.shape}")
    # Output should be (B, 1) logits
    assert model_output.shape == (batch_size, 1)
    print("Full model forward pass successful.")


def demo_training_engine():
    print("\n=== Demonstrating Training Engine ===")

    # The train_engine function initializes the model, optimizer, and runs the loop
    # based on Config parameters.
    try:
        train_engine()
        print("Training engine execution completed.")
    except Exception as e:
        print(f"Training engine failed with error: {e}")
        raise e

    # Check if model artifact was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model artifact found at: {Config.MODEL_SAVE_PATH}")
    else:
        # If the model wasn't saved (e.g. validation AUC didn't improve from 0, which is unlikely but possible in 1 epoch),
        # we manually save it to allow the next step to proceed for demonstration purposes.
        print(
            "Model artifact not found (possibly due to short training). Saving dummy model."
        )
        model = SteerableCactusNet()
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)


def demo_submission_generation():
    print("\n=== Demonstrating Submission Generation ===")

    try:
        generate_submission_csv()
        print("Submission generation completed.")
    except Exception as e:
        print(f"Submission generation failed with error: {e}")
        raise e

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file found at: {Config.SUBMISSION_PATH}")
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission Head:")
        print(df.head())

        # Validate submission format
        assert "id" in df.columns
        assert "has_cactus" in df.columns
        assert len(df) > 0
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )


def main():
    # 1. Setup Environment and Config for Speed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # Modify Config for rapid demonstration
    print("Overriding Config for demonstration speed...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Small subset for quick execution
    Config.EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure directories exist (handled by Config, but double checking)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Run Demonstrations
    demo_data_pipeline(device)
    demo_model_components(device)
    demo_training_engine()
    demo_submission_generation()

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
