import os
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library files
import library.utils as utils
import library.dataset as dataset_lib
import library.model as model_lib
import library.losses as losses_lib
import library.training as training_lib

# -------------------------------------------------------------------------
# 1. Configuration and Setup
# -------------------------------------------------------------------------

# Define temporary directories for the demo
DEMO_DIR = "./working/demo_execution"
DEMO_METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
DEMO_CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "demo_submission")

# Clean up previous runs if they exist
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)

os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

# Monkey-patch the library constants to point to our demo directories
# This forces the library code to use our mini-dataset and output locations
dataset_lib.METADATA_DIR = DEMO_METADATA_DIR
dataset_lib.CACHE_DIR = DEMO_CACHE_DIR
training_lib.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
training_lib.SUBMISSION_DIR = DEMO_SUBMISSION_DIR

# Set seeds for reproducibility
utils.set_seed(42)
print("Setup complete. Library constants patched for demo environment.")


def create_mini_dataset():
    """
    Reads the original metadata, samples a few rows, and saves them
    to the demo metadata directory.
    """
    print("\n--- Creating Mini Dataset ---")

    # Read original metadata
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    df_train = pd.read_csv(orig_train_path)
    df_val = pd.read_csv(orig_val_path)
    df_test = pd.read_csv(orig_test_path)

    # Sample 10 rows for speed
    mini_train = df_train.head(10).copy()
    mini_val = df_val.head(5).copy()
    mini_test = df_test.head(5).copy()

    # Save to the patched metadata directory
    # Note: dataset.py expects files named 'train.csv', 'val.csv', 'test.csv'
    mini_train.to_csv(os.path.join(DEMO_METADATA_DIR, "train.csv"), index=False)
    mini_val.to_csv(os.path.join(DEMO_METADATA_DIR, "val.csv"), index=False)
    mini_test.to_csv(os.path.join(DEMO_METADATA_DIR, "test.csv"), index=False)

    print(f"Mini train shape: {mini_train.shape}")
    print(f"Mini val shape: {mini_val.shape}")
    print(f"Mini test shape: {mini_test.shape}")


def verify_utils():
    """
    Verifies RLE encoding/decoding and IoU calculation.
    """
    print("\n--- Verifying Utils ---")

    # 1. RLE Encode/Decode
    # Create a simple 101x101 mask with a 10x10 square of 1s
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    encoded = utils.rle_encode(mask)
    decoded = utils.rle_decode(encoded, shape=(101, 101))

    assert np.array_equal(mask, decoded), "RLE Decode does not match original mask!"
    print("RLE Encode/Decode logic verified.")

    # 2. IoU Calculation
    # Perfect match
    iou_perfect = utils.calculate_iou_map(mask[None, ...], mask[None, ...])
    assert np.isclose(iou_perfect, 1.0), "IoU for perfect match should be 1.0"

    # No overlap
    mask_inv = np.zeros_like(mask)
    mask_inv[30:40, 30:40] = 1
    iou_zero = utils.calculate_iou_map(mask[None, ...], mask_inv[None, ...])
    assert np.isclose(iou_zero, 0.0), "IoU for disjoint masks should be 0.0"

    print("IoU calculation logic verified.")


def verify_dataset_and_model():
    """
    Loads data using the library and runs a forward pass through the model.
    """
    print("\n--- Verifying Dataset and Model ---")

    # 1. Get DataLoaders
    # We use load_cached_data=False to force processing of our new mini CSVs
    batch_size = 4
    train_loader, val_loader, test_loader = dataset_lib.get_dataloaders(
        batch_size=batch_size,
        load_cached_data=False,
        num_workers=0,  # Use 0 workers for simple debugging/demo
    )

    print(f"Train batches: {len(train_loader)}")

    # 2. Inspect one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Verify shapes
    # Images: (B, 1, 128, 128)
    # Masks: (B, 1, 128, 128)
    # Depths: (B, 1)
    assert images.shape == (
        batch_size,
        1,
        128,
        128,
    ), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (
        batch_size,
        1,
        128,
        128,
    ), f"Unexpected mask shape: {masks.shape}"
    assert depths.shape == (batch_size, 1), f"Unexpected depth shape: {depths.shape}"

    print("DataLoader shapes verified.")

    # 3. Model Forward Pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.DeepResUNet(in_channels=1, out_channels=1).to(device)

    images = images.to(device)
    depths = depths.to(device)

    # Training mode: returns list of outputs (Deep Supervision)
    model.train()
    outputs = model(images, depths)
    assert isinstance(outputs, list), "Model in train mode should return a list"
    assert len(outputs) == 3, "Model should return 3 outputs for deep supervision"
    assert outputs[0].shape == (batch_size, 1, 128, 128), "Main output shape mismatch"

    # Eval mode: returns single output
    model.eval()
    with torch.no_grad():
        output = model(images, depths)
    assert isinstance(output, torch.Tensor), "Model in eval mode should return a Tensor"
    assert output.shape == (batch_size, 1, 128, 128), "Eval output shape mismatch"

    print("Model forward pass verified.")
    return train_loader, val_loader, test_loader, model


def verify_loss():
    """
    Verifies the CurriculumLoss computation.
    """
    print("\n--- Verifying Loss ---")

    criterion = losses_lib.CurriculumLoss()

    # Create dummy data
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 128, 128)).float()

    # Test Cycle 1 (Epoch 0) - Should be BCE + Dice only
    loss_c1 = criterion(logits, targets, epoch=0)
    loss_c1.backward()
    assert not torch.isnan(loss_c1), "Loss is NaN"
    assert logits.grad is not None, "Gradients not computed"

    # Test Cycle 2 (Epoch 60) - Should include Lovasz
    # Reset grads
    logits.grad.zero_()
    loss_c2 = criterion(logits, targets, epoch=60)
    loss_c2.backward()
    assert not torch.isnan(loss_c2), "Loss is NaN"

    print("Loss computation verified.")


def run_training_demo(train_loader, val_loader, test_loader, model):
    """
    Runs a short training loop using the Trainer class.
    """
    print("\n--- Running Training Demo ---")

    # Initialize Trainer
    trainer = training_lib.Trainer(model, train_loader, val_loader, test_loader)

    # Run for 2 epochs to verify the loop
    # This will use the monkey-patched paths for saving checkpoints
    trainer.run(epochs=2)

    # Verify checkpoint creation (best_cycle_X might not be saved if epoch < 50,
    # but the logic runs. We check if the code completed without error).
    print("Training loop completed.")

    # Run Inference
    # This will generate submission.csv in DEMO_SUBMISSION_DIR
    trainer.predict_test_set()

    sub_path = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created!"

    # Check submission content
    df_sub = pd.read_csv(sub_path)
    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Create Data
    create_mini_dataset()

    # 2. Verify Utils
    verify_utils()

    # 3. Verify Dataset & Model
    train_loader, val_loader, test_loader, model = verify_dataset_and_model()

    # 4. Verify Loss
    verify_loss()

    # 5. Run Training Integration Test
    run_training_demo(train_loader, val_loader, test_loader, model)

    print("\nAll demonstrations completed successfully.")
