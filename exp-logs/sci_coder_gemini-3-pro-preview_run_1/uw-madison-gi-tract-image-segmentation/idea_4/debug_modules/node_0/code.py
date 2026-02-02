import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_and_preprocess_metadata
from library.dataset import UWDataset, get_transforms
from library.model import UnetPlusPlus
from library.loss import DeepSupervisionLoss
from library.train import Trainer
from library.inference import InferencePipeline


def test_dataset_and_transforms():
    print("\n=== Testing Dataset and Transforms ===")

    # Load metadata
    df = load_and_preprocess_metadata(Config.TRAIN_CSV)

    # Create dataset in train mode
    transforms = get_transforms(mode="train")
    ds = UWDataset(df, mode="train", transforms=transforms)

    # Fetch one sample
    sample = ds[0]
    image = sample["image"]
    mask = sample["mask"]

    # Verify Shapes
    # Image: (3, H, W) - Config.IMG_SIZE is (320, 320)
    expected_shape = (3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        image.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {image.shape}"

    # Mask: (Num_Classes, H, W)
    expected_mask_shape = (Config.NUM_CLASSES, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        mask.shape == expected_mask_shape
    ), f"Mask shape mismatch. Expected {expected_mask_shape}, got {mask.shape}"

    # Verify Data Types
    assert image.dtype == torch.float32, "Image should be float32"
    assert mask.dtype == torch.float32, "Mask should be float32"

    # Verify Normalization (Robust Percentile Normalization should result in values roughly 0-1,
    # but can exceed slightly due to clipping logic or be exactly 0/1)
    print(f"Sample Image Range: [{image.min():.4f}, {image.max():.4f}]")

    print("Dataset verification passed.")


def test_model_and_loss():
    print("\n=== Testing Model and Loss Logic ===")

    device = Config.DEVICE
    batch_size = 2

    # Instantiate Model
    model = UnetPlusPlus(
        backbone_name=Config.BACKBONE, classes=Config.NUM_CLASSES, deep_supervision=True
    ).to(device)

    # Create Dummy Input
    dummy_input = torch.randn(
        batch_size, Config.IN_CHANNELS, Config.IMG_SIZE[0], Config.IMG_SIZE[1]
    ).to(device)
    dummy_target = (
        torch.randint(
            0,
            2,
            (batch_size, Config.NUM_CLASSES, Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
        )
        .float()
        .to(device)
    )

    # 1. Test Training Forward Pass (Deep Supervision)
    model.train()
    outputs = model(dummy_input)

    # Should return a list of 4 tensors (levels of deep supervision)
    assert isinstance(
        outputs, list
    ), "Model in training mode should return a list (Deep Supervision)"
    assert (
        len(outputs) == 4
    ), f"Expected 4 outputs from deep supervision, got {len(outputs)}"
    assert outputs[0].shape == (
        batch_size,
        Config.NUM_CLASSES,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Output shape mismatch"

    # 2. Test Loss Function
    criterion = DeepSupervisionLoss()
    loss = criterion(outputs, dummy_target)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print(f"Calculated Loss: {loss.item():.4f}")

    # 3. Test Inference Forward Pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Should return a single tensor
    assert torch.is_tensor(output), "Model in eval mode should return a single tensor"
    assert output.shape == (
        batch_size,
        Config.NUM_CLASSES,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Inference output shape mismatch"

    print("Model and Loss verification passed.")


def run_training_demo():
    print("\n=== Running Training Demo ===")

    # Initialize Trainer
    trainer = Trainer()

    # Run fit with debug=True (subsamples data) and epochs=1
    best_dice = trainer.fit(debug=True, epochs=Config.EPOCHS)

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    print(f"Training demo finished. Best Dice: {best_dice:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")


def run_inference_demo():
    print("\n=== Running Inference Demo ===")

    # 1. Create a small subset of test data for speed
    # We read the full test csv, take the first case, and save it to a temp file
    full_test_df = pd.read_csv(Config.TEST_CSV)

    if len(full_test_df) > 0:
        first_case = full_test_df.iloc[0]["case"]
        subset_df = full_test_df[full_test_df["case"] == first_case].copy()

        temp_test_csv = os.path.join(Config.WORKING_DIR, "temp_test_subset.csv")
        subset_df.to_csv(temp_test_csv, index=False)

        # Override Config to point to this subset
        original_test_csv = Config.TEST_CSV
        Config.TEST_CSV = temp_test_csv
        print(f"Using subset of test data ({len(subset_df)} slices) for demonstration.")
    else:
        print("Test CSV is empty, skipping inference subset creation.")
        return

    # 2. Run Inference Pipeline
    # It will pick up best_model.pth from the checkpoint dir automatically
    pipeline = InferencePipeline()
    pipeline.generate_submission()

    # 3. Validate Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check columns
    expected_cols = ["id", "class", "predicted"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {sub_df.columns}"

    # Restore Config
    Config.TEST_CSV = original_test_csv
    print("Inference demo passed.")


if __name__ == "__main__":
    # 1. Setup Environment & Config Overrides for Demo Speed
    set_seed(42)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG = True  # Ensures Trainer uses a small subset of training data

    # Ensure working directory is clean/ready
    if os.path.exists(Config.WORKING_DIR):
        # We don't delete it to avoid removing pre-calculated metadata cache if it exists,
        # but we ensure subdirs exist.
        pass
    Config.setup_directories()

    # 2. Run Verifications
    try:
        test_dataset_and_transforms()
        test_model_and_loss()
        run_training_demo()
        run_inference_demo()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
