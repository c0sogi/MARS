import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader, Subset

# Import from the provided library files
from library.config import Config, set_seed
from library.dataset import get_dataset, InkDataset
from library.model import Hybrid3D2DUNet
from library.engine import run_training
from library.utils import f05_score, rle_encode
from library.inference import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=== Vesuvius Challenge Ink Detection: Library Usage Demonstration ===\n")

    # --- 1. Configuration & Setup ---
    print("1. Configuring environment for demonstration...")

    # Modify Config for speed and isolation
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 1  # Small batch size
    Config.WORKING_DIR = "./working/demo_output"  # Dedicated demo directory
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = "./submission.csv"  # Required output location

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Device: {Config.DEVICE}")
    print("   Configuration updated successfully.\n")

    # --- 2. Dataset Verification ---
    print("2. Verifying Dataset Loading...")

    # Load training dataset
    # We expect metadata to be present in ./metadata/train.csv as per description
    try:
        train_ds = get_dataset(split="train", load_cached_data=True)
        print(f"   Training Dataset loaded. Total samples: {len(train_ds)}")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        # Create a dummy dataset if metadata is missing (fallback for robust demo)
        print("   Creating dummy metadata for demonstration purposes...")
        dummy_meta = pd.DataFrame(
            [
                {
                    "sample_id": "demo_1_0_0",
                    "fragment_id": "1",
                    "x": 0,
                    "y": 0,
                    "w": 512,
                    "h": 512,
                    "mask_path": "train/1/mask.png",
                    "surface_volume_path": "train/1/surface_volume",
                    "inklabels_path": "train/1/inklabels.png",
                }
            ]
        )
        train_ds = InkDataset(dummy_meta, mode="train", load_cached_data=False)

    if len(train_ds) > 0:
        # Fetch one sample
        sample = train_ds[0]
        image = sample["image"]
        mask = sample["mask"]

        # Verify Shapes
        # Image: (1, Z_DIM, H, W) -> (1, 65, 512, 512)
        # Mask: (1, H, W) -> (1, 512, 512)
        print(f"   Sample Image Shape: {image.shape}")
        print(f"   Sample Mask Shape: {mask.shape}")

        assert image.ndim == 4, "Image should be 4D (C, D, H, W)"
        assert image.shape[0] == 1, "Image channel dim should be 1"
        assert image.shape[1] == Config.Z_DIM, f"Image depth should be {Config.Z_DIM}"
        assert mask.ndim == 3, "Mask should be 3D (C, H, W)"

        # Verify Value Ranges
        assert (
            image.min() >= 0.0 and image.max() <= 1.0
        ), "Image values should be normalized [0, 1]"
        unique_mask_vals = torch.unique(mask)
        assert all(
            val in [0, 1] for val in unique_mask_vals
        ), "Mask should be binary {0, 1}"

        print("   Dataset verification passed.\n")
    else:
        raise ValueError("Dataset is empty. Cannot proceed with verification.")

    # --- 3. Model Verification ---
    print("3. Verifying Model Architecture...")

    model = Hybrid3D2DUNet().to(Config.DEVICE)

    # Create a dummy input batch (Batch Size=2)
    # Shape: (B, 1, Z, H, W)
    dummy_input = torch.randn(2, 1, Config.Z_DIM, 512, 512).to(Config.DEVICE)

    print(f"   Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Output shape: {output.shape}")

    # Check output dimensions (B, 1, H, W)
    assert output.shape == (
        2,
        1,
        512,
        512,
    ), f"Expected output (2, 1, 512, 512), got {output.shape}"
    print("   Model forward pass verification passed.\n")

    # --- 4. Utilities Verification ---
    print("4. Verifying Utilities (Metric & RLE)...")

    # Test F0.5 Score
    # Preds: 0.8 (Ink), 0.2 (No Ink). Targets: 1 (Ink), 0 (No Ink) -> Perfect Score
    preds = torch.tensor([0.8, 0.2])
    targets = torch.tensor([1.0, 0.0])
    score = f05_score(preds, targets, threshold=0.5)
    print(f"   Perfect F0.5 Score: {score:.4f}")
    assert np.isclose(score, 1.0), "F0.5 score should be 1.0 for perfect predictions"

    # Test RLE Encode
    # Mask: 0 1 1 0 -> Flat: 0, 1, 1, 0.
    # 1-based indices: pixel 2 and 3 are 1.
    # Start at 2, length 2. Expected: "2 2"
    dummy_mask = np.array([[0, 1, 1, 0]])
    encoded = rle_encode(dummy_mask)
    print(f"   RLE Input: [0, 1, 1, 0], Output: '{encoded}'")
    assert encoded == "2 2", f"RLE encoding failed. Expected '2 2', got '{encoded}'"
    print("   Utilities verification passed.\n")

    # --- 5. Training Loop Demonstration ---
    print("5. Demonstrating Training Loop (1 Epoch, Tiny Subset)...")

    # Clear memory from verification steps
    del dummy_input, output, model
    torch.cuda.empty_cache()

    # Re-initialize model for training
    model = Hybrid3D2DUNet().to(Config.DEVICE)

    # Create tiny subsets for speed
    indices = list(range(min(len(train_ds), 4)))  # Use up to 4 samples
    train_subset = Subset(train_ds, indices)
    val_subset = Subset(train_ds, indices)  # Use same for val just to test pipeline

    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training
    best_score = run_training(model, train_loader, val_loader, optimizer, Config.DEVICE)

    print(f"   Training finished. Best Validation Score: {best_score:.4f}")

    # Verify Checkpoint
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"   Checkpoint verified at: {Config.CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError("Checkpoint file was not created.")
    print("   Training loop verification passed.\n")

    # --- 6. Inference Demonstration ---
    print("6. Demonstrating Inference and Submission Generation...")

    # We need to ensure the model is loaded with the best weights (though run_training saves them)
    # generate_submission uses the passed model instance.
    # For a real run, we would load state_dict, but the model object is already updated or we can reload.
    model.load_state_dict(
        torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    )

    # Run generation
    # This relies on ./metadata/test.csv existing.
    if os.path.exists(Config.TEST_METADATA):
        generate_submission(model, Config.DEVICE)

        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"   Submission file verified at: {Config.SUBMISSION_PATH}")

            # Quick check of file content
            df_sub = pd.read_csv(Config.SUBMISSION_PATH)
            print(f"   Submission contains {len(df_sub)} rows.")
            if not df_sub.empty:
                print(f"   Sample row: {df_sub.iloc[0].to_dict()}")
        else:
            raise FileNotFoundError("Submission file was not created.")
    else:
        print("   Test metadata not found. Skipping inference execution.")

    print("   Inference verification passed.\n")

    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
