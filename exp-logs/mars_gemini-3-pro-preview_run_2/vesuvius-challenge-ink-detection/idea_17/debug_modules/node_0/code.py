import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import (
    set_seed,
    rle_encoding,
    dice_coef,
    fbeta_score,
    get_cached_volume,
    create_3ch_input,
)
from library.dataset import InkDataset, get_dataset
from library.model import build_segformer_model
from library.engine import fit, evaluate


def demo_utils():
    print("\n=== Demonstrating Library Utils ===")

    # 1. Test RLE Encoding
    # Create a simple 5x5 mask with a line of 1s
    dummy_mask = np.zeros((5, 5), dtype=np.uint8)
    dummy_mask[1, 1:4] = 1  # Row 1, cols 1,2,3 are ink
    # Flattened indices: 6, 7, 8 (0-indexed) -> Pixels 7, 8, 9 (1-indexed)
    # RLE should be "7 3" (Start at 7, length 3)
    encoded = rle_encoding(dummy_mask)
    print(f"RLE Input (Row 1 is 1s): \n{dummy_mask}")
    print(f"RLE Output: '{encoded}'")

    # Assert correctness
    # Pixels are 1-indexed in RLE.
    # Row 0 has 5 pixels. Row 1 starts at index 5 (0-based) -> pixel 6 (1-based).
    # wait, flattened:
    # 0 1 2 3 4
    # 5 6 7 8 9
    # Indices of 1s: 6, 7, 8.
    # 1-based: 7, 8, 9.
    # Start 7, Length 3.
    assert encoded == "7 3", f"RLE failed. Expected '7 3', got '{encoded}'"
    print("RLE Encoding verified.")

    # 2. Test Metrics
    # Perfect match
    preds = torch.tensor([1.0, 0.0, 1.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    dice = dice_coef(preds, targets, threshold=0.5)
    f05 = fbeta_score(preds, targets, beta=0.5, threshold=0.5)

    assert np.isclose(dice, 1.0), f"Dice failed for perfect match: {dice}"
    assert np.isclose(f05, 1.0), f"F0.5 failed for perfect match: {f05}"
    print(f"Metrics verified (Perfect Match): Dice={dice}, F0.5={f05}")

    # No match
    preds = torch.tensor([0.0, 1.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    dice = dice_coef(preds, targets, threshold=0.5)
    assert np.isclose(dice, 0.0), f"Dice failed for no match: {dice}"
    print(f"Metrics verified (No Match): Dice={dice}")


def demo_dataset_and_loader():
    print("\n=== Demonstrating Dataset and DataLoader ===")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    print(f"Original Train Metadata Length: {len(df_train)}")

    # Subset for speed - take 4 samples
    df_subset = df_train.head(4).copy()

    # Initialize Dataset manually with subset
    # We use 'train' split to get labels
    from library.utils import get_transforms

    transforms = get_transforms(data="train")

    dataset = InkDataset(
        df_subset, split="train", z_start=Config.Z_START_TRAIN, transforms=transforms
    )

    # Verify length
    assert len(dataset) == 4

    # Verify item structure
    sample = dataset[0]
    image = sample["image"]
    label = sample["label"]
    valid_mask = sample["valid_mask"]

    print(f"Sample keys: {sample.keys()}")
    print(f"Image Shape: {image.shape} (C, H, W)")
    print(f"Label Shape: {label.shape} (C, H, W)")
    print(f"Valid Mask Shape: {valid_mask.shape} (C, H, W)")

    # Assertions
    # Image should be 3 channels (from 3 MIP slabs)
    assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
    assert image.shape[1] == Config.TILE_SIZE
    assert image.shape[2] == Config.TILE_SIZE

    # Label should be 1 channel
    assert label.shape[0] == 1

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch Image Shape: {batch['image'].shape}")
    print("Dataset and DataLoader verified.")

    return loader


def demo_model_forward(loader, device):
    print("\n=== Demonstrating Model Architecture ===")

    model = build_segformer_model()
    model.to(device)
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)

    with torch.no_grad():
        outputs = model(images)

    print(f"Input Shape: {images.shape}")
    print(f"Output Logits Shape: {outputs.shape}")

    # Assertions
    # Output should be (B, 1, H, W)
    assert outputs.shape == (
        images.shape[0],
        1,
        images.shape[2],
        images.shape[3],
    ), f"Output shape mismatch. Expected {(images.shape[0], 1, images.shape[2], images.shape[3])}, got {outputs.shape}"

    print("Model forward pass verified.")
    return model


def demo_training_loop(model, train_loader, device):
    print("\n=== Demonstrating Training Loop (Engine) ===")

    # Create a dummy validation loader (same as train for demo purposes)
    valid_loader = train_loader

    # Run fit
    # Note: Config has been monkey-patched in main to reduce epochs
    print(f"Running training for {Config.NUM_EPOCHS} epoch(s)...")

    best_score = fit(
        model, train_loader, valid_loader, epochs=Config.NUM_EPOCHS, device=device
    )

    print(f"Training finished. Best Score: {best_score}")

    # Verify output file
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Checkpoint found at: {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")


def demo_inference_and_submission(model, device):
    print("\n=== Demonstrating Inference and Submission Generation ===")

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print("Test metadata not found, skipping inference demo.")
        return

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    # We need to generate patches for the test set using the utility in dataset.py
    # But get_dataset('test') does this internally.

    # For demo speed, we will manually create a tiny test dataframe
    # referencing the first fragment in the test metadata
    from library.dataset import get_test_patches

    # Generate patches for the first test fragment
    # Note: get_test_patches reads the mask file to determine size.
    # We'll just use the provided function.
    df_test_patches = get_test_patches(df_test.head(1))

    # Subset to just 2 patches for speed
    df_test_subset = df_test_patches.head(2).copy()
    print(f"Running inference on {len(df_test_subset)} test patches...")

    from library.utils import get_transforms

    transforms = get_transforms(data="test")

    test_dataset = InkDataset(
        df_test_subset,
        split="test",
        z_start=Config.Z_START_TRAIN,  # Using training Z-start for demo simplicity
        transforms=transforms,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            frag_id = batch["fragment_id"][0]

            # Forward
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Binarize
            mask = (probs > Config.THRESHOLD).cpu().numpy().astype(np.uint8)
            # Remove batch and channel dim: (1, 1, H, W) -> (H, W)
            mask = mask[0, 0, :, :]

            # Encode
            rle = rle_encoding(mask)

            # In a real scenario, we would stitch these patches back into a full image
            # before RLE. Here we just demonstrate the encoding of a patch.
            # The competition requires RLE of the full fragment.
            # Since this is a code example, we will just print the patch RLE.
            submission_rows.append({"Id": f"{frag_id}_patch_demo", "Predicted": rle})

    # Create submission dataframe
    sub_df = pd.DataFrame(submission_rows)
    print("Sample Submission Data:")
    print(sub_df)

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission file saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    # 0. Setup
    set_seed(42)

    # Monkey-patch Config for speed
    print("Configuring parameters for demo speed...")
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.EARLY_STOPPING_PATIENCE = 1
    # Ensure working dir exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Utils
    demo_utils()

    # 2. Dataset
    loader = demo_dataset_and_loader()

    # 3. Model
    model = demo_model_forward(loader, device)

    # 4. Training
    demo_training_loop(model, loader, device)

    # 5. Inference
    demo_inference_and_submission(model, device)

    print("\n=== Demo Completed Successfully ===")
