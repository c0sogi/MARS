import os
import shutil
import torch
import pandas as pd
import numpy as np
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encoding, dice_coef, fbeta_score
from library.model import InkSegFormer
from library.data import get_dataloaders, get_test_dataloader
from library.engine import train_one_epoch, validate, BCEDiceLoss


def run_demo():
    print("--- Starting Vesuvius Ink Detection Demo ---")

    # 1. Setup and Configuration Overrides
    set_seed(Config.SEED)

    # Create temporary directories for this demo
    demo_dir = "./working/demo_execution"
    demo_meta_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Override Config paths and parameters for the demo
    Config.WORKING_DIR = demo_dir
    Config.METADATA_DIR = demo_meta_dir
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for demo

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Prepare Subset Metadata (Micro-Dataset)
    # We create a tiny subset of the data to ensure the demo runs in seconds/minutes.
    print("\n--- Preparing Micro-Dataset ---")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/validation.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Slice to create a micro-dataset (e.g., 4 batches of train, 2 batches of val)
    # Ensure we have enough samples for the batch size
    n_train = Config.BATCH_SIZE * 2
    n_val = Config.BATCH_SIZE * 2

    micro_train = orig_train.head(n_train)
    micro_val = orig_val.head(n_val)
    micro_test = orig_test.head(1)  # Just one fragment for test

    # Save to the demo metadata directory
    micro_train.to_csv(os.path.join(demo_meta_dir, "train.csv"), index=False)
    micro_val.to_csv(os.path.join(demo_meta_dir, "validation.csv"), index=False)
    micro_test.to_csv(os.path.join(demo_meta_dir, "test.csv"), index=False)

    print(f"Created micro-train set: {len(micro_train)} samples")
    print(f"Created micro-val set: {len(micro_val)} samples")

    # 3. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]
    masks = batch["valid_mask"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect Image Shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect Label Shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect Mask Shape"

    # 4. Verify Model Initialization and Forward Pass
    print("\n--- Verifying Model ---")
    device = Config.DEVICE
    model = InkSegFormer().to(device)

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)
    masks = masks.to(device)

    # Forward pass
    logits = model(images)
    print(f"Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect Logits Shape"

    # 5. Verify Loss and Metrics
    print("\n--- Verifying Loss and Metrics ---")
    criterion = BCEDiceLoss()
    loss = criterion(logits, labels, masks)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Check metrics
    probs = torch.sigmoid(logits)
    dice = dice_coef(probs, labels)
    f05 = fbeta_score(probs, labels, beta=0.5)
    print(f"Dice: {dice:.4f}, F0.5: {f05:.4f}")

    # 6. Run Training Loop (One Epoch)
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)
    print(f"Training finished. Avg Loss: {avg_loss:.4f}")

    # 7. Run Validation
    print("\n--- Running Validation ---")
    val_metrics = validate(model, val_loader, device)
    print(f"Validation Metrics: {val_metrics}")

    # 8. Verify Inference and RLE Encoding
    print("\n--- Verifying Inference and Submission ---")

    # Test RLE on a simple dummy array
    dummy_mask = np.array([[0, 1, 1, 0], [0, 0, 1, 0]])
    # Flattened: 0, 1, 1, 0, 0, 0, 1, 0
    # Indices (1-based): 2, 3, 7
    # Runs: Start 2 Len 2, Start 7 Len 1 -> "2 2 7 1"
    rle_str = rle_encoding(dummy_mask)
    print(f"Dummy RLE: {rle_str}")
    assert rle_str == "2 2 7 1", f"RLE failed. Expected '2 2 7 1', got '{rle_str}'"

    # Run actual test pipeline
    test_loader = get_test_dataloader(load_cached_data=True)

    submission_data = []
    model.eval()

    # For the demo, we just process the first batch of the test loader to show it works
    # In a real scenario, we would iterate the whole loader and reconstruct the full image.
    # Here we will just mock the prediction aggregation for the sake of the single-file demo
    # by predicting on patches and creating a dummy submission entry.

    print(f"Processing {len(test_loader)} test batches...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            t_images = batch["image"].to(device)
            # t_ids = batch["fragment_id"] # Not used directly in loop but available

            t_logits = model(t_images)
            t_probs = torch.sigmoid(t_logits)
            t_preds = (t_probs > Config.THRESHOLD).float().cpu().numpy()

            # Just verify we got predictions
            if i == 0:
                print(f"Test Batch Predictions Shape: {t_preds.shape}")
                assert t_preds.shape == (
                    t_images.size(0),
                    1,
                    Config.TILE_SIZE,
                    Config.TILE_SIZE,
                )
                break

    # Create a dummy submission file based on the logic
    # (Since full reconstruction requires stitching which is logic usually outside the provided library files,
    # we demonstrate the format generation).

    # Mock prediction for fragment 'a'
    # In reality, this would be the result of stitching the patches back together.
    # We'll use a small random mask for demonstration.
    mock_full_mask = np.random.randint(0, 2, (100, 100))
    mock_rle = rle_encoding(mock_full_mask)

    submission_df = pd.DataFrame({"Id": ["a"], "Predicted": [mock_rle]})

    submission_path = "./submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file generated at {submission_path}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
