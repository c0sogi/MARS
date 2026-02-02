import os
import numpy as np
import pandas as pd
import torch
from library.utils import set_seed, rle_encode, rle_decode
from library.dataset import get_dataloaders
from library.model import DepthAwareUNet
from library.trainer import SaltTrainer


def main():
    # 1. Setup
    print("Setting up demonstration...")
    set_seed(42)
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    # Define paths for mini metadata files
    mini_train_path = os.path.join(working_dir, "mini_train.csv")
    mini_val_path = os.path.join(working_dir, "mini_val.csv")
    mini_test_path = os.path.join(working_dir, "mini_test.csv")

    # 2. Create Mini Metadata for Speed
    # We take a small slice of the actual metadata to ensure the code runs quickly
    print("Creating mini-datasets...")
    df_train = pd.read_csv("./metadata/train.csv").head(32)  # Enough for a few batches
    df_val = pd.read_csv("./metadata/val.csv").head(16)
    df_test = pd.read_csv("./metadata/test.csv").head(16)

    df_train.to_csv(mini_train_path, index=False)
    df_val.to_csv(mini_val_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    # 3. Demonstrate Data Loading
    print("Initializing DataLoaders...")
    # load_cached_data=False ensures we process our new mini-CSVs instead of loading
    # any pre-existing full-dataset cache files.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead for this small demo
        load_cached_data=False,
        train_metadata=mini_train_path,
        val_metadata=mini_val_path,
        test_metadata=mini_test_path,
    )

    # Verify DataLoader Output
    batch_img, batch_mask, batch_ids = next(iter(train_loader))
    print(f"Train Batch Shapes - Image: {batch_img.shape}, Mask: {batch_mask.shape}")

    # Assertions to verify data pipeline logic
    # Expected shape: (Batch, 2, 128, 128). 2 Channels are (Image, Depth)
    assert batch_img.shape == (
        8,
        2,
        128,
        128,
    ), f"Unexpected image shape: {batch_img.shape}"
    # Expected shape: (Batch, 1, 128, 128). 1 Channel for binary mask
    assert batch_mask.shape == (
        8,
        1,
        128,
        128,
    ), f"Unexpected mask shape: {batch_mask.shape}"
    assert len(batch_ids) == 8, "IDs list length mismatch"
    assert batch_img.dtype == torch.float32, "Image tensor should be float32"

    # 4. Demonstrate RLE Utilities
    print("Verifying RLE Encoding/Decoding...")
    # Create a synthetic 101x101 mask with a square in the middle
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[20:40, 20:40] = 1

    # Round-trip check
    encoded_rle = rle_encode(dummy_mask)
    decoded_mask = rle_decode(encoded_rle, shape=(101, 101))

    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Encode -> Decode roundtrip failed"
    print("RLE utilities verified.")

    # 5. Demonstrate Model
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_channels=2 (Image + Depth), n_classes=1 (Binary Salt Mask)
    model = DepthAwareUNet(n_channels=2, n_classes=1).to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 2, 128, 128).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass successful.")

    # 6. Demonstrate Training
    print("Starting Training Demonstration...")
    checkpoint_dir = os.path.join(working_dir, "checkpoints")

    trainer = SaltTrainer(
        model=model, device=device, learning_rate=1e-3, checkpoint_dir=checkpoint_dir
    )

    # Train for 2 epochs to demonstrate the loop and checkpointing
    # Since dataset is tiny (32 images), this will be very fast.
    trainer.train(train_loader, val_loader, epochs=2, patience=2)

    # Verify Checkpoint
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("Training completed and checkpoint verified.")

    # 7. Demonstrate Submission
    print("Generating Submission...")
    submission_path = os.path.join(working_dir, "submission_demo.csv")

    trainer.generate_submission(test_loader, output_file=submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    expected_cols = ["id", "rle_mask"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"
    assert (
        len(sub_df) == 16
    ), f"Expected 16 predictions (from mini test set), got {len(sub_df)}"

    # Check RLE format (simple check if it's string or NaN/empty)
    # Note: If model predicts all zeros, rle might be empty string or NaN depending on pandas loading
    # The sample submission usually has strings.
    print("Submission generated successfully.")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
