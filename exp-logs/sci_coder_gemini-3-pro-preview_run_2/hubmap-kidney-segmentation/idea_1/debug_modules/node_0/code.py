import os
import numpy as np
import pandas as pd
import torch
import rasterio
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, dice_coef
from library.dataset import HuBMAPDataset
from library.model import FPNResNet18
from library.losses import SoftDiceLoss
from library.trainer import Trainer
from library.inference import InferenceRunner


def run_demo():
    # 1. Setup and Configuration Overrides
    print("--- Setting up Demo Environment ---")

    # Define paths for synthetic data
    demo_base_dir = "./working/demo_env"
    demo_input_dir = os.path.join(demo_base_dir, "input")
    demo_train_dir = os.path.join(demo_input_dir, "train")
    demo_test_dir = os.path.join(demo_input_dir, "test")

    os.makedirs(demo_train_dir, exist_ok=True)
    os.makedirs(demo_test_dir, exist_ok=True)

    # Override Config to use our demo directories and speed up execution
    Config.INPUT_DIR = demo_input_dir
    Config.WORKING_DIR = os.path.join(demo_base_dir, "working")
    Config.SUBMISSION_DIR = os.path.join(demo_base_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.TILE_SIZE = 512  # Smaller tile for faster processing
    Config.BACKGROUND_SAMPLE_RATE = 1.0  # Keep all for this tiny demo

    seed_everything(Config.SEED)

    # 2. Generate Synthetic Data
    print("--- Generating Synthetic Data ---")

    # Create a dummy image (H, W, C)
    img_h, img_w = 1024, 1024
    dummy_img = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)

    # Create a dummy mask (H, W) - Draw a square in the middle
    dummy_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    dummy_mask[256:768, 256:768] = 1

    # Save dummy image as TIFF
    img_filename = "synthetic_01.tiff"
    img_path = os.path.join(demo_train_dir, img_filename)

    # Rasterio expects (Count, H, W)
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        height=img_h,
        width=img_w,
        count=3,
        dtype=dummy_img.dtype,
    ) as dst:
        dst.write(np.moveaxis(dummy_img, -1, 0))

    # Generate RLE
    rle_str = rle_encode(dummy_mask)

    # Create Metadata DataFrame
    data = {
        "id": ["synthetic_01"],
        "image_file": [img_filename],
        "width_pixels": [img_w],
        "height_pixels": [img_h],
        "encoding": [rle_str],
        "image_path": [os.path.join("train", img_filename)],
    }
    metadata_df = pd.DataFrame(data)

    print(f"Created synthetic image at {img_path}")
    print(f"Created metadata with {len(metadata_df)} rows")

    # 3. Verify Utils
    print("\n--- Verifying Utils ---")

    # Test RLE Encode/Decode
    decoded_mask = rle_decode(rle_str, (img_h, img_w))
    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Decode failed to match original mask"
    print("RLE Encode/Decode logic verified.")

    # Test Dice Coefficient
    y_true = np.array([1, 1, 0, 0])
    y_pred_perfect = np.array([1, 1, 0, 0])
    y_pred_worst = np.array([0, 0, 1, 1])

    score_perfect = dice_coef(y_pred_perfect, y_true)
    score_worst = dice_coef(y_pred_worst, y_true)

    assert np.isclose(
        score_perfect, 1.0
    ), f"Dice score should be 1.0, got {score_perfect}"
    assert np.isclose(
        score_worst, 0.0, atol=1e-5
    ), f"Dice score should be ~0.0, got {score_worst}"
    print("Dice Coefficient logic verified.")

    # 4. Verify Dataset
    print("\n--- Verifying Dataset ---")

    # Initialize Dataset (disable caching to force processing of our new fake data)
    dataset = HuBMAPDataset(
        metadata_df=metadata_df, split="train", load_cached_data=False
    )

    print(f"Dataset length (tiles): {len(dataset)}")
    assert len(dataset) > 0, "Dataset should have generated tiles."

    # Fetch one sample
    img_tensor, mask_tensor, tile_info = dataset[0]

    # Check shapes
    # Image: (3, TILE_SIZE, TILE_SIZE)
    # Mask: (1, TILE_SIZE, TILE_SIZE)
    assert img_tensor.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Image shape mismatch: {img_tensor.shape}"
    assert mask_tensor.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Mask shape mismatch: {mask_tensor.shape}"

    # Check values
    assert (
        mask_tensor.max() <= 1.0 and mask_tensor.min() >= 0.0
    ), "Mask values out of range [0, 1]"
    print("Dataset loading and tiling verified.")

    # 5. Verify Model
    print("\n--- Verifying Model ---")

    model = FPNResNet18(num_classes=Config.CLASSES)
    model.to(Config.DEVICE)

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE).to(
        Config.DEVICE
    )

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.CLASSES,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass verified.")

    # 6. Verify Training Loop
    print("\n--- Verifying Trainer ---")

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    # Use same dataset for validation for demo purposes
    val_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    trainer = Trainer(model, device=Config.DEVICE)

    # Run fit (1 epoch)
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Check if checkpoint exists
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print("Training loop completed and model saved.")

    # 7. Verify Inference
    print("\n--- Verifying Inference ---")

    # Create a dummy test metadata file pointing to the same synthetic image
    test_metadata = pd.DataFrame(
        {
            "id": ["synthetic_01"],
            "image_path": [
                os.path.join("train", img_filename)
            ],  # Reusing train image for test demo
        }
    )
    test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.csv")

    # We need to temporarily mock the metadata file location in Config or save it where InferenceRunner expects
    # InferenceRunner reads from Config.METADATA_DIR/test_metadata.csv
    # We can't easily change Config.METADATA_DIR without affecting other things,
    # but we can save our dummy test metadata there.
    # Note: The prompt says "metadata files are already generated... Do not modify".
    # However, InferenceRunner *requires* test_metadata.csv to run generate_submission.
    # To avoid modifying the provided read-only metadata, we will instantiate InferenceRunner
    # and call predict_large_image directly on our synthetic image.

    inference_runner = InferenceRunner(checkpoint_path)

    # Predict on the synthetic image
    print(f"Running inference on {img_path}...")
    prediction_rle = inference_runner.predict_large_image(img_path)

    assert isinstance(prediction_rle, str), "Prediction should return a string (RLE)"

    # Decode prediction to check if it's valid
    pred_mask = rle_decode(prediction_rle, (img_h, img_w))
    print(f"Prediction RLE Length: {len(prediction_rle)}")
    print(f"Predicted Mask Non-zero pixels: {np.sum(pred_mask)}")

    print("Inference pipeline verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
