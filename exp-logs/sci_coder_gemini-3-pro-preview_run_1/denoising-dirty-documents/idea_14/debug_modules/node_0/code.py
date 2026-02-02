import os
import shutil
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import cv2

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model
from library import engine
from library import inference


def main():
    print("=== Starting Demonstration of Denoising Pipeline ===")

    # 1. Setup and Configuration
    # We will use the default working directory defined in config.py
    # Ensure it exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    utils.set_seed(42)
    device = config.DEVICE
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Demonstrate library.utils
    # -------------------------------------------------------------------------
    print("\n--- Testing library.utils ---")

    # Test pad_image_to_multiple
    dummy_img = np.random.rand(100, 100).astype(np.float32)
    padded_img, pads = utils.pad_image_to_multiple(dummy_img, multiple=8)

    # 100 is not divisible by 8. Next multiple is 104.
    # 104 - 100 = 4 padding needed.
    expected_shape = (104, 104)
    assert (
        padded_img.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {padded_img.shape}"
    print(
        f"Padding successful: {dummy_img.shape} -> {padded_img.shape} with pads {pads}"
    )

    # Test unpad_image
    unpadded_img = utils.unpad_image(padded_img, dummy_img.shape)
    assert (
        unpadded_img.shape == dummy_img.shape
    ), "Unpadding failed to restore original shape"
    assert np.allclose(dummy_img, unpadded_img), "Unpadded content mismatch"
    print("Unpadding successful.")

    # Test calculate_rmse
    rmse = utils.calculate_rmse(dummy_img, dummy_img)
    assert rmse == 0.0, "RMSE of identical arrays should be 0"

    rmse_diff = utils.calculate_rmse(np.zeros((5, 5)), np.ones((5, 5)))
    assert np.isclose(rmse_diff, 1.0), "RMSE of 0s and 1s should be 1"
    print("RMSE calculation verified.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate library.dataset
    # -------------------------------------------------------------------------
    print("\n--- Testing library.dataset ---")

    # We will use the metadata files provided in ./metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Define temporary cache paths for this demo to avoid messing with real training caches if they exist
    # or to ensure we test the caching logic.
    demo_train_cache = os.path.join(config.WORKING_DIR, "demo_train_cache.npz")
    demo_val_cache = os.path.join(config.WORKING_DIR, "demo_val_cache.npz")

    # Remove if exist to force creation
    if os.path.exists(demo_train_cache):
        os.remove(demo_train_cache)
    if os.path.exists(demo_val_cache):
        os.remove(demo_val_cache)

    print("Initializing Training Dataset...")
    train_ds = dataset.DenoisingDataset(
        metadata_df=train_df,
        root_dir=config.INPUT_DIR,
        mode="train",
        patch_size=config.PATCH_SIZE,
        cache_path=demo_train_cache,
        load_cached_data=True,
    )

    assert len(train_ds) > 0, "Training dataset is empty"
    print(f"Training Dataset Size: {len(train_ds)}")

    # Check item structure (Train returns noisy, clean tensors)
    sample = train_ds[0]
    if isinstance(sample, tuple):
        # Expect (noisy, clean)
        noisy_t, clean_t = sample
        assert (
            noisy_t.ndim == 3 and noisy_t.shape[0] == 1
        ), "Noisy tensor shape incorrect"
        assert (
            clean_t.ndim == 3 and clean_t.shape[0] == 1
        ), "Clean tensor shape incorrect"
        # Check patch size
        assert (
            noisy_t.shape[1] == config.PATCH_SIZE
            and noisy_t.shape[2] == config.PATCH_SIZE
        ), f"Patch size mismatch. Expected {config.PATCH_SIZE}, got {noisy_t.shape[1:]}"
        print("Training sample structure verified.")
    else:
        # Should be tuple if clean data exists (which it does for train)
        raise AssertionError("Dataset __getitem__ did not return tuple for train mode")

    print("Initializing Validation Dataset...")
    val_ds = dataset.DenoisingDataset(
        metadata_df=val_df,
        root_dir=config.INPUT_DIR,
        mode="val",
        cache_path=demo_val_cache,
        load_cached_data=True,
    )

    # Check item structure (Val returns noisy, clean, meta)
    val_sample = val_ds[0]
    assert (
        len(val_sample) == 3
    ), "Validation sample should have 3 elements (noisy, clean, meta)"
    v_noisy, v_clean, v_meta = val_sample

    # Check padding logic in val
    # The original images are likely not multiples of 8, so padding should have happened
    assert (
        v_noisy.shape[1] % 8 == 0 and v_noisy.shape[2] % 8 == 0
    ), "Validation image not padded to multiple of 8"
    print("Validation sample structure verified.")

    # Test DataLoaders
    # Use a small batch size for demo
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=4, num_workers=2, load_cached=True
    )
    print("DataLoaders initialized successfully.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate library.model
    # -------------------------------------------------------------------------
    print("\n--- Testing library.model ---")

    net = model.WideBottleneckUNet(n_channels=1, n_classes=1).to(device)

    # Create a dummy input tensor matching batch size and patch size
    dummy_input = torch.randn(2, 1, 160, 160).to(device)

    # Forward pass
    output = net(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
    print("Model forward pass successful. Output shape matches input.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate library.engine (Training)
    # -------------------------------------------------------------------------
    print("\n--- Testing library.engine (Training) ---")

    # We will train for just 1 epoch to demonstrate the loop and saving mechanism
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Use a specific seed for this demo model
    demo_seed = 42
    save_path = config.get_checkpoint_path(demo_seed)

    print(f"Training for 1 epoch (Demo)...")
    best_rmse = engine.fit(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=1,  # Override config.NUM_EPOCHS for speed
        save_path=save_path,
    )

    assert os.path.exists(save_path), f"Model checkpoint not found at {save_path}"
    print(
        f"Training demo complete. Model saved to {save_path}. Best RMSE: {best_rmse:.4f}"
    )

    # -------------------------------------------------------------------------
    # 6. Demonstrate library.inference
    # -------------------------------------------------------------------------
    print("\n--- Testing library.inference ---")

    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Run ensemble inference with the single model we just trained
    # This tests the TTA logic, model loading, and submission generation
    inference.run_ensemble_inference(seeds=[demo_seed], output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify submission format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    expected_cols = ["id", "value"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"
    assert not sub_df.isnull().values.any(), "Submission contains null values"

    # Verify ID format (image_row_col)
    sample_id = sub_df.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) >= 3, f"ID format incorrect: {sample_id}"

    print("Inference and submission generation verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
