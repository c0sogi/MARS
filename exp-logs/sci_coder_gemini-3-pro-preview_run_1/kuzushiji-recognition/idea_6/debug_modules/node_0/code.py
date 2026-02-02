import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    IMG_SIZE,
    DOWN_RATIO,
    NUM_CLASSES,
)
from library.utils import (
    seed_everything,
    load_and_parse_metadata,
    collate_fn,
    post_process_coords,
)
from library.dataset import KuzushijiDataset
from library.model import ConvNextCenterNet
from library.loss import CenterNetLoss
from library.engine import run_training, predict


def demo_utils():
    print("\n=== 1. Demonstrating Utils ===")

    # Test Metadata Parsing
    print(f"Parsing metadata from {TRAIN_METADATA_PATH}...")
    metadata = load_and_parse_metadata(TRAIN_METADATA_PATH, load_cached_data=False)
    assert isinstance(metadata, list), "Metadata should be a list"
    assert len(metadata) > 0, "Metadata should not be empty"
    print(f"Successfully parsed {len(metadata)} entries.")

    sample = metadata[0]
    assert (
        "image_id" in sample and "annotations" in sample
    ), "Incorrect metadata structure"

    # Test Coordinate Post-processing
    # Simulate a 1024x1024 input derived from a 2048x1024 original image (Aspect Ratio 2:1)
    # Padding would be added to the width to make it square before resizing?
    # Logic in transforms: LongestMaxSize(1024) -> 1024x512. PadIfNeeded -> 1024x1024 (pads bottom/top or sides).
    # Let's test the math function directly.

    input_coord = 512.0  # Center of model input
    orig_shape = (2048, 1024)  # H, W

    # In post_process_coords:
    # scale = min(1024/2048, 1024/1024) = 0.5
    # new_h = 2048 * 0.5 = 1024
    # new_w = 1024 * 0.5 = 512
    # pad_y = (1024 - 1024)/2 = 0
    # pad_x = (1024 - 512)/2 = 256
    # orig_x = (512 - 256) / 0.5 = 256 / 0.5 = 512
    # orig_y = (512 - 0) / 0.5 = 1024

    ox, oy = post_process_coords(input_coord, input_coord, orig_shape, input_size=1024)

    print(
        f"Coordinate Mapping Test: Input(512, 512) -> Original({ox}, {oy}) for Shape {orig_shape}"
    )
    assert ox == 512.0 and oy == 1024.0, f"Coordinate mapping failed. Got {ox}, {oy}"
    print("Coordinate mapping verified.")


def demo_dataset_and_loader():
    print("\n=== 2. Demonstrating Dataset & DataLoader ===")

    # Initialize Dataset in Debug mode (loads small subset)
    ds = KuzushijiDataset(TRAIN_METADATA_PATH, split="train", debug=True)
    print(f"Dataset initialized (Debug Mode). Length: {len(ds)}")

    # Fetch one sample
    sample = ds[0]
    img = sample["image"]
    target = sample["target"]

    # Verify Image Shape (Channels, H, W)
    assert img.shape == (3, IMG_SIZE, IMG_SIZE), f"Unexpected image shape: {img.shape}"

    # Verify Target Shapes
    # Heatmap should be (1, H/4, W/4)
    out_size = IMG_SIZE // DOWN_RATIO
    assert target["hm"].shape == (
        1,
        out_size,
        out_size,
    ), f"Unexpected heatmap shape: {target['hm'].shape}"

    print("Single sample shapes verified.")

    # Test DataLoader with Collate
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch["image"].shape == (
        4,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), "Batch image shape incorrect"
    assert batch["target"]["hm"].shape == (
        4,
        1,
        out_size,
        out_size,
    ), "Batch heatmap shape incorrect"
    assert isinstance(batch["image_id"], list), "image_id should be a list"

    print("DataLoader batch collation verified.")
    return batch


def demo_model_and_loss(batch):
    print("\n=== 3. Demonstrating Model & Loss ===")

    # Initialize Model
    model = ConvNextCenterNet().to(DEVICE)
    print(f"Model {model.__class__.__name__} initialized on {DEVICE}.")

    # Move batch to device
    images = batch["image"].to(DEVICE)
    targets = batch["target"]
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            targets[k] = v.to(DEVICE)
    batch["target"] = targets

    # Forward Pass
    outputs = model(images)

    # Verify Output Shapes
    out_h = IMG_SIZE // DOWN_RATIO
    assert outputs["hm"].shape == (4, 1, out_h, out_h), "Output heatmap shape incorrect"
    assert outputs["cls_logits"].shape == (
        4,
        NUM_CLASSES,
        out_h,
        out_h,
    ), "Output class logits shape incorrect"
    assert outputs["wh"].shape == (4, 2, out_h, out_h), "Output WH shape incorrect"
    assert outputs["reg"].shape == (
        4,
        2,
        out_h,
        out_h,
    ), "Output regression shape incorrect"
    print("Forward pass successful. Output shapes verified.")

    # Loss Calculation
    criterion = CenterNetLoss()
    loss, stats = criterion(outputs, batch)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss > 0, "Loss should be positive"

    for k, v in stats.items():
        assert not torch.isnan(v), f"Loss component {k} is NaN"

    print("Loss calculation verified.")


def demo_engine():
    print("\n=== 4. Demonstrating Engine (Train/Predict) ===")

    # Run Training
    # Using debug=True limits the dataset to 100 images
    # Using epochs=1 ensures quick execution
    print("Starting short training run (1 epoch, debug mode)...")
    best_model_path = run_training(debug=True, epochs=1)

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training finished. Model saved at {best_model_path}")

    # Run Inference
    print("Starting inference run (debug mode)...")
    predict(model_path=best_model_path, debug=True)

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    # Validate Submission Content
    df = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission generated with {len(df)} rows.")
    assert (
        "image_id" in df.columns and "labels" in df.columns
    ), "Submission columns missing"

    # Check if we have at least some predictions (though with 1 epoch they might be sparse/poor)
    # We just check the file structure is valid.
    print("Engine workflow verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        # 1. Utils
        demo_utils()

        # 2. Dataset
        batch = demo_dataset_and_loader()

        # 3. Model & Loss
        demo_model_and_loss(batch)

        # 4. Engine
        demo_engine()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Print full traceback for debugging if needed, but keeping it simple here
        import traceback

        traceback.print_exc()
        exit(1)
