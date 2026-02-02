import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided library files
from library.utils import set_seed
from library.dataset import HubmapDataset
from library.model import ConvNeXtUNetPlusPlus
from library.trainer import Trainer
from library.inference import InferencePipeline


def main():
    # 1. Setup and Configuration
    print("--- Setting up configuration ---")
    set_seed(42)

    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    os.makedirs(WORKING_DIR, exist_ok=True)

    # Configuration for training
    # We use a very small number of epochs and a subset of data for demonstration speed
    train_config = {
        "lr": 1e-4,
        "batch_size": 2,
        "num_epochs": 2,  # Short run
        "weight_decay": 1e-2,
        "working_dir": WORKING_DIR,
        "num_workers": 2,
        "tile_size": 512,
        "patience": 2,
    }

    # 2. Dataset Instantiation & Verification
    print("\n--- Initializing Datasets ---")

    # Load metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Use a small subset for speed (e.g., first 2 images)
    train_meta_subset = train_meta.head(2).copy()
    val_meta_subset = val_meta.head(1).copy()

    # Define augmentations
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Instantiate Datasets
    # Note: This will generate/load cache for tiles
    train_dataset = HubmapDataset(
        metadata_df=train_meta_subset,
        root_dir=INPUT_DIR,
        transform=train_transform,
        tile_size=train_config["tile_size"],
        stride=train_config["tile_size"],  # Non-overlapping for speed
        split="train",
        cache_dir=CACHE_DIR,
    )

    val_dataset = HubmapDataset(
        metadata_df=val_meta_subset,
        root_dir=INPUT_DIR,
        transform=val_transform,
        tile_size=train_config["tile_size"],
        stride=train_config["tile_size"],
        split="validation",
        cache_dir=CACHE_DIR,
    )

    print(f"Train tiles: {len(train_dataset)}, Val tiles: {len(val_dataset)}")

    # Verify Dataset Output
    if len(train_dataset) > 0:
        img, mask = train_dataset[0]
        # Check shapes: Image (3, H, W), Mask (1, H, W)
        assert img.shape == (
            3,
            train_config["tile_size"],
            train_config["tile_size"],
        ), f"Expected image shape (3, {train_config['tile_size']}, {train_config['tile_size']}), got {img.shape}"
        assert mask.shape == (
            1,
            train_config["tile_size"],
            train_config["tile_size"],
        ), f"Expected mask shape (1, {train_config['tile_size']}, {train_config['tile_size']}), got {mask.shape}"
        print("Dataset verification passed.")
    else:
        print(
            "Warning: Dataset subset resulted in 0 tiles (likely due to tissue threshold). Skipping shape check."
        )

    # 3. Model Initialization & Verification
    print("\n--- Initializing Model ---")
    model = ConvNeXtUNetPlusPlus(num_classes=1, pretrained=True)

    # Verify Model Output
    # Create a dummy batch
    dummy_input = torch.randn(
        2, 3, train_config["tile_size"], train_config["tile_size"]
    )
    with torch.no_grad():
        outputs = model(dummy_input)

    # Model returns a list of tensors for deep supervision
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 4, f"Expected 4 output scales, got {len(outputs)}"
    assert outputs[0].shape == (
        2,
        1,
        train_config["tile_size"],
        train_config["tile_size"],
    ), f"Output shape mismatch. Expected (2, 1, {train_config['tile_size']}, {train_config['tile_size']}), got {outputs[0].shape}"
    print("Model verification passed.")

    # 4. Training Loop
    print("\n--- Starting Training Loop ---")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
    )

    trainer.fit()

    # Verify model checkpoint exists
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("Training complete and checkpoint verified.")

    # 5. Inference Pipeline
    print("\n--- Starting Inference ---")

    inference_config = {
        "tile_size": 512,  # Using smaller size for demo speed
        "stride": 256,
        "batch_size": 4,
        "num_classes": 1,
        "model_path": best_model_path,
        "input_dir": INPUT_DIR,
        "submission_dir": WORKING_DIR,
        "working_dir": CACHE_DIR,
    }

    pipeline = InferencePipeline(config=inference_config)

    # Run inference on test set
    # Note: This reads from metadata/test.csv
    pipeline.run()

    # Verify Submission
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head())

    # Check columns
    assert (
        "id" in sub_df.columns and "predicted" in sub_df.columns
    ), "Submission file missing required columns."

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
