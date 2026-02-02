import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SUBMISSION_PATH,
    MEGADETECTOR_PATH,
    NUM_CLASSES,
    IMAGE_SIZE,
)
from library.utils import set_seed, load_megadetector_data, calculate_class_weights
from library.dataset import create_datasets, CameraTrapDataset
from library.model import DetectorGuidedCNN
from library.trainer import Trainer


def demonstrate_utils():
    print("\n=== Demonstrating Utils ===")

    # 1. Set Seed
    print("Setting random seed...")
    set_seed(42)

    # 2. Load MegaDetector Data
    print("Loading MegaDetector data...")
    # We force load from source to verify the parsing logic, though cache is preferred in prod
    bbox_map = load_megadetector_data(MEGADETECTOR_PATH, load_cached_data=False)

    assert isinstance(bbox_map, dict), "bbox_map should be a dictionary"
    if len(bbox_map) > 0:
        first_key = next(iter(bbox_map))
        bbox = bbox_map[first_key]
        assert (
            isinstance(bbox, list) and len(bbox) == 4
        ), "Bbox should be a list of 4 floats"
        print(f"Verified bbox structure for {first_key}: {bbox}")

    # 3. Calculate Class Weights
    print("Verifying class weight calculation...")
    # Create a dummy dataframe to test logic independent of actual data loading
    dummy_data = {
        "image_id": ["img1", "img2", "img3", "img4"],
        "category_id": [0, 0, 1, 2],  # Class 0 appears twice, 1 and 2 once
    }
    dummy_df = pd.DataFrame(dummy_data)
    weights = calculate_class_weights(dummy_df, num_classes=5)

    assert isinstance(weights, torch.Tensor), "Weights should be a torch Tensor"
    assert weights.shape == (
        5,
    ), f"Weights shape mismatch. Expected (5,), got {weights.shape}"
    # Class 0 is more frequent, so it should have lower weight than class 1
    assert weights[0] < weights[1], "Frequent class should have lower weight"
    print("Class weights verification successful.")


def demonstrate_dataset():
    print("\n=== Demonstrating Dataset ===")

    # 1. Create Datasets
    # This loads the actual metadata files from ./metadata
    print("Creating datasets from metadata...")
    train_ds, val_ds, test_ds = create_datasets(load_cached_data=True)

    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"
    assert len(test_ds) > 0, "Test dataset is empty"
    print(
        f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    # 2. Verify __getitem__
    print("Verifying dataset item retrieval...")
    # Get the first item from training set
    img, label = train_ds[0]

    # Check Image
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert img.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ), f"Image shape mismatch. Expected (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {img.shape}"

    # Check Label
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.ndim == 0, "Label should be a scalar tensor"

    print(f"Sample retrieved successfully. Image shape: {img.shape}, Label: {label}")


def demonstrate_model():
    print("\n=== Demonstrating Model ===")

    # 1. Instantiate Model
    # Using pretrained=False for speed in this unit test
    print("Instantiating DetectorGuidedCNN (pretrained=False for demo)...")
    model = DetectorGuidedCNN(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()

    # 2. Forward Pass
    print("Running forward pass with dummy input...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        batch_size,
        NUM_CLASSES,
    ), f"Output shape mismatch. Expected ({batch_size}, {NUM_CLASSES}), got {output.shape}"
    print(f"Forward pass successful. Output shape: {output.shape}")


def demonstrate_trainer():
    print("\n=== Demonstrating Trainer (Fast Run) ===")

    # 1. Initialize Trainer
    # We limit max_samples to 50 to ensure the epoch finishes almost instantly
    print("Initializing Trainer with max_samples=50...")
    trainer = Trainer(load_cached_data=True, max_samples=50)

    # 2. Run Training Loop
    # We run for only 1 epoch to demonstrate the loop works
    print("Running fit() for 1 epoch...")
    trainer.fit(epochs=1)

    # 3. Generate Submission
    print("Generating submission...")
    trainer.generate_submission()

    # 4. Verify Submission File
    assert os.path.exists(
        SUBMISSION_PATH
    ), f"Submission file not found at {SUBMISSION_PATH}"

    sub_df = pd.read_csv(SUBMISSION_PATH)
    assert "Id" in sub_df.columns, "Submission missing 'Id' column"
    assert "Category" in sub_df.columns, "Submission missing 'Category' column"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"Submission generated successfully at {SUBMISSION_PATH}")
    print(f"First few rows:\n{sub_df.head()}")


if __name__ == "__main__":
    try:
        demonstrate_utils()
        demonstrate_dataset()
        demonstrate_model()
        demonstrate_trainer()
        print("\nAll demonstrations completed successfully!")
    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
