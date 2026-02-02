import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.dataset import CovidDataset
from library.model import MultiTaskEfficientDet
from library.trainer import Trainer
from library.utils import seed_everything, collate_fn


# 1. Configuration Override for Demo
class DemoConfig(Config):
    """
    Configuration for the demo run.
    Reduces epochs and ensures paths are correct.
    """

    # Run for only 1 epoch to save time
    EPOCHS = 1
    # Adjust batch size for speed/memory safety during demo
    BATCH_SIZE = 8
    # Reduce workers to minimize overhead
    NUM_WORKERS = 2
    # Ensure patience doesn't interfere with single epoch run
    PATIENCE = 1

    # We will use the standard paths defined in Config
    # SUBMISSION_PATH defaults to ./submission/submission.csv


def test_dataset_logic():
    """
    Verifies that the dataset loads correctly and produces expected shapes.
    """
    print("\n=== Testing Dataset Logic ===")

    # Instantiate training dataset
    # load_cached_data=False to ensure we test the parsing logic at least once,
    # though True is faster if cache exists.
    dataset = CovidDataset("train", load_cached_data=True)

    print(f"Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty."

    # Fetch one sample
    image_tensor, target, image_id = dataset[0]

    # Verify Image Tensor
    # Shape should be (3, IMG_SIZE, IMG_SIZE) -> (3, 640, 640)
    print(f"Sample image shape: {image_tensor.shape}")
    assert image_tensor.ndim == 3, "Image should be 3D tensor (C, H, W)"
    assert image_tensor.shape[0] == 3, "Image should have 3 channels"
    assert (
        image_tensor.shape[1] == Config.IMG_SIZE
    ), f"Height should be {Config.IMG_SIZE}"
    assert (
        image_tensor.shape[2] == Config.IMG_SIZE
    ), f"Width should be {Config.IMG_SIZE}"

    # Verify Target Dictionary
    expected_keys = {"boxes", "labels", "study_label", "image_id", "study_id"}
    assert expected_keys.issubset(
        target.keys()
    ), f"Target missing keys. Found: {target.keys()}"

    # Verify Shapes
    assert isinstance(target["boxes"], torch.Tensor), "Boxes should be a tensor"
    assert isinstance(target["labels"], torch.Tensor), "Labels should be a tensor"
    assert isinstance(
        target["study_label"], torch.Tensor
    ), "Study label should be a tensor"

    print("Dataset verification passed.")
    return dataset


def test_model_logic(dataset):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n=== Testing Model Logic ===")

    device = torch.device(DemoConfig.DEVICE)
    model = MultiTaskEfficientDet(DemoConfig).to(device)

    # Create a small dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Get a batch
    images, targets, ids = next(iter(loader))
    images = images.to(device)
    targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]

    # 1. Test Training Mode (Loss Calculation)
    model.train()
    loss_dict = model(images, targets)

    print(f"Loss keys: {loss_dict.keys()}")
    assert "loss_cls" in loss_dict, "Missing classification loss"
    assert "loss_box" in loss_dict, "Missing regression loss"
    assert "loss_study" in loss_dict, "Missing study loss"

    total_loss = sum(loss_dict.values())
    assert not torch.isnan(total_loss), "Loss should not be NaN"

    # 2. Test Eval Mode (Inference)
    model.eval()
    with torch.no_grad():
        detections = model(images)

    assert len(detections) == 2, "Should have detections for each image in batch"
    sample_det = detections[0]
    assert "boxes" in sample_det
    assert "scores" in sample_det
    assert "study_probs" in sample_det

    print("Model verification passed.")


def run_training_demo():
    """
    Runs the Trainer with debug=True to use a subset of data.
    """
    print("\n=== Running Training Demo ===")

    # Initialize Trainer with DemoConfig
    trainer = Trainer(config=DemoConfig)

    # Run fit with debug=True
    # This uses a small subset (100 train, 50 val) and runs for DemoConfig.EPOCHS (1)
    print("Starting training loop (debug mode)...")
    trainer.fit(debug=True)

    print("Training finished.")

    # Run prediction on test set
    print("Generating predictions on test set...")
    trainer.predict()

    print("Inference finished.")


def validate_submission():
    """
    Validates the generated submission file.
    """
    print("\n=== Validating Submission ===")

    sub_path = DemoConfig.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df = pd.read_csv(sub_path)
    print(f"Submission shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check columns
    assert "id" in df.columns, "Missing 'id' column"
    assert "PredictionString" in df.columns, "Missing 'PredictionString' column"

    # Check content
    assert len(df) > 0, "Submission file is empty"

    # Check format of a sample prediction string
    sample_pred = df.iloc[0]["PredictionString"]
    assert isinstance(sample_pred, str), "PredictionString should be a string"

    # Check that we have both study and image rows
    # Study IDs end with _study, Image IDs end with _image
    has_study = df["id"].str.contains("_study").any()
    has_image = df["id"].str.contains("_image").any()

    assert has_study, "Submission should contain study-level predictions"
    assert has_image, "Submission should contain image-level predictions"

    print("Submission verification passed.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed_everything(DemoConfig.SEED)

    try:
        # 1. Verify Dataset
        dataset = test_dataset_logic()

        # 2. Verify Model
        test_model_logic(dataset)

        # 3. Run Training and Inference
        run_training_demo()

        # 4. Validate Output
        validate_submission()

        print("\nAll demo steps completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during the demo: {e}")
        # Explicitly raise to fail the run if something goes wrong
        raise e
