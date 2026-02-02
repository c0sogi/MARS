import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, AverageMeter, calculate_map5
from library.dataset import WhaleDataset, get_transforms, get_class_mapping
from library.model import WhaleDenseNet
from library.loss import LabelSmoothingCrossEntropy
from library.engine import train_model


def test_utils():
    print("\n=== Testing Utilities ===")

    # 1. Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("AverageMeter: OK")

    # 2. Test MAP@5 Calculation
    # Scenario: Batch size 2.
    # Sample 0: Target 10. Preds: [10, 1, 2, 3, 4] -> Rank 1 -> Score 1.0
    # Sample 1: Target 20. Preds: [1, 20, 3, 4, 5] -> Rank 2 -> Score 0.5
    # Mean Score: (1.0 + 0.5) / 2 = 0.75

    targets = np.array([10, 20])
    # Logits: We create a dummy logits array where the target indices have high values
    # Shape: (2, 100) assuming 100 classes for this dummy test
    logits = torch.randn(2, 100)

    # Set high values for specific indices to control topk
    # Sample 0
    logits[0, 10] = 100.0
    logits[0, 1] = 90.0
    logits[0, 2] = 80.0

    # Sample 1
    logits[1, 1] = 100.0
    logits[1, 20] = 90.0  # 2nd place
    logits[1, 3] = 80.0

    score = calculate_map5(logits, targets)
    print(f"MAP@5 Score: {score}")
    assert np.isclose(
        score, 0.75
    ), f"MAP@5 calculation failed: expected 0.75, got {score}"
    print("MAP@5 Calculation: OK")


def test_dataset_and_transforms():
    print("\n=== Testing Dataset & Transforms ===")

    # Ensure class mapping cache is generated
    class_to_idx, classes = get_class_mapping(load_cached_data=False)
    print(f"Number of classes: {len(classes)}")

    # Initialize Dataset in DEBUG mode
    ds = WhaleDataset(mode="train", transform=get_transforms("train"), debug=True)
    print(f"Dataset length (Debug): {len(ds)}")

    assert len(ds) > 0, "Dataset is empty."

    # Fetch one sample
    img_tensor, label = ds[0]

    # Verify Image Shape: (C, H, W) -> (3, 320, 320)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {img_tensor.shape}"

    # Verify Label
    assert isinstance(label.item(), int), "Label is not an integer."
    assert 0 <= label.item() < Config.NUM_CLASSES, "Label index out of bounds."

    print("Dataset & Transforms: OK")


def test_model_and_loss():
    print("\n=== Testing Model & Loss ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    model = WhaleDenseNet(pretrained=False)  # False for speed in demo
    model.to(device)
    model.eval()

    # Create Dummy Batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        device
    )
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (batch_size,)).to(device)

    # Forward Pass (Training Mode with Labels)
    logits = model(dummy_input, labels=dummy_targets)

    # Check Output Shape
    assert logits.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {logits.shape}"

    # Test Loss
    criterion = LabelSmoothingCrossEntropy()
    loss = criterion(logits, dummy_targets)

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."

    # Forward Pass (Inference Mode without Labels)
    inference_logits = model(dummy_input, labels=None)
    assert inference_logits.shape == (batch_size, Config.NUM_CLASSES)

    print("Model & Loss: OK")


def run_training_pipeline():
    print("\n=== Running Training Pipeline (Demo) ===")

    # Override Config for Demo Speed
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 16  # Small batch for demo

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Run Training
    # This uses library.engine.train_model
    best_score = train_model(seed=42, debug=True)

    print(f"Training finished. Best MAP@5: {best_score:.4f}")

    # Verify Checkpoint Creation
    seed_dir = os.path.join(Config.WORKING_DIR, "seed_42")
    checkpoint_path = os.path.join(seed_dir, "model_best.pth.tar")

    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Checkpoint saved successfully.")

    return checkpoint_path


def run_inference_demo(checkpoint_path):
    print("\n=== Running Inference Demo ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = WhaleDenseNet(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Load Test Dataset (Debug mode for speed)
    test_ds = WhaleDataset(mode="test", transform=get_transforms("test"), debug=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Class Mapping to convert indices back to IDs
    # Note: In a real scenario, we load the classes.npy saved during training
    _, classes = get_class_mapping()

    results = []

    print(f"Predicting on {len(test_ds)} test images...")

    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)

            # Forward pass (Inference)
            logits = model(images, labels=None)

            # Get Top 5
            _, top5_indices = logits.topk(5, dim=1)
            top5_indices = top5_indices.cpu().numpy()

            for i, img_name in enumerate(image_names):
                indices = top5_indices[i]
                predicted_labels = [classes[idx] for idx in indices]
                prediction_string = " ".join(predicted_labels)
                results.append({"Image": img_name, "Id": prediction_string})

    # Create Submission DataFrame
    df_sub = pd.DataFrame(results)
    print("Sample Predictions:")
    print(df_sub.head())

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission", "submission.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_sub.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created."
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Unit Tests
    test_utils()
    test_dataset_and_transforms()
    test_model_and_loss()

    # 3. Integration Test (Training)
    ckpt_path = run_training_pipeline()

    # 4. Inference Test
    run_inference_demo(ckpt_path)

    print("\nAll demonstrations completed successfully.")
