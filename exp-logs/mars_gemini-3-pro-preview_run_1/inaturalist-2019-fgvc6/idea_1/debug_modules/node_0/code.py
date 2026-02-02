import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, AverageMeter
from library.dataset import INatDataset, get_transforms
from library.model import get_mobilenet_model
from library.train import accuracy


def main():
    print("Starting library demonstration and verification script...")

    # --------------------------------------------------------------------------
    # 1. Setup and Reproducibility
    # --------------------------------------------------------------------------
    print("\n[1] Setting up environment...")
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Classes
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utility classes...")
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Expected average: ((10*2) + (20*2)) / 4 = 15.0
    assert (
        meter.avg == 15.0
    ), f"AverageMeter logic incorrect. Expected 15.0, got {meter.avg}"
    print("AverageMeter verified.")

    # --------------------------------------------------------------------------
    # 3. Verify Dataset and Transforms
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and Transforms...")
    assert os.path.exists(Config.TRAIN_CSV), "Train metadata CSV missing."
    assert os.path.exists(Config.TEST_CSV), "Test metadata CSV missing."

    # Instantiate Training Dataset
    train_transform = get_transforms(stage="train")
    full_train_dataset = INatDataset(
        csv_path=Config.TRAIN_CSV, mode="train", transform=train_transform
    )

    assert len(full_train_dataset) > 0, "Training dataset is empty."

    # Check a single sample
    img, target = full_train_dataset[0]

    # Verify Image Tensor Shape (C, H, W) -> (3, 224, 224)
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"

    # Verify Target is a scalar tensor
    assert isinstance(target, torch.Tensor), "Target is not a tensor."
    assert target.ndim == 0, "Target should be a scalar tensor."

    # Instantiate Test Dataset
    test_transform = get_transforms(stage="test")
    full_test_dataset = INatDataset(
        csv_path=Config.TEST_CSV, mode="test", transform=test_transform
    )

    # Test dataset returns (image, image_id)
    img_test, img_id = full_test_dataset[0]
    assert isinstance(
        img_id, (int, np.integer)
    ), f"Test image ID should be integer, got {type(img_id)}"
    print("Dataset and Transforms verified.")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    # Initialize model (pretrained=False for speed in this demo, though Config uses True)
    model = get_mobilenet_model(
        pretrained=False, num_classes=Config.NUM_CLASSES, device=device
    )

    # Check the classifier head replacement
    # MobileNetV3 classifier is a Sequential block, we replaced the last Linear layer
    last_layer = model.classifier[-1]
    assert isinstance(last_layer, nn.Linear), "Last layer is not Linear."
    assert (
        last_layer.out_features == Config.NUM_CLASSES
    ), f"Output features mismatch. Expected {Config.NUM_CLASSES}, got {last_layer.out_features}"

    # Dummy forward pass to check dimensions
    batch_size_demo = 2
    dummy_input = torch.randn(batch_size_demo, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        device
    )
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        batch_size_demo,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected ({batch_size_demo}, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model architecture verified.")

    # --------------------------------------------------------------------------
    # 5. Verify Training Logic (on Subset)
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Training Logic (Subset)...")

    # Create a small subset for rapid execution
    subset_indices = list(range(10))
    train_subset = Subset(full_train_dataset, subset_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small test
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    # Fetch one batch
    images, targets = next(iter(train_loader))
    images, targets = images.to(device), targets.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, targets)

    # Check loss validity
    assert not torch.isnan(loss), "Loss is NaN."
    print(f"Batch Loss: {loss.item():.4f}")

    # Check accuracy calculation
    acc1 = accuracy(outputs, targets, topk=(1,))[0]
    print(f"Batch Top-1 Accuracy: {acc1.item():.2f}%")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Training step verified.")

    # --------------------------------------------------------------------------
    # 6. Verify Prediction Logic (on Subset)
    # --------------------------------------------------------------------------
    print("\n[6] Verifying Prediction Logic (Subset)...")

    test_subset = Subset(full_test_dataset, subset_indices)
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=0)

    model.eval()
    predictions = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Get Top 5 predictions
            _, top5_indices = torch.topk(outputs, k=5, dim=1)
            top5_indices = top5_indices.cpu().numpy()

            # Handle Tensor image_ids
            if isinstance(image_ids, torch.Tensor):
                image_ids = image_ids.numpy()

            for img_id, preds in zip(image_ids, top5_indices):
                pred_str = " ".join(map(str, preds))
                predictions.append({"id": img_id, "predicted": pred_str})

    # Verify output format
    assert len(predictions) == 10, f"Expected 10 predictions, got {len(predictions)}"
    sample_pred = predictions[0]
    assert (
        "id" in sample_pred and "predicted" in sample_pred
    ), "Prediction dictionary keys mismatch."

    # Verify predicted string format (5 integers separated by space)
    pred_values = sample_pred["predicted"].split()
    assert (
        len(pred_values) == 5
    ), f"Expected 5 top-k predictions, got {len(pred_values)}"
    assert all(
        x.isdigit() for x in pred_values
    ), "Prediction string contains non-digits."

    print("Prediction logic verified.")

    # --------------------------------------------------------------------------
    # 7. Output Verification
    # --------------------------------------------------------------------------
    print("\n[7] Generating sample submission file...")
    df = pd.DataFrame(predictions)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    df.to_csv(output_path, index=False)

    assert os.path.exists(output_path), "Submission file was not created."
    print(f"Sample submission saved to {output_path}")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
