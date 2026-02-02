import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.utils import get_device
from library.dataset import AppleDataset, get_transforms, TARGET_COLS
from library.model import ResNet34Baseline


def generate_predictions(model, loader, device):
    """
    Generates predictions for a given model and dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: (predictions, image_ids)
            predictions (np.ndarray): Array of probabilities with shape (n_samples, n_classes).
            image_ids (list): List of image IDs corresponding to predictions.
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    if len(all_probs) > 0:
        return np.concatenate(all_probs), all_ids
    else:
        return np.array([]), []


def predict_and_submit(
    model_path,
    test_metadata_path="./metadata/test_metadata.csv",
    input_dir="./input",
    output_path="./submission/submission.csv",
    batch_size=32,
    device=None,
):
    """
    Loads a trained model, generates predictions on the test set, and saves the submission file.

    Args:
        model_path (str): Path to the saved model state dict.
        test_metadata_path (str): Path to test metadata CSV.
        input_dir (str): Path to image directory.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device to use. Defaults to auto-detect.
    """
    if device is None:
        device = get_device()

    # 1. Load Model Architecture
    # We must match the num_classes used during training.
    # TARGET_COLS is defined in library.dataset
    num_classes = len(TARGET_COLS)
    # Pretrained is False because we are loading custom weights
    model = ResNet34Baseline(num_classes=num_classes, pretrained=False)

    # 2. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 3. Prepare Data
    test_dataset = AppleDataset(
        metadata_path=test_metadata_path,
        transform=get_transforms("test", image_size=256),
        input_dir=input_dir,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 4. Generate Predictions
    print(f"Generating predictions using model at {model_path}...")
    probs, image_ids = generate_predictions(model, test_loader, device)

    # 5. Format Submission
    # Create DataFrame
    # Columns must be: image_id, healthy, multiple_diseases, rust, scab
    # TARGET_COLS order in dataset.py is ["healthy", "multiple_diseases", "rust", "scab"]
    df_sub = pd.DataFrame(probs, columns=TARGET_COLS)
    df_sub.insert(0, "image_id", image_ids)

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
