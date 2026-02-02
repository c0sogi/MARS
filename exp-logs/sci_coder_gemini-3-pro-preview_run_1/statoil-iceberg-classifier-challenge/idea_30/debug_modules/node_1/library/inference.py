import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.utils import set_seed


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Exhaustive Closed-Group Test-Time Augmentation (TTA).
    The four groups are: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test set.
        device (torch.device): Device to perform inference on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities (0 to 1).
    """
    set_seed(42)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both labeled (val) and unlabeled (test) dataloaders
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch

            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip (dim 3 for NCHW)
            images_h = torch.flip(images, dims=[3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # 3. Vertical Flip (dim 2 for NCHW)
            images_v = torch.flip(images, dims=[2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # 4. Rotate 180 (Horizontal + Vertical Flip)
            images_hv = torch.flip(images, dims=[2, 3])
            out4 = model(images_hv, angles)
            prob4 = torch.sigmoid(out4)

            # Average probabilities
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            all_preds.append(avg_prob.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def validate_model(model, dataloader, device):
    """
    Evaluates the model on the validation set using TTA and calculates Log Loss.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Validation DataLoader (must yield labels).
        device (torch.device): Device.

    Returns:
        float: The calculated Log Loss.
        np.ndarray: The predictions.
        np.ndarray: The targets.
    """
    preds = predict_with_tta(model, dataloader, device)

    # Extract targets
    targets = []
    for batch in dataloader:
        if len(batch) == 3:
            _, _, t = batch
            targets.append(t.numpy())
        else:
            raise ValueError("DataLoader must provide labels for validation.")

    targets = np.concatenate(targets).flatten()

    # Calculate Log Loss
    # eps=1e-15 is standard for log_loss to avoid log(0)
    loss = log_loss(targets, preds, eps=1e-15)

    return loss, preds, targets


def load_test_ids(metadata_dir="./metadata"):
    """
    Loads the test IDs from the metadata file to ensure alignment with predictions.

    Args:
        metadata_dir (str): Directory containing test_metadata.csv.

    Returns:
        np.ndarray: Array of test IDs.
    """
    meta_path = os.path.join(metadata_dir, "test_metadata.csv")
    df = pd.read_csv(meta_path)
    return df["id"].values


def save_submission(predictions, test_ids, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Array of predicted probabilities.
        test_ids (np.ndarray): Array of corresponding image IDs.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
