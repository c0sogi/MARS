import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.optim.swa_utils import update_bn
from library.utils import calc_log_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, mixup_fn=None):
    """
    Trains the model for one epoch.
    Handles both standard training and Mixup/CutMix training via the mixup_fn argument.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    # CrossEntropyLoss in PyTorch handles:
    # 1. Target = Class Indices (LongTensor) -> Standard Classification
    # 2. Target = Probabilities (FloatTensor) -> Soft Target Classification (for Mixup)
    criterion = nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup/CutMix if function is provided
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} Training Loss: {avg_loss}")
    return avg_loss


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set and calculates Log Loss.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate Log Loss
    metric = calc_log_loss(all_labels, all_preds)

    # Print full precision as requested
    print(f"Validation Log Loss: {metric}")

    return metric


def predict(model, data_loader, device, use_tta=False):
    """
    Generates predictions for the test set.
    Supports Test Time Augmentation (Horizontal Flip).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            # Standard forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            if use_tta:
                # Horizontal Flip TTA
                # Flip along width dimension (dim 3: B, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = F.softmax(outputs_flipped, dim=1)

                # Average predictions
                probs = (probs + probs_flipped) / 2.0

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def update_swa_model(swa_model, model):
    """
    Updates the SWA model parameters with the current model's parameters.
    """
    swa_model.update_parameters(model)


def update_bn_statistics(swa_model, data_loader, device):
    """
    Updates Batch Normalization statistics for the SWA model.
    """
    update_bn(data_loader, swa_model, device=device)


def save_submission(predictions, test_metadata_path, label_map, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.array): (N_samples, N_classes) probability matrix.
        test_metadata_path (str): Path to test.csv to retrieve IDs.
        label_map (dict): Mapping {breed: index} to construct headers.
        output_path (str): Destination path for submission.csv.
    """
    # Load test IDs
    test_df = pd.read_csv(test_metadata_path)
    ids = test_df["id"].values

    # Sort breeds by index to ensure correct column order
    # label_map is {breed_name: index}
    sorted_breeds = sorted(label_map.keys(), key=lambda x: label_map[x])

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=sorted_breeds)
    submission_df.insert(0, "id", ids)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
