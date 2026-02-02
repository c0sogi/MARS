import os
import torch
import pandas as pd
import numpy as np
from library.utils import AverageMeter, calculate_map5


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Optimizer.
        device (str or torch.device): Device to compute on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # ArcFace head requires labels during training to calculate margin penalty
        outputs = model(images, labels)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Training Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str or torch.device): Device to compute on.

    Returns:
        tuple: (average_loss, map5_score)
    """
    model.eval()
    loss_meter = AverageMeter()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Compute Loss
            # Pass labels to get the margin-adjusted logits for loss consistency
            outputs_loss = model(images, labels)
            loss = criterion(outputs_loss, labels)
            loss_meter.update(loss.item(), images.size(0))

            # 2. Compute Predictions for MAP@5
            # Pass label=None to get raw cosine similarities (inference mode)
            # This returns the cosine similarity between embeddings and class centers
            outputs_eval = model(images, label=None)

            # Get top 5 predictions
            _, indices = torch.topk(outputs_eval, k=5, dim=1)

            preds_list.append(indices.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Calculate MAP@5
    map5_score = calculate_map5(predictions, targets)

    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation MAP@5: {map5_score}")

    return loss_meter.avg, map5_score


def generate_submission(model, dataloader, test_df, class_map, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader (must be sequential/shuffle=False).
        test_df (pd.DataFrame): Test metadata containing 'image' column.
        class_map (dict): Mapping from hotel_id to label index.
        device (str or torch.device): Device to compute on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    all_preds = []

    # Invert class map: index -> hotel_id
    # class_map is {hotel_id: index}, we need {index: hotel_id}
    index_to_hotel = {v: k for k, v in class_map.items()}

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # Inference: get raw cosine similarities
            outputs = model(images, label=None)

            # Get top 5 indices
            _, indices = torch.topk(outputs, k=5, dim=1)

            all_preds.append(indices.cpu().numpy())

    # Concatenate all predictions
    predictions_indices = np.concatenate(all_preds, axis=0)

    # Convert indices to space-delimited hotel IDs
    formatted_preds = []
    for row in predictions_indices:
        # Map each index in the row to its hotel_id
        hotel_ids = [str(index_to_hotel[idx]) for idx in row]
        formatted_preds.append(" ".join(hotel_ids))

    # Create Submission DataFrame
    # Note: We rely on the dataloader preserving the order of test_df
    submission = pd.DataFrame({"image": test_df["image"], "hotel_id": formatted_preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
