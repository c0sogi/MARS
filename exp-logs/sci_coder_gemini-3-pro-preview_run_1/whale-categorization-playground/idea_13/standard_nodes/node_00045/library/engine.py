import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, map5


def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): Optimizer instance.
        device (str or torch.device): Device to compute on.
        scheduler (LRScheduler, optional): Batch-level learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # CrossEntropyLoss with Label Smoothing
    # ArcFace outputs logits, so we use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Pass labels to ArcFace head to enforce margin
        logits = model(images, labels)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(model, dataloader, device, label_encoder):
    """
    Evaluates the model on the validation set.
    Uses Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation dataloader.
        device (str or torch.device): Device to compute on.
        label_encoder (LabelEncoder): Fitted encoder to decode predictions.

    Returns:
        tuple: (average_loss, map5_score)
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA: Original Image
            logits_orig = model(images, labels=None)

            # TTA: Horizontally Flipped Image
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, labels=None)

            # Average Logits
            logits = (logits_orig + logits_flip) / 2.0

            # Compute Loss (Proxy metric)
            loss = criterion(logits, labels)
            loss_meter.update(loss.item(), images.size(0))

            # Get Top 5 Predictions
            _, topk_indices = torch.topk(logits, k=5, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Decode Labels and Predictions
            # batch_preds is a list of lists (e.g., [['w_1', 'w_2', ...], ...])
            batch_preds = []
            for idx_row in topk_indices:
                batch_preds.append(list(label_encoder.inverse_transform(idx_row)))

            batch_labels = list(label_encoder.inverse_transform(labels_np))

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    # Compute MAP@5
    map5_score = map5(all_labels, all_preds)

    print(f"Val Loss: {loss_meter.avg}, MAP@5: {map5_score}")
    return loss_meter.avg, map5_score


def inference(model, dataloader, device, label_encoder):
    """
    Generates predictions for the test set.
    Uses Test-Time Augmentation (Horizontal Flip).
    Saves predictions to submission.csv in the configured directory.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Test dataloader.
        device (str or torch.device): Device to compute on.
        label_encoder (LabelEncoder): Fitted encoder to decode predictions.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()

    image_names_list = []
    predictions_str_list = []

    print("Starting Inference with TTA...")

    with torch.no_grad():
        for images, image_names in dataloader:
            images = images.to(device)

            # TTA: Original Image
            logits_orig = model(images, labels=None)

            # TTA: Horizontally Flipped Image
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, labels=None)

            # Average Logits
            logits = (logits_orig + logits_flip) / 2.0

            # Get Top 5 Predictions
            _, topk_indices = torch.topk(logits, k=5, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            # Decode and Format
            for i in range(len(image_names)):
                preds = label_encoder.inverse_transform(topk_indices[i])
                pred_str = " ".join(preds)

                image_names_list.append(image_names[i])
                predictions_str_list.append(pred_str)

    # Create DataFrame
    df_submission = pd.DataFrame(
        {"Image": image_names_list, "Id": predictions_str_list}
    )

    # Save to Configured Submission Directory
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    # Also save to ./submission/submission.csv if the directory exists or can be created,
    # to satisfy the specific path requirement if needed.
    alt_dir = "./submission"
    os.makedirs(alt_dir, exist_ok=True)
    alt_path = os.path.join(alt_dir, "submission.csv")
    df_submission.to_csv(alt_path, index=False)

    return df_submission
