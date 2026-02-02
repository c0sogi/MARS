import torch
import numpy as np
from library import utils
from library import config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training dataloader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # WhaleDenseNet accepts labels to calculate ArcFace margin loss
        outputs = model(images, labels)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate loss (loss.item() is the mean loss of the batch)
        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, class_mapping=None):
    """
    Evaluates the model on the validation set using TTA (Horizontal Flip) and MAP@5.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to run on.
        class_mapping (np.ndarray, optional): Array of class names (not used for calculation
                                              since we use indices, but kept for interface).

    Returns:
        float: MAP@5 score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # labels are indices here

            # TTA View 1: Original
            # Passing labels=None to ArcFace head returns scaled cosine similarities (logits)
            logits_1 = model(images, labels=None)

            # TTA View 2: Horizontal Flip
            # Dim 3 is width (B, C, H, W)
            images_flipped = torch.flip(images, dims=[3])
            logits_2 = model(images_flipped, labels=None)

            # Average the logits
            logits = (logits_1 + logits_2) / 2.0

            # Get Top 5 predictions
            # logits shape: (Batch, NumClasses)
            _, top_indices = torch.topk(logits, k=5, dim=1)

            # Store predictions and targets
            all_preds.extend(top_indices.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Compute MAP@5
    # utils.map5 handles normalization of 'actual' list to list of lists
    score = utils.map5(all_targets, all_preds)

    print(f"Validation MAP@5: {score}")
    return score


def inference_tta(model, dataloader, device):
    """
    Performs inference on the test set using TTA (Horizontal Flip).

    Args:
        model (nn.Module): The model to use.
        dataloader (DataLoader): Test dataloader.
        device (str): Device to run on.

    Returns:
        tuple: (image_ids, logits)
            image_ids (list): List of image filenames.
            logits (np.ndarray): Array of shape (N, NumClasses) containing averaged logits.
    """
    model.eval()

    all_logits = []
    all_image_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Test dataset returns (image, img_name)
            images, img_names = batch
            images = images.to(device)

            # TTA View 1: Original
            logits_1 = model(images, labels=None)

            # TTA View 2: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            logits_2 = model(images_flipped, labels=None)

            # Average the logits
            logits = (logits_1 + logits_2) / 2.0

            all_logits.append(logits.cpu())
            all_image_ids.extend(img_names)

    # Concatenate all logits into a single numpy array
    all_logits = torch.cat(all_logits, dim=0).numpy()

    return all_image_ids, all_logits
