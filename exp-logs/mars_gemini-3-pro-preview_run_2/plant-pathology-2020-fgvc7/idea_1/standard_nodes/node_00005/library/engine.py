import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import calculate_class_weights


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use for evaluation.

    Returns:
        tuple: (average_loss, mean_column_wise_roc_auc)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)
            dataset_size += batch_size

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert labels to one-hot encoding for ROC AUC calculation
    num_classes = Config.NUM_CLASSES
    one_hot_labels = np.zeros((all_labels.size, num_classes))
    one_hot_labels[np.arange(all_labels.size), all_labels] = 1

    # Calculate Mean Column-wise ROC AUC
    # multi_class='ovr' handles the one-vs-rest calculation for each class
    # average='macro' computes the metric for each class and takes the unweighted mean
    try:
        roc_auc = roc_auc_score(
            one_hot_labels, all_preds, average="macro", multi_class="ovr"
        )
    except ValueError:
        # Handle edge cases where a class might be missing in the batch/set
        roc_auc = 0.0

    return epoch_loss, roc_auc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler (optional).
        device (torch.device): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.

    Returns:
        float: The best validation ROC AUC score achieved.
    """
    # Load training metadata to calculate class weights for imbalance handling
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    class_weights = calculate_class_weights(train_df, device=device)

    # Initialize weighted CrossEntropyLoss with Label Smoothing
    # Cite solution_lesson_node_00001: Using class weights to handle imbalance.
    # Label smoothing helps prevent overfitting and improves calibration.
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_score = -1.0
    patience_counter = 0

    # Ensure directory exists for saving model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val ROC AUC: {val_score}"
        )

        # Check for improvement (Maximize ROC AUC)
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_score


def predict(model, test_loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).

    Args:
        model (torch.nn.Module): The model to use for prediction.
        test_loader (torch.utils.data.DataLoader): Test dataloader.
        device (torch.device): Device.

    Returns:
        tuple: (image_ids, probabilities)
    """
    model.eval()
    all_probs = []
    all_image_ids = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # 1. Standard Prediction
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # 2. TTA: Horizontal Flip
            # Flip along width dimension (dim 3: N, C, H, W)
            images_flip = torch.flip(images, [3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # Average probabilities
            avg_probs = (probs + probs_flip) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_image_ids.extend(image_ids)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_image_ids, all_probs


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions and saves them to a CSV file in the required format.

    Args:
        model (torch.nn.Module): The model to use.
        test_loader (torch.utils.data.DataLoader): Test dataloader.
        device (torch.device): Device.
        output_path (str): Path to save the submission CSV.
    """
    image_ids, probs = predict(model, test_loader, device)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=Config.TARGET_COLS)
    df.insert(0, "image_id", image_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
