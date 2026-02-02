import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import weighted_auc_score
from library.dataset import StegoDataset, get_transforms, load_metadata
from library.model import HPF_EfficientNet


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (stepped per batch).
        device: Torch device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # OneCycleLR steps per batch
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    return running_loss / dataset_size


def valid_one_epoch(model, loader, criterion, device):
    """
    Performs one epoch of validation.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (Average loss, Weighted AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Flatten arrays for metric calculation
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    score = weighted_auc_score(all_labels, all_preds)

    return epoch_loss, score


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
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
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Scheduler.
        device: Torch device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model weights.
    """
    best_score = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_score = valid_one_epoch(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Weighted AUC: {val_score}"
        )

        # Save best model based on Weighted AUC
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break


def predict(model_path, device, debug=False):
    """
    Generates predictions for the test set and saves them to the submission file.

    Args:
        model_path: Path to the saved model weights.
        device: Torch device.
        debug: If True, runs on a subset of data.
    """
    # Load Test Metadata
    df_test = load_metadata(
        Config.test_csv,
        debug=debug,
        subset_size=Config.val_subset_size if debug else None,
    )

    test_dataset = StegoDataset(df_test, transform=get_transforms(mode="test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model and load weights
    model = HPF_EfficientNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Cite solution_lesson_node_00005: Zero-Shot Performance Boosting via Test-Time Augmentation (TTA)
            # Average predictions of original and horizontally flipped images
            outputs = model(images)
            outputs_flip = model(torch.flip(images, dims=[3]))

            probs = torch.sigmoid(outputs)
            probs_flip = torch.sigmoid(outputs_flip)

            avg_probs = (probs + probs_flip) / 2.0
            predictions.extend(avg_probs.cpu().numpy().ravel())

            # Retrieve image IDs for the current batch
            start_idx = i * Config.batch_size
            end_idx = start_idx + len(images)
            batch_ids = df_test.iloc[start_idx:end_idx]["image_id"].values
            ids.extend(batch_ids)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Label": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
