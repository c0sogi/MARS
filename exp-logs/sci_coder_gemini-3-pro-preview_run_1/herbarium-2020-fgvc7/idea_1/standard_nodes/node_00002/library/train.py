import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library import config
from library import utils
from library import dataset
from library import model


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training dataloader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run training on.
        scheduler (lr_scheduler): Learning rate scheduler (optional).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = utils.AverageMeter()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        device (str): Device to run validation on.

    Returns:
        tuple: (Average validation loss, Macro F1 score)
    """
    model.eval()
    losses = utils.AverageMeter()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate Metric
    f1 = utils.calculate_metrics(all_labels, all_preds)

    return losses.avg, f1


def inference(model, dataloader, device, id2label):
    """
    Runs inference on the test set.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test dataloader.
        device (str): Device to run inference on.
        id2label (dict): Mapping from label index to category_id.

    Returns:
        pd.DataFrame: DataFrame with 'Id' and 'Predicted' columns.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Map predictions back to category_id
            preds_np = preds.cpu().numpy()
            mapped_preds = [id2label[p] for p in preds_np]

            ids_list.extend(image_ids.numpy())
            preds_list.extend(mapped_preds)

    return pd.DataFrame({"Id": ids_list, "Predicted": preds_list})


def run_training(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    lr=config.LEARNING_RATE,
    debug=config.DEBUG,
):
    """
    Main driver function to run the training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for dataloaders.
        lr (float): Learning rate.
        debug (bool): Whether to run in debug mode (subsampled data).
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"Starting training on device: {device}")
    print(f"Debug mode: {debug}")

    # 2. Data Preparation
    # Get label mapping to ensure consistency
    label2id, id2label = dataset.get_label_mapping(load_cached_data=True)

    train_dataset = dataset.PlantDataset(
        split="train", transform=dataset.get_transforms("train"), debug=debug
    )
    val_dataset = dataset.PlantDataset(
        split="val", transform=dataset.get_transforms("val"), debug=debug
    )
    test_dataset = dataset.PlantDataset(
        split="test", transform=dataset.get_transforms("test"), debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    net = model.PlantClassifier(pretrained=True)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_f1 = -1.0
    patience = 3
    patience_counter = 0
    best_model_path = config.CACHE_DIR / "best_model.pth"

    # Ensure cache dir exists
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss, val_f1 = validate(net, val_loader, criterion, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val F1: {val_f1}")

        # Checkpoint & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 5. Inference
    print("\nLoading best model for inference...")
    if best_model_path.exists():
        net.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found, using current weights.")

    print("Generating predictions on test set...")
    submission_df = inference(net, test_loader, device, id2label)

    # 6. Save Submission
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
