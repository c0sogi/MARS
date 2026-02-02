import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.utils import seed_everything, get_rmse_score
from library.dataset import PawpularityDataset, get_transforms
from library.model import PawpularitySwinModel


def train_one_epoch(model, optimizer, scheduler, dataloader, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for images, metadata, targets in dataloader:
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images, metadata)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device, criterion):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    preds = []
    actuals = []
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for images, metadata, targets in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images, metadata)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            # Rescale outputs and targets back to [1, 100] for RMSE calculation
            # Model outputs are approximately in [0, 1] range due to training targets
            batch_preds = torch.sigmoid(outputs).cpu().numpy().flatten() * 100.0
            batch_targets = targets.cpu().numpy().flatten() * 100.0

            preds.extend(batch_preds)
            actuals.extend(batch_targets)

    epoch_loss = running_loss / dataset_size
    rmse = get_rmse_score(preds, actuals)
    return epoch_loss, rmse


def run_training(
    train_csv_path="./metadata/train.csv",
    val_csv_path="./metadata/validation.csv",
    output_dir="./working/idea_2",
    epochs=10,
    batch_size=32,
    learning_rate_backbone=1e-5,
    learning_rate_head=1e-4,
    patience=3,
    debug=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Orchestrates the training process with Early Stopping.
    """
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(42)

    print(f"Using device: {device}")

    # Load Dataframes
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if debug:
        print("Debug mode enabled: Training on a small subset.")
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Create Datasets
    train_dataset = PawpularityDataset(train_df, transforms=get_transforms("train"))
    val_dataset = PawpularityDataset(val_df, transforms=get_transforms("valid"))

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    model = PawpularitySwinModel()
    model.to(device)

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": learning_rate_backbone},
            {"params": model.mlp.parameters(), "lr": learning_rate_head},
        ]
    )

    # Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function (BCEWithLogitsLoss)
    # Cite solution_lesson_node_00010: Optimizing Bounded Regression with BCE Loss and Sigmoid Activation
    criterion = nn.BCEWithLogitsLoss()

    best_rmse = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, criterion
        )
        val_loss, val_rmse = valid_one_epoch(model, val_loader, device, criterion)

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val RMSE: {val_rmse}")

        # Save best model and check early stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with RMSE: {best_rmse}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best RMSE: {best_rmse}")
    return best_model_path


def predict(
    model_path,
    test_csv_path="./metadata/test.csv",
    submission_path="./submission/submission.csv",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generates predictions for the test set using the trained model.
    """
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Load Test Data
    test_df = pd.read_csv(test_csv_path)
    test_dataset = PawpularityDataset(
        test_df, transforms=get_transforms("valid"), test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Model
    model = PawpularitySwinModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ids = []
    predictions = []

    print("Generating predictions...")

    with torch.no_grad():
        for images, metadata, batch_ids in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)

            # Rescale predictions: model outputs [0, 1] -> [0, 100]
            preds = torch.sigmoid(outputs).cpu().numpy().flatten() * 100.0

            ids.extend(batch_ids)
            predictions.extend(preds)

    # Clip predictions to valid range [1, 100]
    predictions = np.clip(predictions, 1.0, 100.0)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
