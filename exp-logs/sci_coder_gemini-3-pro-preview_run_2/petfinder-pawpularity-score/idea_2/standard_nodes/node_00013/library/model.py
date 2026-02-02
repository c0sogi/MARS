import os
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.utils import seed_everything, get_rmse_score
from library.dataset import PawpularityDataset, get_transforms


class PawpularitySwinModel(nn.Module):
    """
    Neural network model for Pawpularity prediction.
    Uses a pre-trained Swin Transformer (Large) backbone and a custom MLP head
    that fuses image features with metadata.
    """

    def __init__(
        self, model_name="swin_large_patch4_window7_224.ms_in22k", pretrained=True
    ):
        super(PawpularitySwinModel, self).__init__()

        # Initialize Swin Transformer backbone
        # num_classes=0 returns the pooled feature vector (1536 dim for Swin Large)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.num_features = self.backbone.num_features
        self.meta_features = 12

        # Custom MLP head for fusion
        # Input: Swin features (1536) + Metadata (12)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features + self.meta_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, images, metadata):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
            metadata (torch.Tensor): Batch of metadata features (B, 12).

        Returns:
            torch.Tensor: Predicted Pawpularity scores (B, 1).
        """
        # Extract image features
        features = self.backbone(images)  # Shape: (B, 768)

        # Concatenate image features with metadata
        combined = torch.cat([features, metadata], dim=1)  # Shape: (B, 780)

        # Pass through MLP head
        output = self.mlp(combined)  # Shape: (B, 1)

        return output


def train_one_epoch(model, optimizer, scheduler, dataloader, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

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

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device, criterion):
    """
    Validates the model on the validation set.
    Returns average loss and RMSE.
    """
    model.eval()
    running_loss = 0.0
    preds = []
    actuals = []

    with torch.no_grad():
        for images, metadata, targets in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images, metadata)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            # Rescale outputs and targets back to [1, 100] for RMSE calculation
            # Dataset targets are scaled by / 100.0
            batch_preds = outputs.cpu().numpy().flatten() * 100.0
            batch_targets = targets.cpu().numpy().flatten() * 100.0

            preds.extend(batch_preds)
            actuals.extend(batch_targets)

    epoch_loss = running_loss / len(dataloader.dataset)
    rmse = get_rmse_score(preds, actuals)
    return epoch_loss, rmse


def run_training(
    train_csv_path="./metadata/train.csv",
    val_csv_path="./metadata/validation.csv",
    output_dir="./working/idea_2",
    epochs=10,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    learning_rate_backbone=1e-5,
    learning_rate_head=1e-4,
):
    """
    Orchestrates the training process.
    """
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(42)

    print(f"Device: {device}")

    # Load Dataframes
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

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

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, criterion
        )
        val_loss, val_rmse = validate(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val RMSE: {val_rmse}")

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with RMSE: {best_rmse}")

    print(f"Training complete. Best RMSE: {best_rmse}")
    return best_model_path


def generate_submission(
    model_path,
    test_csv_path="./metadata/test.csv",
    submission_path="./submission/submission.csv",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generates predictions for the test set and saves the submission file.
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
            preds = outputs.cpu().numpy().flatten() * 100.0

            ids.extend(batch_ids)
            predictions.extend(preds)

    # Clip predictions to valid range [1, 100]
    predictions = np.clip(predictions, 1.0, 100.0)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
