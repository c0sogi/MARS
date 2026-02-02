import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from library.config import Config
from library.dataset import get_dataloaders

# Suppress warnings as requested
warnings.filterwarnings("ignore")


class FrozenResNetLinear(nn.Module):
    """
    A hybrid model using a frozen ResNet-18 backbone for image features
    concatenated with metadata features, fed into a single linear layer.
    """

    def __init__(self):
        super(FrozenResNetLinear, self).__init__()

        # Load pre-trained ResNet-18
        # Using weights=None and loading state_dict if needed, or using 'DEFAULT'
        # Since internet access might be restricted or versions vary, we use the standard constructor
        # and handle the weights carefully. The prompt implies standard torchvision.
        # We use the most stable way to get pretrained weights:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        except:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=True)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the final FC layer with Identity to get the 512 feature vector
        self.backbone.fc = nn.Identity()

        # Define the head: 512 (Image) + 12 (Metadata) -> 1 (Pawpularity)
        input_dim = 512 + len(Config.METADATA_COLS)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, images, metadata):
        # Extract image features
        img_features = self.backbone(images)

        # Concatenate with metadata
        # Ensure metadata is on the same device
        combined_features = torch.cat((img_features, metadata), dim=1)

        # Predict
        output = self.head(combined_features)
        return output


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, metadata, targets in dataloader:
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device).unsqueeze(1)  # Match shape (Batch, 1)

        optimizer.zero_grad()

        outputs = model(images, metadata)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    return running_loss / num_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, metadata, targets in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, metadata)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    epoch_loss = running_loss / num_samples

    # Calculate RMSE
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)

    return epoch_loss, rmse


def train_model(debug=False):
    """
    Main training loop with Early Stopping.
    """
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Get Dataloaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Initialize Model
    model = FrozenResNetLinear().to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    # Optimize only the head
    optimizer = optim.AdamW(model.head.parameters(), lr=Config.LEARNING_RATE)

    # Training Configuration
    num_epochs = Config.NUM_EPOCHS
    best_rmse = float("inf")
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val RMSE: {val_rmse} - "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping and Model Saving
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  [New Best Model Saved] RMSE: {best_rmse}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val RMSE: {best_rmse}")
    return best_rmse


def predict_and_submit(debug=False):
    """
    Loads the best model, generates predictions on the test set, and saves to CSV.
    """
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(debug=debug)

    # Load Model
    model = FrozenResNetLinear().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    predictions = []
    ids = []

    # The test loader dataset df has the Ids.
    # We can iterate the loader for predictions and use the dataset for Ids,
    # provided shuffle=False (which it is in get_dataloaders).

    print("Generating predictions...")
    with torch.no_grad():
        for images, metadata, _ in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)
            preds = outputs.cpu().numpy().flatten()
            predictions.extend(preds)

    # Retrieve Ids from the dataset dataframe
    # We must ensure we take the same slice if in debug mode
    test_df = test_loader.dataset.df
    ids = test_df["Id"].values

    # Clip predictions to valid range [1, 100] (optional but recommended for this metric)
    predictions = np.clip(predictions, 1.0, 100.0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())


def run_pipeline(debug=False):
    """
    Helper to run training then submission.
    """
    train_model(debug=debug)
    predict_and_submit(debug=debug)
