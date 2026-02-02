import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import get_dataloaders


from torchvision.models import resnet18


class PathologyResNet(nn.Module):
    """
    ResNet18 adapted for small input size (48x48).
    """

    def __init__(self):
        super(PathologyResNet, self).__init__()
        # Load resnet18 without pretrained weights
        self.model = resnet18(weights=None)

        # Modify first layer for 48x48 input to preserve spatial dimensions
        # Replace 7x7 stride 2 with 3x3 stride 1
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove maxpool to prevent early downsampling
        self.model.maxpool = nn.Identity()

        # Modify FC layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for AUC
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict_and_submit(model, loader, device, output_path):
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            outputs = model(images)
            # Apply sigmoid to get probability
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids_list.extend(ids)
            preds_list.extend(probs)

    df = pd.DataFrame({"id": ids_list, "label": preds_list})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model(train_loader, val_loader, test_loader):
    Config.setup_directories()
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = PathologyResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision as requested
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved.")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for inference
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate submission
    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)


def run():
    """
    Main entry point to run the pipeline.
    """
    Config.set_seed(Config.SEED)
    train_loader, val_loader, test_loader = get_dataloaders()
    train_model(train_loader, val_loader, test_loader)
