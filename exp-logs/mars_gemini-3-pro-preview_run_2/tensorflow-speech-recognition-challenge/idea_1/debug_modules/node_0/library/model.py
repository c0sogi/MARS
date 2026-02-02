import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.dataset import get_dataloaders, get_test_loader


class SimpleConvNet(nn.Module):
    """
    A simple 4-block Convolutional Neural Network for Audio Classification.
    Input: (Batch, 1, Freq, Time) -> Log Mel-Spectrograms
    Output: (Batch, Num_Classes) -> Logits
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(SimpleConvNet, self).__init__()

        # Block 1: 1 -> 16 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 16 -> 32 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 32 -> 64 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 64 -> 128 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (Batch, 128)
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(debug=Config.DEBUG, epochs=Config.NUM_EPOCHS):
    """
    Main training loop with early stopping.
    """
    # 1. Reproducibility
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # 3. Model setup
    model = SimpleConvNet(num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 4. Training Loop
    best_acc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_CHECKPOINT_PATH

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Acc: {val_acc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training completed. Best Val Acc: {best_acc}")
    return best_acc


def generate_submission():
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = torch.device(Config.DEVICE)
    model = SimpleConvNet(num_classes=Config.NUM_CLASSES)

    # Load best model
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Train the model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = get_test_loader()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Load test metadata to get filenames (order is preserved by test_loader)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    fnames = df_test["fname"].tolist()

    if len(fnames) != len(predictions):
        print(
            f"Warning: Mismatch between number of files ({len(fnames)}) and predictions ({len(predictions)})"
        )

    # Map indices to labels
    predicted_labels = [Config.IDX2LABEL[int(idx)] for idx in predictions]

    # Create submission dataframe
    df_sub = pd.DataFrame({"fname": fnames, "label": predicted_labels})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
