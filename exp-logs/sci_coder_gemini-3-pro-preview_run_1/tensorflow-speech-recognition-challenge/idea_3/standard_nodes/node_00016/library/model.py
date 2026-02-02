import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import mixup_data, mixup_criterion, save_checkpoint, load_checkpoint


class EfficientNetAudio(nn.Module):
    """
    EfficientNet-B1 adapted for 1-channel audio spectrograms.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(EfficientNetAudio, self).__init__()

        # Load EfficientNet-B1 (Cite Lesson 13 for transfer learning base)
        # We use "DEFAULT" weights (ImageNet) for transfer learning
        weights = "DEFAULT" if pretrained else None
        self.model = models.efficientnet_b1(weights=weights)

        # 1. Adapt First Convolutional Layer
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # We need in_channels=1 to handle spectrogram inputs.
        original_conv = self.model.features[0][0]

        # Create new 1-channel layer
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize with average of pretrained weights (Cite Lesson 13 enhancement)
        if pretrained:
            with torch.no_grad():
                self.model.features[0][0].weight.data = original_conv.weight.data.mean(
                    dim=1, keepdim=True
                )

        # 2. Adapt Classifier Head
        # EfficientNet classifier is a Sequential block.
        # Index 1 is the Linear layer: Linear(in_features=1280, out_features=1000, bias=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, loader, criterion, optimizer, device, alpha):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    device=Config.DEVICE,
    patience=10,
):
    """
    Full training loop with Early Stopping and Scheduler.
    """
    model = model.to(device)

    # Optimizer & Scheduler as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.MIXUP_ALPHA
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}: LR={current_lr}, Train Loss={train_loss}, Val Loss={val_loss}, Val Acc={val_acc}"
        )

        # Checkpoint & Early Stopping
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best=True,
                checkpoint_dir=Config.WORKING_DIR,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. Best Acc: {best_acc} at epoch {best_epoch}"
            )
            break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")
    return best_acc


def predict_and_submit(
    model, test_loader, device=Config.DEVICE, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    # Load best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for inference...")
        load_checkpoint(best_model_path, model, device=device)
    else:
        print("Warning: Best model not found. Using current model weights.")

    model.eval()
    model = model.to(device)

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())

    # Reconstruct filenames from metadata
    # We rely on the fact that test_loader iterates sequentially over the dataset
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Handle potential size mismatch if debug mode was used
    if len(predictions) != len(df_test):
        if len(predictions) < len(df_test):
            df_test = df_test.iloc[: len(predictions)]

    # Map IDs to Labels
    predicted_labels = [Config.ID2LABEL[p] for p in predictions]

    # Extract filename from filepath (e.g., test/audio/clip_000.wav -> clip_000.wav)
    fnames = df_test["filepath"].apply(os.path.basename).tolist()

    submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
