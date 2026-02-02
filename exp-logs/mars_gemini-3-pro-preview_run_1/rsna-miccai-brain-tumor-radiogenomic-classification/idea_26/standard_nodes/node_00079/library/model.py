import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.utils import get_device, save_checkpoint


class SIA_DS_EfficientNet(nn.Module):
    """
    Scale-Invariant Anatomically-Anchored Dense-Slab (SIA-DS) Network.
    Uses EfficientNet-B0 backbone with a modified first layer to accept 9 channels.
    Channels are initialized using Gaussian Weight Inflation to preserve ImageNet priors
    while integrating volumetric context from relative anatomical depths (45%, 50%, 55%).
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        num_classes=1,
        drop_rate=0.3,
    ):
        super(SIA_DS_EfficientNet, self).__init__()

        # Load backbone with specified dropout rate
        # EfficientNet-B0 is used for parameter efficiency
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

        # Adapt the first convolutional layer for 9-channel input
        self._adapt_first_layer()

    def _adapt_first_layer(self):
        """
        Replaces the standard 3-channel input layer with a 9-channel layer.
        Initializes weights using Gaussian Weight Inflation:
        - Center channels (3-5, 50% depth): 50% of original energy.
        - Peripheral channels (0-2 & 6-8, 45% & 55% depth): 25% of original energy.
        """
        # In timm efficientnet, the first layer is typically named 'conv_stem'
        old_conv = self.backbone.conv_stem

        # Create new convolution layer
        new_conv = nn.Conv2d(
            in_channels=9,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # Weight Initialization
        with torch.no_grad():
            old_weights = old_conv.weight.data  # Shape: (Out, 3, K, K)
            new_weights = new_conv.weight.data  # Shape: (Out, 9, K, K)

            # Channels 0-2: Relative Depth 45% (Peripheral) -> 25% Energy
            new_weights[:, 0:3, :, :] = old_weights * 0.25

            # Channels 3-5: Relative Depth 50% (Center) -> 50% Energy
            new_weights[:, 3:6, :, :] = old_weights * 0.50

            # Channels 6-8: Relative Depth 55% (Peripheral) -> 25% Energy
            new_weights[:, 6:9, :, :] = old_weights * 0.25

            # Copy bias if it exists
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def run_training(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    patience=5,
    save_path="./working/best_model.pth",
):
    """
    Orchestrates the training process with Early Stopping.
    """
    device = get_device()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # AdamW with weight decay 1e-2 as specified
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f} - "
            f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                save_path,
            )
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_auc


def generate_submission(model, test_loader, output_path="./submission/submission.csv"):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    ids = []
    predictions = []

    print("Generating submission...")

    with torch.no_grad():
        for inputs, subject_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids.extend(subject_ids.numpy())
            predictions.extend(probs)

    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Format IDs as required (usually just integer in CSV, but checking sample format)
    # Sample submission shows integers.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
