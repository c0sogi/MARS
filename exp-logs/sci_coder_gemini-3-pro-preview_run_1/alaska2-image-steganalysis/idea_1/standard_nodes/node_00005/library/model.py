import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

from library.config import Config
from library.dataset import StegoDataset, get_transforms, load_metadata
from library.utils import seed_everything, weighted_auc_score

# ==========================================
# Model Architecture
# ==========================================


class HPF(nn.Module):
    """
    Fixed High-Pass Filter module using the KV kernel.
    Extracts noise residuals from RGB images to suppress semantic content.
    """

    def __init__(self):
        super(HPF, self).__init__()
        # KV filter kernel (5x5), normalized
        kv_kernel = (
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            )
            / 12.0
        )

        # Reshape to (out_channels, in_channels/groups, kH, kW)
        # We apply the same kernel to each of the 3 channels independently (groups=3).
        # Shape: (3, 1, 5, 5)
        weights = torch.tensor(kv_kernel).view(1, 1, 5, 5).repeat(3, 1, 1, 1)

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=5,
            padding=2,
            groups=3,
            bias=False,
        )
        self.conv.weight.data = weights
        self.conv.weight.requires_grad = False  # Fixed weights, non-trainable

    def forward(self, x):
        return self.conv(x)


class HPF_EfficientNet(nn.Module):
    """
    Single-Stream HPF-CNN using EfficientNet-B0 backbone.
    Pipeline: Input -> HPF (Residuals) -> EfficientNet Backbone -> Binary Classifier
    """

    def __init__(self, backbone_name=None, pretrained=None, num_classes=None):
        super(HPF_EfficientNet, self).__init__()

        # Use Config defaults if not provided to allow flexibility
        backbone_name = backbone_name or Config.backbone
        pretrained = pretrained if pretrained is not None else Config.pretrained
        num_classes = num_classes or Config.num_classes

        self.hpf = HPF()

        # Load backbone
        # in_chans=3 because HPF outputs 3 channels (residuals)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=3
        )

        # Replace the classifier head for binary classification
        # EfficientNet in timm uses 'classifier' as the final layer
        if hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "fc"):  # ResNet style fallback
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "head") and hasattr(
            self.backbone.head, "fc"
        ):  # Transformer fallback
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Extract noise residuals
        x = self.hpf(x)
        # Pass through backbone
        x = self.backbone(x)
        return x


# ==========================================
# Training & Evaluation Functions
# ==========================================


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Weighted AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Store predictions (sigmoid probability) and labels for AUC calculation
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    # Calculate Weighted AUC using the provided utility
    score = weighted_auc_score(all_labels, all_preds)

    return epoch_loss, score


def predict_and_submit(model_path, debug=False):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission...")
    Config.setup()
    device = Config.device

    # Load Test Metadata
    # Note: test.csv has labels=0 placeholder
    # If debug is True, we use a small subset to verify pipeline works
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

    # Load Model
    model = HPF_EfficientNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            predictions.extend(probs)

            # Get IDs for this batch from the dataframe
            start_idx = i * Config.batch_size
            end_idx = start_idx + len(images)
            batch_ids = df_test.iloc[start_idx:end_idx]["image_id"].values
            ids.extend(batch_ids)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Label": predictions})

    # Save to disk
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def run_task(debug=Config.debug):
    """
    Main entry point: Orchestrates training, validation, and submission generation.
    """
    # 1. Setup
    Config.setup()
    device = Config.device
    seed_everything(Config.seed)

    print(f"Starting task execution. Device: {device}, Debug: {debug}")

    # 2. Data Loading
    df_train = load_metadata(
        Config.train_csv, debug=debug, subset_size=Config.train_subset_size
    )
    df_val = load_metadata(
        Config.val_csv, debug=debug, subset_size=Config.val_subset_size
    )

    train_dataset = StegoDataset(df_train, transform=get_transforms(mode="train"))
    val_dataset = StegoDataset(df_val, transform=get_transforms(mode="val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HPF_EfficientNet().to(device)

    # 4. Optimizer & Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.max_lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 5. Training Loop
    best_score = -1.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # --- Validate ---
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Weighted AUC: {val_score}"
        )

        # --- Checkpoint & Early Stopping ---
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with score: {best_score}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # 6. Prediction
    if os.path.exists(best_model_path):
        predict_and_submit(best_model_path, debug=debug)
    else:
        print("Error: No model file generated.")
