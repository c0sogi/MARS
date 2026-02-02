import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library import config, data


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Asymmetric Grouped Convolutional Stem.

    Attributes:
        backbone (nn.Module): Modified EfficientNet-B0.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained EfficientNet-B0
        # Using 'IMAGENET1K_V1' weights as standard for transfer learning
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # 2. Modify the First Convolutional Layer (Stem)
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = self.backbone.features[0][0]

        # New: Conv2d(12, 32, ... groups=4)
        # 12 input channels (4 modalities * 3 slices)
        # groups=4 ensures strict modality isolation (3 channels -> 8 filters per group)
        new_conv = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
            groups=4,
        )

        # 3. Asymmetric Filter Initialization
        # The shape of original weights is (32, 3, 3, 3).
        # The shape of new weights with groups=4 is also (32, 3, 3, 3)
        # (calculated as out_channels, in_channels // groups, k, k).
        # We directly copy the weights. This maps:
        # Filters 0-7 (trained on RGB) -> Group 0 (FLAIR)
        # Filters 8-15 (trained on RGB) -> Group 1 (T1w)
        # ... and so on.
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.clone()
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 4. Reconstruct Classifier Head
        # EfficientNet-B0 final feature map depth is 1280.
        # We use Dropout -> Linear(1) for binary classification.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=1, bias=True),
        )

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
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

        # Store for AUC calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())

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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict_and_submit(
    model, loader, device, output_path="./submission/submission.csv"
):
    """
    Runs inference with Test-Time Augmentation (TTA) and saves submission file.
    TTA: Original + Horizontal Flip + Vertical Flip.
    """
    model.eval()
    predictions = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            # 1. Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            inputs_h = torch.flip(inputs, [3])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2 is height)
            inputs_v = torch.flip(inputs, [2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            prob_avg = (prob_orig + prob_h + prob_v) / 3.0
            predictions.extend(prob_avg.cpu().numpy().flatten())

    # Load test metadata to get BraTS21IDs (order is preserved by loader)
    df_test = pd.read_csv(config.TEST_METADATA)

    if len(predictions) != len(df_test):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of test samples ({len(df_test)})"
        )

    submission_df = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_pipeline():
    """
    Main execution pipeline:
    1. Setup
    2. Data Loading
    3. Training with Early Stopping
    4. Inference with Best Model
    """
    # Setup
    config.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_data_loaders(load_cached_data=True)

    # Model Initialization
    model = AsymmetricEfficientNet().to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Training Loop
    best_val_auc = 0.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (AUC: {best_val_auc:.6f})")

    print(f"Training complete. Best Val AUC: {best_val_auc:.6f}")

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    predict_and_submit(
        model, test_loader, device, output_path="./submission/submission.csv"
    )
