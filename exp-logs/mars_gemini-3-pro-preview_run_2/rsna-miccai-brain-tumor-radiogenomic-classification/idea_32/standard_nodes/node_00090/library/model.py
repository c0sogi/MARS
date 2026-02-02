import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torchvision import models
from sklearn.metrics import roc_auc_score
from library import config, utils, data_loader

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------


class AsymmetricEfficientNet(nn.Module):
    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # Load pre-trained EfficientNet-B0
        # We use the default weights (IMAGENET1K_V1)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # ---------------------------------------------------------------------
        # 1. Stem Modification (Grouped Convolution)
        # ---------------------------------------------------------------------
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = self.backbone.features[0][0]

        # New: Conv2d(12, 32, ..., groups=4)
        # Groups=4 ensures modality isolation (FLAIR, T1w, T1wCE, T2w processed separately)
        new_conv = nn.Conv2d(
            in_channels=config.IN_CHANNELS,  # 12
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            groups=config.NUM_MODALITIES,  # 4
        )

        # ---------------------------------------------------------------------
        # 2. Asymmetric Filter Initialization
        # ---------------------------------------------------------------------
        # Original weights shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K) for groups=1
        # New weights shape:      (32, 3, 3, 3) -> (Out, In/Groups, K, K) for groups=4
        # We copy the weights directly. This distributes the 32 pre-trained filters
        # across the 4 modality groups (8 filters per modality).
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # ---------------------------------------------------------------------
        # 3. Classifier Modification
        # ---------------------------------------------------------------------
        # Get the number of input features to the final layer (1280 for B0)
        # The classifier in torchvision EfficientNet is a Sequential block.
        # We replace it to ensure we have specific Dropout and Output size.
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = utils.AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = utils.AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5  # Handle edge cases with single class in batch

    return losses.avg, auc


# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------


def predict_and_submit(model, test_loader, device, output_path):
    model.eval()
    results = []

    print("Generating predictions with TTA...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # 1. Original
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is width)
            inputs_h = torch.flip(inputs, dims=[3])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is height)
            inputs_v = torch.flip(inputs, dims=[2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # Average Probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            results.extend(avg_probs.cpu().numpy().flatten())

    # Create submission DataFrame
    # We need to map predictions back to BraTS21IDs.
    # The test_loader preserves order from the metadata dataframe.
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    submission = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": results}
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def run():
    # Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=True
    )

    # Model
    print("Initializing AsymmetricEfficientNet...")
    model = AsymmetricEfficientNet().to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience_counter = 0
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                checkpoint_dir=config.CACHE_DIR,
            )
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation AUC: {best_auc}")

    # Inference
    print("Loading best model for inference...")
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    checkpoint = utils.load_checkpoint(model, path=best_model_path, device=device)

    predict_and_submit(model, test_loader, device, config.SUBMISSION_PATH)
