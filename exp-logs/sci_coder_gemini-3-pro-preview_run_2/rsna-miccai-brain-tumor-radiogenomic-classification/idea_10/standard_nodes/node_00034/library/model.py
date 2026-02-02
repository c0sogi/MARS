import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import set_seed, get_device, save_checkpoint
from library.data import get_dataloaders

# ------------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------------


class AsymmetricEfficientNet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.2):
        super(AsymmetricEfficientNet, self).__init__()

        # Load pre-trained EfficientNet-B0
        # Weights='DEFAULT' loads the best available ImageNet weights
        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        # ----------------------------------------------------------------------
        # 1. Grouped Convolutional Stem & 2. Asymmetric Filter Initialization
        # ----------------------------------------------------------------------
        # Original stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # New stem: Conv2d(12, 32, ..., groups=4, ...)
        #
        # Logic:
        # Input is (N, 12, H, W). Groups=4 splits input into 4 chunks of 3 channels.
        # Output is 32 channels. Groups=4 splits output into 4 chunks of 8 channels.
        # Weights shape for both is (Out, In/Groups, K, K) = (32, 3, 3, 3).
        # We can directly copy the pretrained weights. This assigns filters 0-7 to
        # Modality 1, 8-15 to Modality 2, etc.

        old_stem = self.backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=4,
            bias=False,
        )

        # Copy weights (Asymmetric Initialization)
        new_stem.weight.data = old_stem.weight.data.clone()

        # Replace the stem in the backbone
        self.backbone.features[0][0] = new_stem

        # ----------------------------------------------------------------------
        # 3. Regularized Head
        # ----------------------------------------------------------------------
        # EfficientNet B0 classifier input features is 1280
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ------------------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for metrics
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy().flatten())
        all_preds.extend(probs.flatten())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where batch size might be 1 or all same class in a batch
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
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
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy().flatten())
            all_preds.extend(probs.flatten())

    epoch_loss = running_loss / len(loader.dataset)

    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

    return epoch_loss, epoch_auc


# ------------------------------------------------------------------------------
# Inference Logic
# ------------------------------------------------------------------------------


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    Returns list of averaged probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            # 1. Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (last dim)
            inputs_h = torch.flip(inputs, dims=[-1])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (second to last dim)
            inputs_v = torch.flip(inputs, dims=[-2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            all_preds.extend(avg_prob.cpu().numpy().flatten())

    return all_preds


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    print("Generating submission with TTA...")
    preds = predict_with_tta(model, test_loader, device)

    # Load test metadata to get IDs in correct order
    # The DataLoader iterates sequentially over the test dataframe used in library.data
    test_df = pd.read_csv("./metadata/test.csv")

    # Ensure lengths match
    if len(preds) != len(test_df):
        print(f"Warning: Prediction count {len(preds)} != Test ID count {len(test_df)}")
        # Handle debug/subset case by slicing the metadata
        test_df = test_df.iloc[: len(preds)]

    submission_df = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# ------------------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------------------


def run_training(
    epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-2,
    patience=5,
    debug_limit=None,
):
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=True, debug_limit=debug_limit
    )

    # 2. Model Initialization
    print("Initializing AsymmetricEfficientNet...")
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.5)
    model = model.to(device)

    # 3. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = "./working/idea_10_float32/best_model.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        scheduler.step(val_auc)

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"  [New Best Model] Saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] No improvement for {patience} epochs.")
                break

    # 5. Final Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device)
