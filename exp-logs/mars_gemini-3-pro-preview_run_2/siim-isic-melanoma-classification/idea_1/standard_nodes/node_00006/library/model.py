import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.dataset import get_dataloaders


class HybridLinearProbe(nn.Module):
    """
    A Hybrid Linear Probe model that uses a frozen MobileNetV3-Small backbone
    for visual features and concatenates them with metadata features before
    passing through a linear classification head.
    """

    def __init__(self, meta_dim):
        super(HybridLinearProbe, self).__init__()

        # Load pre-trained MobileNetV3-Small
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = torchvision.models.mobilenet_v3_small(weights=weights)

        # Extract feature extractor layers
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool

        # Unfreeze the backbone
        # Cite solution_lesson_node_00004: Unfreezing Backbones for Domain-Shifted Transfer Learning
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.avgpool.parameters():
            param.requires_grad = True

        # MobileNetV3-Small output channels before classifier is 576
        self.visual_dim = 576

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

        # Single Linear Layer Head
        # Input: Visual Features (576) + Metadata Features (meta_dim)
        # Output: 1 (Logit)
        self.head = nn.Linear(self.visual_dim + meta_dim, 1)

    def forward(self, images, meta):
        # Extract visual features
        # Shape: (Batch, 576, H, W)
        x = self.features(images)
        # Global Average Pooling -> Shape: (Batch, 576, 1, 1)
        x = self.avgpool(x)
        # Flatten -> Shape: (Batch, 576)
        x = torch.flatten(x, 1)

        # Concatenate with metadata
        # meta shape: (Batch, meta_dim)
        x = torch.cat([x, meta], dim=1)

        # Apply Dropout
        x = self.dropout(x)

        # Linear Classification
        logits = self.head(x)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, meta, targets in loader:
        images = images.to(device)
        meta = meta.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        logits = model(images, meta)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, meta, targets in loader:
            images = images.to(device)
            meta = meta.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, meta)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where only one class is present in batch/set
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def generate_submission(model, loader, device, output_path):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, meta, _ in loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten().tolist())

    # Create submission DataFrame
    # We access the dataframe from the dataset to get image names
    image_names = loader.dataset.df["image_name"].values

    submission_df = pd.DataFrame({"image_name": image_names, "target": all_preds})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine metadata dimension from a sample batch
    sample_img, sample_meta, _ = next(iter(train_loader))
    meta_dim = sample_meta.shape[1]
    print(f"Metadata Dimension: {meta_dim}")

    print("Initializing Model...")
    model = HybridLinearProbe(meta_dim=meta_dim)
    model = model.to(device)

    # Loss Function with Class Imbalance handling
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training Complete. Best Val AUC: {best_auc}")

    # Load best model for inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating Submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run_training()
