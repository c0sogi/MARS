import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.dataset import get_dataloaders


class HybridLinearProbe(nn.Module):
    """
    A Hybrid model that uses an EfficientNet-B0 backbone for visual features
    and concatenates them with metadata features.
    """

    def __init__(self, meta_dim):
        super(HybridLinearProbe, self).__init__()

        # Load pre-trained EfficientNet-B0
        # num_classes=0 returns the pooled feature vector
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        # EfficientNet-B0 output features
        # Dynamically determine visual_dim using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            dummy_output = self.backbone(dummy_input)
            self.visual_dim = dummy_output.shape[1]

        # Single Linear Layer Head
        self.head = nn.Linear(self.visual_dim + meta_dim, 1)

    def forward(self, images, meta):
        # Extract visual features
        # Shape: (Batch, visual_dim)
        x = self.backbone(images)

        # Concatenate with metadata
        x = torch.cat([x, meta], dim=1)

        # Linear Classification
        logits = self.head(x)
        return logits

    def set_backbone_trainable(self, trainable=True):
        for param in self.backbone.parameters():
            param.requires_grad = trainable


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
