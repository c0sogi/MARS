import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np
import pandas as pd
import os
import sys

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_CHECKPOINT_PATH,
    SUBMISSION_PATH,
    TTA_STEPS,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything, calculate_metric
from library.data_loader import (
    get_dataloaders,
    preprocess_metadata,
    MelanomaDataset,
    get_transforms,
)
from torch.utils.data import DataLoader


class EfficientNetFusion(nn.Module):
    def __init__(self, num_tabular_features, num_classes=1):
        super(EfficientNetFusion, self).__init__()

        # 1. Image Backbone (EfficientNet-B3)
        # Load pre-trained weights
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b3(weights=weights)

        # EfficientNet-B3 output before classifier is 1536
        self.img_feature_dim = 1536

        # Remove the original classifier to use our own fusion head
        # We will use backbone.features and backbone.avgpool explicitly in forward
        del self.backbone.classifier

        # 2. Tabular Branch (MLP)
        self.tab_feature_dim = 64
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.tab_feature_dim),
            nn.ReLU(),
        )

        # 3. Fusion Head
        # Concatenate Image (1280) + Tabular (64)
        fusion_dim = self.img_feature_dim + self.tab_feature_dim
        self.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(fusion_dim, num_classes))

    def forward(self, images, tabular_data):
        # Image Branch
        x = self.backbone.features(images)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # (Batch, 1280)

        # Tabular Branch
        tab = self.tabular_mlp(tabular_data)  # (Batch, 64)

        # Fusion
        combined = torch.cat([x, tab], dim=1)  # (Batch, 1344)

        # Classification
        logits = self.head(combined)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, tabular, targets in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images, tabular)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, tabular)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    auc = calculate_metric(np.array(all_targets), np.array(all_preds))

    return avg_loss, auc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, tabular, _ in loader:
            images = images.to(device)
            tabular = tabular.to(device)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().flatten())

    return np.array(all_preds)


def main():
    seed_everything(SEED)

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine tabular input dimension from dataset
    # train_loader.dataset is MelanomaDataset
    num_tabular_features = train_loader.dataset.tabular_data.shape[1]
    print(f"Detected {num_tabular_features} tabular features.")

    # Model Setup
    print("Initializing Model...")
    model = EfficientNetFusion(num_tabular_features=num_tabular_features).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training Loop
    best_auc = 0.0
    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        print(f"Validation AUC: {val_auc}")

        if val_auc > best_auc:
            best_auc = val_auc
            print(f"AUC improved. Saving model to {MODEL_CHECKPOINT_PATH}")
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)

        print("-" * 30)

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Inference and Submission
    print("Starting Inference...")

    # Load Best Model
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 1. Standard Prediction (Original Test Images)
    print("Predicting on standard test set...")
    final_preds = predict(model, test_loader, DEVICE)

    # 2. Test-Time Augmentation (TTA)
    if TTA_STEPS > 0:
        print(f"Performing TTA with {TTA_STEPS} steps...")

        # Need to reconstruct test dataset with TRAIN transforms (augmentations)
        # Load metadata/tabular data again
        _, _, test_df, _, _, test_tab = preprocess_metadata(load_cached_data=True)

        # Use 'train' transforms which include flips/rotations
        tta_dataset = MelanomaDataset(
            test_df, test_tab, transform=get_transforms("train"), is_test=True
        )

        tta_loader = DataLoader(
            tta_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        tta_accum = np.zeros_like(final_preds)

        for i in range(TTA_STEPS):
            print(f"TTA Step {i+1}/{TTA_STEPS}")
            preds = predict(model, tta_loader, DEVICE)
            tta_accum += preds

        # Average: (Standard + Sum(TTA)) / (1 + TTA_STEPS)
        final_preds = (final_preds + tta_accum) / (1 + TTA_STEPS)

    # Save Submission
    print("Generating submission file...")
    test_df = pd.read_csv(os.path.join("./metadata", "test.csv"))
    submission = pd.DataFrame(
        {"image_name": test_df["image_name"], "target": final_preds}
    )

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


# Execute main function
if __name__ == "__main__":
    main()
