import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from library import config
from library import dataset


class DeepSupervisedResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load hierarchy mappings to determine output dimensions
        mappings = config.get_hierarchy_mappings()
        self.num_l1 = mappings["num_classes_l1"]
        self.num_l2 = mappings["num_classes_l2"]
        self.num_l3 = mappings["num_classes_l3"]

        # Initialize Backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Deconstruct ResNet50
        # Stem
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        # Stages
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Spatial Pooling (Global Average Pooling for spatial dims)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Heads
        # Stage 3 output channels: 1024
        self.head_coarse = nn.Linear(1024, self.num_l1)

        # Stage 4 output channels: 2048
        self.head_mid = nn.Linear(2048, self.num_l2)
        self.head_fine = nn.Linear(2048, self.num_l3)

    def _aggregate_features(self, x, batch_size, num_imgs):
        """
        Aggregates features from multiple images of the same product.
        Input x: (B*N, C, H, W)
        Output: (B, C)
        Strategy: Spatial Avg Pool -> Flatten -> Reshape -> Multi-view Max Pool
        """
        # Spatial Pooling: (B*N, C, H, W) -> (B*N, C, 1, 1)
        x = self.spatial_pool(x)

        # Flatten: (B*N, C)
        x = x.flatten(1)

        # Reshape to separate batch and views: (B, N, C)
        x = x.view(batch_size, num_imgs, -1)

        # Global Max Pooling across views: (B, C)
        x, _ = torch.max(x, dim=1)

        return x

    def forward(self, x):
        # Input x: (B, N, 3, H, W)
        b, n, c, h, w = x.shape

        # Merge batch and views for backbone processing
        x = x.view(b * n, c, h, w)

        # Pass through Stem and early layers
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # --- Stage 3 (Coarse Level) ---
        x = self.layer3(x)

        # Branch for Coarse Head
        # We clone or just use the tensor. Since _aggregate doesn't modify in-place, it's safe.
        feat_coarse = self._aggregate_features(x, b, n)
        logits_coarse = self.head_coarse(feat_coarse)

        # --- Stage 4 (Mid & Fine Levels) ---
        x = self.layer4(x)

        # Branch for Mid and Fine Heads
        feat_final = self._aggregate_features(x, b, n)
        logits_mid = self.head_mid(feat_final)
        logits_fine = self.head_fine(feat_final)

        return {"coarse": logits_coarse, "mid": logits_mid, "fine": logits_fine}


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()

    # Loss Functions
    # Label Smoothing only for the fine-grained target which is noisy/hard
    crit_fine = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    crit_mid = nn.CrossEntropyLoss()
    crit_coarse = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_fine = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Train", leave=False)

    for images, (l1, l2, l3) in pbar:
        images = images.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss_fine = crit_fine(outputs["fine"], l3)
        loss_mid = crit_mid(outputs["mid"], l2)
        loss_coarse = crit_coarse(outputs["coarse"], l1)

        # Weighted Sum
        loss = (
            loss_fine * config.LOSS_WEIGHTS["fine"]
            + loss_mid * config.LOSS_WEIGHTS["mid"]
            + loss_coarse * config.LOSS_WEIGHTS["coarse"]
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

        # Calculate Fine-grained Accuracy
        _, preds = torch.max(outputs["fine"], 1)
        correct_fine += (preds == l3).sum().item()
        total += images.size(0)

        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / total
    epoch_acc = correct_fine / total

    return epoch_loss, epoch_acc


def validate(model, loader, device):
    model.eval()

    crit_fine = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_fine = 0
    total = 0

    with torch.no_grad():
        for images, (l1, l2, l3) in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            l3 = l3.to(device)

            outputs = model(images)

            # We only track fine-grained loss/acc for validation selection
            loss = crit_fine(outputs["fine"], l3)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs["fine"], 1)
            correct_fine += (preds == l3).sum().item()
            total += images.size(0)

    val_loss = running_loss / total
    val_acc = correct_fine / total

    return val_loss, val_acc


def run_training(debug_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Initializing DataLoaders...")
    train_loader, val_loader = dataset.get_dataloaders(debug_size=debug_size)

    # Model
    print("Initializing Model...")
    model = DeepSupervisedResNet50(pretrained=True)
    model = model.to(device)

    # Optimization
    # Scale LR by batch size (Linear Scaling Rule approximation)
    # Base 0.01 is good for BS=256, we have BS=512, so maybe 0.02, but OneCycle handles peak.
    # We stick to config.LEARNING_RATE as max_lr
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Warmup for first 10%
    )

    best_acc = -1.0
    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New Best Accuracy! Saving model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Training Complete. Best Validation Accuracy: {best_acc}")
    return checkpoint_path


def generate_submission(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = DeepSupervisedResNet50(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load Test Data
    print("Initializing Test Loader...")
    test_loader = dataset.get_test_loader()

    # Mappings to convert class_idx back to category_id
    mappings = config.get_hierarchy_mappings()
    idx_to_cat = mappings["idx_to_cat"]

    predictions = []

    print("Running Inference...")
    with torch.no_grad():
        for images, sample_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)

            outputs = model(images)
            logits = outputs["fine"]

            _, preds = torch.max(logits, 1)

            preds_cpu = preds.cpu().numpy()
            ids_cpu = sample_ids.numpy()

            for pid, cls_idx in zip(ids_cpu, preds_cpu):
                cat_id = idx_to_cat[cls_idx]
                predictions.append({"_id": pid, "category_id": cat_id})

    # Save Submission
    df_sub = pd.DataFrame(predictions)
    # Ensure columns are in correct order
    df_sub = df_sub[["_id", "category_id"]]

    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def main():
    # Example entry point logic
    # 1. Train
    best_model_path = run_training()

    # 2. Predict
    generate_submission(best_model_path)
