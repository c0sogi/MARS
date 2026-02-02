import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import ModelEMA, seed_everything
from library.dataset import AnimalDataset, get_transforms


class MultiTaskConvNeXt(nn.Module):
    """
    Multi-Task Learning model using ConvNeXt-Small backbone.

    Heads:
    1. Species Head: Classifies into 23 categories (Empty + 22 Species).
    2. Detection Head: Binary classification (Animal Present vs Empty).
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        super(MultiTaskConvNeXt, self).__init__()

        # Load Backbone
        # num_classes=0 and global_pool='avg' ensures we get the pooled feature vector
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine input feature dimension (768 for convnext_small)
        n_features = self.backbone.num_features

        # Species Classification Head
        self.species_head = nn.Linear(n_features, Config.NUM_CLASSES)

        # Detection Auxiliary Head (Binary)
        self.detection_head = nn.Linear(n_features, Config.NUM_DETECTION_CLASSES)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Forward through heads
        species_logits = self.species_head(features)
        detection_logits = self.detection_head(features)

        return {"species_logits": species_logits, "detection_logits": detection_logits}


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Ensure alpha is on the correct device
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def calculate_class_weights(df, device):
    """
    Calculates class weights using Square-Root Inverse Frequency.
    W_c = sqrt(N_total / N_c)
    """
    class_counts = df["Category"].value_counts().sort_index()
    # Ensure all classes are represented
    counts = np.ones(Config.NUM_CLASSES)
    for cat, count in class_counts.items():
        counts[cat] = count

    total = sum(counts)
    weights = np.sqrt(total / counts)

    # Normalize weights so they sum to num_classes (optional, keeps loss scale similar)
    weights = weights / weights.mean()

    return torch.FloatTensor(weights).to(device)


def train_one_epoch(
    model, loader, optimizer, species_criterion, detection_criterion, device, ema_model
):
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        species_labels = batch["species_label"].to(device)
        detection_labels = batch["detection_label"].to(device).unsqueeze(1)  # [N, 1]

        optimizer.zero_grad()

        outputs = model(images)
        s_logits = outputs["species_logits"]
        d_logits = outputs["detection_logits"]

        # Calculate losses
        loss_s = species_criterion(s_logits, species_labels)
        loss_d = detection_criterion(d_logits, detection_labels)

        # Composite Loss
        total_loss = loss_s + (Config.LAMBDA_DETECTION * loss_d)

        total_loss.backward()
        optimizer.step()

        # Update EMA
        if ema_model:
            ema_model.update(model)

        running_loss += total_loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["species_label"].to(device)

            outputs = model(images)
            # Use species head for prediction
            preds = torch.argmax(outputs["species_logits"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate Macro F1
    score = f1_score(all_targets, all_preds, average="macro")
    return score


def train_model():
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Dataset & Dataloader
    print("Initializing Datasets...")
    # Load full training data (no sampling in production)
    train_dataset = AnimalDataset(mode="train", transform=get_transforms("train"))
    val_dataset = AnimalDataset(mode="val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Setup
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = MultiTaskConvNeXt(pretrained=True).to(device)

    # Initialize EMA
    ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # 3. Loss & Optimizer
    # Calculate class weights for Focal Loss
    class_weights = calculate_class_weights(train_dataset.df, device)

    species_criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    detection_criterion = nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_f1 = 0.0
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            species_criterion,
            detection_criterion,
            device,
            ema_model,
        )

        # Validate using EMA model for stability
        val_f1 = validate(ema_model.module, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Macro F1: {val_f1}"
        )

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ema_model.module.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! F1: {best_f1}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save EMA model state as well
    torch.save(ema_model.module.state_dict(), Config.EMA_MODEL_PATH)


def generate_submission():
    """
    Generates predictions for the test set using the best saved model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_dataset = AnimalDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = MultiTaskConvNeXt(pretrained=False).to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model from {Config.BEST_MODEL_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            "Warning: Best model not found. Using random initialization (will likely fail)."
        )

    model.eval()

    predictions = []
    ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            batch_ids = batch["id"]

            outputs = model(images)
            preds = torch.argmax(outputs["species_logits"], dim=1)

            predictions.extend(preds.cpu().numpy())
            ids.extend(batch_ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
