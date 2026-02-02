import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import AverageMeter, get_score, set_seed
from library.dataset import get_dataloaders


class LateFusionModel(nn.Module):
    """
    Late-Fusion Time-Distributed CNN.

    Architecture:
    1. Time-Distributed ResNet18 Backbone: Extracts spatial features from each of the 6 cadence panels independently.
    2. Feature Concatenation: Stacks features from all time steps along the channel dimension.
    3. Fusion Head: A 2D Convolutional network that aggregates temporal information and detects signal drift.
    """

    def __init__(self, pretrained=True):
        super(LateFusionModel, self).__init__()

        # --- Backbone Setup ---
        # Load Backbone based on Config (Cite solution_lesson_node_00017)
        if Config.BACKBONE == "resnet34":
            base_model = models.resnet34(pretrained=pretrained)
        else:
            base_model = models.resnet18(pretrained=pretrained)

        # Modify first conv layer: (3, 64, 7, 7) -> (1, 64, 7, 7)
        self.backbone_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Initialize the new 1-channel weight by summing the original 3-channel weights.
        # This preserves the intensity magnitude better than random init or averaging.
        with torch.no_grad():
            self.backbone_conv1.weight.data = base_model.conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Extract layers up to layer4 (before avgpool and fc)
        # We want to preserve spatial dimensions for the fusion head.
        self.backbone_features = nn.Sequential(
            self.backbone_conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        # ResNet18 layer4 output channels are 512
        self.feature_dim = 512
        self.num_frames = Config.NUM_FRAMES

        # --- Fusion Head Setup ---
        # Input channels = feature_dim * num_frames (e.g., 512 * 6 = 3072)
        fusion_in_channels = self.feature_dim * self.num_frames

        self.fusion_head = nn.Sequential(
            # Reduce channels and mix temporal features spatially
            nn.Conv2d(fusion_in_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Global Average Pooling to get a single vector per sample
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # Dropout for regularization (Cite solution_lesson_node_00017)
            nn.Dropout(p=0.5),
            # Final classification
            nn.Linear(512, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Channels, Height, Width)
                              Expected: (B, 6, 1, 273, 256)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        b, t, c, h, w = x.shape

        # 1. Time-Distributed Backbone
        # Reshape to (B*T, C, H, W) to process all frames in parallel
        x = x.view(b * t, c, h, w)

        # Extract features: Output shape (B*T, 512, H', W')
        # For 273x256 input, H'~9, W'~8
        features = self.backbone_features(x)

        _, f_c, f_h, f_w = features.shape

        # 2. Reshape and Concatenate
        # Reshape back to (B, T, 512, H', W')
        features = features.view(b, t, f_c, f_h, f_w)

        # Stack time steps along the channel dimension: (B, T*512, H', W')
        features = features.view(b, t * f_c, f_h, f_w)

        # 3. Fusion Head
        logits = self.fusion_head(features)

        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Runs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Metrics
        probs = torch.sigmoid(logits)
        losses.update(loss.item(), inputs.size(0))

        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(probs.detach().cpu().numpy())

    epoch_score = get_score(all_targets, all_preds)
    return losses.avg, epoch_score


def validate(model, dataloader, criterion, device):
    """
    Runs validation.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            losses.update(loss.item(), inputs.size(0))
            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(probs.detach().cpu().numpy())

    epoch_score = get_score(all_targets, all_preds)
    return losses.avg, epoch_score


def train_and_predict():
    """
    Main function to train the model and generate submission.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # --- Data Loading ---
    train_loader, val_loader, test_loader = get_dataloaders()

    # --- Model Initialization ---
    model = LateFusionModel(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # --- Optimization ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # --- Training Loop ---
    best_score = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_PATH

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_score = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_score:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # --- Inference ---
    print("Starting inference on test set...")

    # Load best model
    model = LateFusionModel(pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions.extend(probs)

    # Create Submission DataFrame
    # We retrieve IDs from the test dataset metadata
    test_ids = test_loader.dataset.metadata["id"].values

    # Ensure lengths match (robustness check)
    if len(test_ids) != len(predictions):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of IDs ({len(test_ids)})."
        )
        # In case of drop_last=True or similar issues (though test loader usually doesn't drop)
        # We truncate to the minimum length
        min_len = min(len(test_ids), len(predictions))
        test_ids = test_ids[:min_len]
        predictions = predictions[:min_len]

    submission = pd.DataFrame({"id": test_ids, "target": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
