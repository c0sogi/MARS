import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config, seed_everything
from library.models import get_detection_model
from library.data import get_dataloaders


def train_detector(debug=False, epochs=Config.DET_EPOCHS):
    """
    Trains the Object Detection Model.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting Detection Training on {device}...")

    # 1. Data
    train_loader = get_dataloaders("detection", "train", debug=debug)
    # Note: Faster R-CNN computes loss only in train mode.
    # For validation, we typically look at mAP, but here we will rely on the
    # final F1 metric calculation in runfile.py to save time and complexity.

    # Determine number of classes
    # +1 for background class
    num_classes = len(train_loader.dataset.label2id) + 1
    print(f"Number of classes (including background): {num_classes}")

    # 2. Model
    model = get_detection_model(num_classes).to(device)

    # 3. Optimization
    # SGD is standard for detection models
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=Config.DET_LR,
        momentum=Config.DET_MOMENTUM,
        weight_decay=Config.DET_WEIGHT_DECAY,
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. Training Loop
    save_path = os.path.join(Config.CACHE_DIR, "det_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()

        # Average loss per batch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        torch.save(model.state_dict(), save_path)

    print("Detection training complete.")
    return save_path
