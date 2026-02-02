import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config, seed_everything
from library.models import get_detection_model
from library.data import get_dataloaders, get_label_map


def train_detector(debug=False, epochs=Config.DET_EPOCHS):
    """
    Trains the Faster R-CNN Detection Model.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting Detection Training on {device}...")

    # 1. Data
    train_loader = get_dataloaders("detection", "train", debug=debug)
    # Validation for detection is complex (mAP), we'll skip formal val loop
    # during training to save time and rely on the final validation step.

    # 2. Model
    # Get num_classes (Background + Actual Classes)
    label2id, _ = get_label_map()
    num_classes = len(label2id) + 1
    print(f"Training with {num_classes} classes (including background).")

    model = get_detection_model(num_classes).to(device)

    # 3. Optimization
    # SGD is standard for R-CNNs
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=Config.DET_LR,
        momentum=Config.DET_MOMENTUM,
        weight_decay=Config.DET_WEIGHT_DECAY,
    )

    # MultiStepLR as per Lesson 5
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 11], gamma=0.1
    )

    # 4. Training Loop
    save_path = os.path.join(Config.CACHE_DIR, "det_model.pth")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Faster R-CNN returns a dict of losses during training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']}"
        )

        # Save checkpoint every epoch
        torch.save(model.state_dict(), save_path)

    print("Detection training complete.")
    return save_path
