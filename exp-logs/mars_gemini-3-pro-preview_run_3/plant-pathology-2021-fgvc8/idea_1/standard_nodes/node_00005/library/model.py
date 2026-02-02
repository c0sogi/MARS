import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_f1_score


class AppleResNet18(nn.Module):
    """
    ResNet-18 model for Apple Disease Detection.
    Replaces the head for multi-label classification.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(AppleResNet18, self).__init__()

        # Load pre-trained weights if requested
        # Using IMAGENET1K_V1 as the standard default for ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # Original: Linear(in_features=512, out_features=1000)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    logits_list = []
    targets_list = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        logits_list.append(logits.detach().cpu())
        targets_list.append(targets.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    epoch_f1 = calculate_f1_score(all_logits, all_targets)

    return epoch_loss, epoch_f1


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    logits_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            logits_list.append(logits.detach().cpu())
            targets_list.append(targets.detach().cpu())

    val_loss = running_loss / len(loader.dataset)
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    val_f1 = calculate_f1_score(all_logits, all_targets)

    return val_loss, val_f1


def train_model(train_loader, val_loader, epochs=Config.EPOCHS, device=Config.DEVICE):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    model = AppleResNet18(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Multi-label classification: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Gradient Scaler for AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_f1 = -1.0
    patience = 3
    patience_counter = 0
    best_model_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss}, Train F1: {train_f1}, "
            f"Val Loss: {val_loss}, Val F1: {val_f1}"
        )

        scheduler.step()

        # Checkpointing and Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} with F1: {best_f1}")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    return model


def generate_submission(model, test_loader, device=Config.DEVICE):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    print("Generating submission...")

    with torch.no_grad():
        for images, _, image_ids in test_loader:
            images = images.to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(images)

            # Apply Sigmoid and Threshold
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).int().cpu().numpy()

            for i, img_id in enumerate(image_ids):
                # Map binary predictions to class labels
                pred_indices = np.where(preds[i] == 1)[0]
                pred_labels = [Config.CLASS_LABELS[idx] for idx in pred_indices]

                label_str = " ".join(pred_labels)
                results.append({"image": img_id, "labels": label_str})

    df = pd.DataFrame(results)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
