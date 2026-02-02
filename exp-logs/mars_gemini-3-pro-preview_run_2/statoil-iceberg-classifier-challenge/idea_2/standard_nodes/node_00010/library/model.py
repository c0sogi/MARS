import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import pandas as pd
from library.config import MODEL_SAVE_PATH, SUBMISSION_PATH


class SAHCN(nn.Module):
    """
    Spatially-Aware Hybrid Convolutional Network (SA-HCN).
    Consists of a CNN branch for image data and a dense branch for metadata (incidence angle),
    fused via concatenation and processed by a dense classification head.
    """

    def __init__(self, dropout_rate=0.5):
        super(SAHCN, self).__init__()

        # ==========================
        # 1. Image Branch (CNN)
        # ==========================
        # Input: (Batch, 3, 75, 75)

        # Block 1: 75x75 -> 37x37
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 37x37 -> 18x18
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 18x18 -> 9x9
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 9x9 -> 4x4
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Flatten size calculation: 64 channels * 4 * 4 spatial dims
        self.flatten_size = 64 * 4 * 4

        # ==========================
        # 2. Metadata Branch
        # ==========================
        # Input: (Batch, 1) - Incidence Angle
        self.angle_fc = nn.Linear(1, 16)
        self.angle_bn = nn.BatchNorm1d(16)

        # ==========================
        # 3. Fusion Head
        # ==========================
        fusion_input_size = self.flatten_size + 16

        self.fc1 = nn.Linear(fusion_input_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(256, 1)

    def forward(self, x_img, x_angle):
        # --- Image Branch Forward ---
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # --- Metadata Branch Forward ---
        # Ensure angle is (Batch, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)

        a = self.angle_fc(x_angle)
        a = self.angle_bn(a)
        a = F.relu(a)

        # --- Fusion ---
        combined = torch.cat((x, a), dim=1)

        # --- Dense Head Forward ---
        x = self.fc1(combined)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output logits (no sigmoid here, handled by loss function or inference)
        logits = self.output(x)
        return logits


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    patience,
    device,
    scheduler=None,
    save_path=MODEL_SAVE_PATH,
):
    """
    Trains the SAHCN model with Early Stopping and Learning Rate Scheduling.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    model = model.to(device)

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)  # Ensure (N, 1)

            optimizer.zero_grad()

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy (Sigmoid > 0.5)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == (labels > 0.5)).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                outputs = model(images, angles)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == (labels > 0.5)).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_loss:.6f}, Train Acc: {epoch_acc:.6f}, "
            f"Val Loss: {epoch_val_loss:.6f}, Val Acc: {epoch_val_acc:.6f}"
        )

        # --- Scheduler Step ---
        if scheduler:
            scheduler.step(epoch_val_loss)

        # --- Early Stopping Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Deep copy to ensure we save the actual weights, not a reference
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    # Load the best model state before returning
    if best_model_state is not None:
        print(f"Loading best model from epoch with Val Loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)

    return model


def predict_and_submit(model, test_loader, device, submission_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    model = model.to(device)

    ids = []
    probs = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            batch_ids = batch["id"]

            outputs = model(images, angles)
            # Apply Sigmoid to get probabilities [0, 1]
            batch_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids.extend(batch_ids)
            probs.extend(batch_probs)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save submission
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
