import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from library.dataset import get_dataloaders
from library.utils import seed_everything, quadratic_weighted_kappa


class RetinopathyModel(nn.Module):
    def __init__(self, pretrained=True):
        super(RetinopathyModel, self).__init__()
        # Load EfficientNet-B5 backbone
        # We use B5 because the analysis showed resolution/fine-features are critical
        weights = EfficientNet_B5_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b5(weights=weights)

        # Modify the classifier for regression
        # The default classifier is Sequential(Dropout, Linear)
        # We replace it to output a single scalar (regression)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Store predictions for metrics
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate QWK on training data
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Clip and round for QWK calculation
    preds_rounded = np.round(np.clip(all_preds, 0, 4)).astype(int)
    labels_int = all_labels.astype(int)
    qwk = quadratic_weighted_kappa(labels_int, preds_rounded)

    return epoch_loss, qwk


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images).view(-1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    preds_rounded = np.round(np.clip(all_preds, 0, 4)).astype(int)
    labels_int = all_labels.astype(int)
    qwk = quadratic_weighted_kappa(labels_int, preds_rounded)

    return epoch_loss, qwk


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    model.eval()
    test_preds = []

    # Predict
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images).view(-1)
            test_preds.append(outputs.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    # Post-processing: Clip to valid range and round to nearest integer
    final_preds = np.round(np.clip(test_preds, 0, 4)).astype(int)

    # Load metadata for IDs
    df_test = pd.read_csv("./metadata/test.csv")

    submission = pd.DataFrame({"id_code": df_test["id_code"], "diagnosis": final_preds})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main(epochs=15, batch_size=16, patience=5, learning_rate=1e-4):
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Data
    # Using 512x512 resolution as per strategy to capture fine details
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, image_size=512, load_cached_data=True
    )

    # Initialize Model
    model = RetinopathyModel(pretrained=True)
    model.to(device)

    # Setup Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    # Scheduler: Monitor QWK (maximize)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    # Training Loop
    best_qwk = -np.inf
    epochs_no_improve = 0
    save_dir = "./working/idea_3"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_qwk = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        duration = time.time() - start

        # Print metrics (full precision)
        print(f"Epoch {epoch+1}/{epochs} [{duration:.2f}s]")
        print(f"Train Loss: {train_loss} QWK: {train_qwk}")
        print(f"Val Loss: {val_loss} QWK: {val_qwk}")

        # Scheduler Step
        scheduler.step(val_qwk)

        # Early Stopping & Checkpointing
        if val_qwk > best_qwk:
            print(f"Validation QWK improved ({best_qwk} -> {val_qwk}). Saving model...")
            best_qwk = val_qwk
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device)
