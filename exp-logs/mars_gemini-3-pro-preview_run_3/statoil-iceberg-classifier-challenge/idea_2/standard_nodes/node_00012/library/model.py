import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    N_FOLDS,
    SUBMISSION_FILE,
    WORKING_DIR,
    BACKBONE_OUT_DIM,
    FUSION_DIM,
    SEED,
)
from library.utils import AverageMeter, save_checkpoint, seed_everything
from library.data_loader import load_and_process_data, get_fold_loaders, get_test_loader


class IcebergCNN(nn.Module):
    """
    Custom 4-layer CNN for Iceberg detection on 75x75 images.
    Cite solution_lesson_node_00009: Custom Architectures vs Pre-trained Models
    """

    def __init__(self, dropout_rate=0.5):
        super(IcebergCNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 4
        # Cite solution_lesson_node_00006: Channel Width Bottlenecks (keep 128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(FUSION_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, angle):
        # x: [Batch, 3, 75, 75]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global Max Pooling
        # Cite solution_lesson_node_00005: Global Pooling
        # Cite solution_lesson_node_00007: Max Pooling vs Average Pooling
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # -> [Batch, 128]

        # Process Angle
        angle = angle.view(-1, 1)  # -> [Batch, 1]

        # Feature Fusion
        x = torch.cat([x, angle], dim=1)  # -> [Batch, 129]

        # Classification
        x = self.classifier(x)

        return x


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for inputs, angles, targets in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for inputs, angles, targets in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

    return losses.avg


def predict(loader, model, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs, angles in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            outputs = model(inputs, angles)
            # Move to CPU and convert to numpy
            preds.extend(outputs.cpu().numpy().flatten())

    return np.array(preds)


def run_training():
    """
    Main execution function:
    1. Loads data
    2. Runs 5-Fold CV training
    3. Generates predictions on test set
    4. Saves submission file
    """
    seed_everything(SEED)
    print(f"Starting training on device: {DEVICE}")

    # 1. Load Data
    # load_cached_data=True will try to load from working/idea_2, or process from input/
    X_train, y_train, angles_train, X_test, ids_test, angles_test = (
        load_and_process_data(load_cached_data=True)
    )

    # Prepare Test Loader
    test_loader = get_test_loader(X_test, angles_test, batch_size=BATCH_SIZE)

    # Array to accumulate predictions from each fold
    test_preds_accum = np.zeros(len(X_test))

    # 2. Cross-Validation Loop
    for fold in range(N_FOLDS):
        print(f"\n{'='*20}")
        print(f"Fold {fold + 1} / {N_FOLDS}")
        print(f"{'='*20}")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_fold_loaders(
            fold, X_train, y_train, angles_train, batch_size=BATCH_SIZE
        )

        # Initialize Model
        model = IcebergCNN(dropout_rate=0.5).to(DEVICE)

        # Loss and Optimizer
        # Using BCELoss because model outputs Sigmoid
        criterion = nn.BCELoss()
        # Optimize only classifier parameters (backbone is frozen)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

        # Training Loop Variables
        best_val_loss = float("inf")
        patience_counter = 0
        fold_checkpoint_dir = os.path.join(WORKING_DIR, f"fold_{fold}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, DEVICE
            )
            val_loss = validate(val_loader, model, criterion, DEVICE)

            print(
                f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=True,
                    checkpoint_dir=fold_checkpoint_dir,
                )
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # 3. Inference for this fold
        print(f"Loading best model for Fold {fold + 1} and predicting...")
        best_model_path = os.path.join(fold_checkpoint_dir, "model_best.pth")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        fold_preds = predict(test_loader, model, DEVICE)
        test_preds_accum += fold_preds

    # 4. Average Predictions and Submit
    avg_preds = test_preds_accum / N_FOLDS

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    # Save
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission saved successfully to: {SUBMISSION_FILE}")
    print(df_sub.head())
