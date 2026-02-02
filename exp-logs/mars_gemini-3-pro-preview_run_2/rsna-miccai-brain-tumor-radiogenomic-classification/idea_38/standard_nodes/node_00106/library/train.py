import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, get_device, print_metrics
from library.data_loader import process_dataset, BraTSDataset
from library.model import AsymmetricEfficientNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # Apply sigmoid for probability calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC safely
    if len(np.unique(all_targets)) > 1:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    else:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    if len(np.unique(all_targets)) > 1:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    else:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict_tta(model, inputs, device):
    """
    Test Time Augmentation: Average of Original, HFlip, and VFlip.
    """
    # Original
    out = torch.sigmoid(model(inputs.to(device)))

    # Horizontal Flip (dim 3 is width)
    inputs_h = torch.flip(inputs, [3])
    out_h = torch.sigmoid(model(inputs_h.to(device)))

    # Vertical Flip (dim 2 is height)
    inputs_v = torch.flip(inputs, [2])
    out_v = torch.sigmoid(model(inputs_v.to(device)))

    # Average
    return (out + out_h + out_v) / 3.0


def run_training():
    """
    Main execution function for training and inference.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    print("Loading Training Data...")
    train_data, train_labels = process_dataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        "train_cache",
        load_cached_data=True,
    )

    print("Loading Validation Data...")
    val_data, val_labels = process_dataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), "val_cache", load_cached_data=True
    )

    # 3. Datasets & Loaders
    # Transform=True for training to enable geometric augmentations
    train_dataset = BraTSDataset(train_data, train_labels, transform=True)
    val_dataset = BraTSDataset(val_data, val_labels, transform=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Model & Optimization
    model = AsymmetricEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print_metrics(
            epoch + 1, Config.EPOCHS, train_loss, train_auc, val_loss, val_auc
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 6. Inference
    print("Starting Inference on Test Set...")

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Process Test Data
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    test_data, _ = process_dataset(test_csv_path, "test_cache", load_cached_data=True)

    # Dummy labels for test dataset
    test_labels = np.zeros(len(test_data))
    test_dataset = BraTSDataset(test_data, test_labels, transform=False)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            preds = predict_tta(model, inputs, device)
            predictions.extend(preds.cpu().numpy().flatten())

    # 7. Submission
    test_df = pd.read_csv(test_csv_path)
    submission = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": predictions}
    )
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
