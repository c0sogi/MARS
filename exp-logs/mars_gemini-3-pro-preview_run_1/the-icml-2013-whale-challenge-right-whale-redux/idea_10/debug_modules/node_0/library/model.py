import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import set_seed, compute_auc
from library.dataset import get_dataloaders
from library.csk_resnet import CSKResNet18CRNN


class WhaleCRNN(CSKResNet18CRNN):
    """
    Wrapper for the Time-Preserving CSK-ResNet-18 CRNN architecture.
    Inherits from the library implementation to satisfy the requirement
    of implementing the WhaleCRNN class while utilizing the provided components.
    """

    def __init__(self):
        super(WhaleCRNN, self).__init__()


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # Apply Mixup
        data, target_a, target_b, lam = mixup_data(
            data, target, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()
        output = model(data)

        # Reshape targets for BCEWithLogitsLoss: (N, 1)
        target_a = target_a.view(-1, 1)
        target_b = target_b.view(-1, 1)

        loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns: Average Loss, AUC
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            target_view = target.view(-1, 1)

            loss = criterion(output, target_view)
            running_loss += loss.item() * data.size(0)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(output).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    auc = compute_auc(all_targets, all_preds)

    return avg_loss, auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns: List of clip IDs, List of probabilities
    """
    model.eval()
    predictions = []
    clips = []

    with torch.no_grad():
        for data, clip_ids in loader:
            data = data.to(device)
            output = model(data)
            preds = torch.sigmoid(output).cpu().numpy()

            predictions.extend(preds.flatten())
            clips.extend(clip_ids)

    return clips, predictions


def run():
    """
    Main execution function:
    1. Setup environment
    2. Load data
    3. Train model with Early Stopping
    4. Generate Submission
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print(f"Initializing {Config.MODEL_NAME} (WhaleCRNN)...")
    model = WhaleCRNN().to(device)

    # 4. Optimization Setup
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Loss Function: BCEWithLogitsLoss with Positive Class Weight
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience = 5
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f} - Val AUC: {val_auc:.8f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best Validation AUC: {best_auc:.8f}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions on test set...")
    test_clips, test_probs = predict(model, test_loader, device)

    # 7. Save Submission
    submission_df = pd.DataFrame({"clip": test_clips, "probability": test_probs})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute the pipeline
run()
