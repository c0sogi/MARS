import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import set_seed, get_device
from library.data_loader import get_dataloader
from library.model import AsymmetricEfficientNet

# Constants
WORKING_DIR = "./working/idea_14"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def train_one_epoch(model, loader, optimizer, criterion, device):
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

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge cases where a batch might have only one class
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
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
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # Handle loader output (tuple if targets exist, tensor if not)
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)

            # 1. Original
            out_orig = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip (dim 3: W)
            inputs_h = inputs.flip(3)
            out_h = torch.sigmoid(model(inputs_h))

            # 3. Vertical Flip (dim 2: H)
            inputs_v = inputs.flip(2)
            out_v = torch.sigmoid(model(inputs_v))

            # Average
            avg_probs = (out_orig + out_h + out_v) / 3.0
            avg_probs = avg_probs.cpu().numpy().flatten()

            if len(all_probs) == 0:
                all_probs = avg_probs
            else:
                all_probs = np.concatenate((all_probs, avg_probs))

    return all_probs


def run_training(
    epochs=20,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-2,
    patience=5,
    seed=42,
    load_cached_data=True,
):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")

    # Initialize Loaders
    train_loader = get_dataloader(
        train_df,
        root_dir="./input",
        phase="train",
        batch_size=batch_size,
        load_cached_data=load_cached_data,
    )
    val_loader = get_dataloader(
        val_df,
        root_dir="./input",
        phase="val",
        batch_size=batch_size,
        load_cached_data=load_cached_data,
    )

    # Initialize Model
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.2, pretrained=True)
    model = model.to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    # Training State
    best_auc = 0.0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.10f} | Train AUC: {train_auc:.10f} | "
            f"Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  New best model saved! AUC: {best_auc:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    return best_auc


def generate_submission(batch_size=32, load_cached_data=True):
    """
    Generates the submission file using the best trained model and TTA.
    """
    device = get_device()
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No model found. Please run training first.")
        return

    # Load Test Data
    test_df = pd.read_csv("./metadata/test.csv")
    test_loader = get_dataloader(
        test_df,
        root_dir="./input",
        phase="test",
        batch_size=batch_size,
        load_cached_data=load_cached_data,
    )

    # Load Model
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.2, pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)

    # Predict
    print("Generating predictions with TTA...")
    probs = predict_tta(model, test_loader, device)

    # Format Submission
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": probs}
    )

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission_df.head())
