import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.utils import seed_everything, get_device
from library.model import SiameseRSFNet
from library.data import get_dataset_arrays, BraTSDataset

# Configuration
CONFIG = {
    "seed": 42,
    "batch_size": 16,
    "epochs": 15,
    "lr": 1e-4,
    "patience": 4,
    "num_workers": 4,
    "metadata_dir": "./metadata",
    "input_dir": "./input",
    "working_dir": "./working",
    "submission_dir": "./submission",
    "model_path": "./working/best_model.pth",
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Iterates over the training DataLoader, performs forward pass, calculates loss,
    and updates weights. Returns average loss and AUC.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        # Unpack batch: x_even, x_odd, label
        x_even, x_odd, y = batch
        x_even = x_even.to(device)
        x_odd = x_odd.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_even, x_odd)

        # Loss calculation
        loss = criterion(logits, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Metrics tracking
        total_loss += loss.item() * y.size(0)

        # Apply sigmoid for probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    # Calculate AUC safely
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def validate(model, loader, criterion, device):
    """
    Iterates over the validation DataLoader in inference mode to calculate Loss and AUC.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_even, x_odd, y = batch
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)
            y = y.to(device)

            logits = model(x_even, x_odd)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def predict(model, loader, device):
    """
    Generates predictions for a dataset.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Test loader might return 2 or 3 items depending on if labels exist
            # Based on BraTSDataset, it returns (xe, xo) if y is None
            if len(batch) == 2:
                x_even, x_odd = batch
            else:
                x_even, x_odd, _ = batch

            x_even = x_even.to(device)
            x_odd = x_odd.to(device)

            logits = model(x_even, x_odd)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)

    return np.concatenate(all_preds)


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.
    """
    torch.save(model.state_dict(), path)


def run():
    # Setup
    seed_everything(CONFIG["seed"])
    device = get_device()
    os.makedirs(CONFIG["working_dir"], exist_ok=True)
    os.makedirs(CONFIG["submission_dir"], exist_ok=True)

    print(f"Using device: {device}")

    # 1. Data Loading
    print("Loading datasets...")
    train_meta = os.path.join(CONFIG["metadata_dir"], "train.parquet")
    val_meta = os.path.join(CONFIG["metadata_dir"], "val.parquet")
    test_meta = os.path.join(CONFIG["metadata_dir"], "test.parquet")

    # Using library.data.get_dataset_arrays which handles caching in ./working/idea_42
    X_train_e, X_train_o, y_train, _ = get_dataset_arrays(
        train_meta, "train", load_cached_data=True, input_dir=CONFIG["input_dir"]
    )
    X_val_e, X_val_o, y_val, _ = get_dataset_arrays(
        val_meta, "val", load_cached_data=True, input_dir=CONFIG["input_dir"]
    )
    X_test_e, X_test_o, _, test_ids = get_dataset_arrays(
        test_meta, "test", load_cached_data=True, input_dir=CONFIG["input_dir"]
    )

    # Create Datasets
    train_dataset = BraTSDataset(X_train_e, X_train_o, y_train)
    val_dataset = BraTSDataset(X_val_e, X_val_o, y_val)
    test_dataset = BraTSDataset(X_test_e, X_test_o, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # 2. Model Initialization
    print("Initializing model...")
    model = SiameseRSFNet(backbone_name="efficientnet_b0", pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # 3. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']} - "
            f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f}, "
            f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, CONFIG["model_path"])
            patience_counter = 0
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 4. Inference and Submission
    print("Loading best model for inference...")
    if os.path.exists(CONFIG["model_path"]):
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    else:
        print("Warning: Best model not found, using current model weights.")

    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_ids, "MGMT_value": predictions.flatten()}
    )

    submission_path = os.path.join(CONFIG["submission_dir"], "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    run()
