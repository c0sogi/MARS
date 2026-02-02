import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HybridLinearProbe


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, meta, targets in loader:
        images = images.to(device)
        meta = meta.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images, meta)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, meta, targets in loader:
            images = images.to(device)
            meta = meta.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, meta)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case where only one class is present in the validation set
        auc = 0.5

    return epoch_loss, auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a list of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, meta, _ in loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten().tolist())

    return all_preds


def run_training(epochs=Config.EPOCHS, patience=2):
    """
    Main function to run the training pipeline with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing DataLoaders...")
    # Use cached data if available, handled by get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine metadata dimension dynamically from a sample batch
    sample_img, sample_meta, _ = next(iter(train_loader))
    meta_dim = sample_meta.shape[1]
    print(f"Metadata Dimension: {meta_dim}")

    print("Initializing Model...")
    model = HybridLinearProbe(meta_dim=meta_dim)
    model = model.to(device)

    # Loss Function with Class Imbalance handling
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    epochs_no_improve = 0

    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training Complete. Best Val AUC: {best_auc}")

    # Load best model for inference
    print("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating Submission...")
    preds = predict(model, test_loader, device)

    # Create submission DataFrame
    # Access image names from the dataset
    image_names = test_loader.dataset.df["image_name"].values
    submission_df = pd.DataFrame({"image_name": image_names, "target": preds})

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
