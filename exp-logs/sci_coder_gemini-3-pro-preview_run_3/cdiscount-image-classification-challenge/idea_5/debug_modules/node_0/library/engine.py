import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

from library import config
from library import utils
from library import dataset
from library import model as lib_model


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training using the multi-task loss.
    """
    model.train()
    total_loss = 0.0

    for features, l1, l2, l3 in dataloader:
        features = features.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        optimizer.zero_grad()

        # Forward pass through the hierarchical MLP
        out_l1, out_l2, out_l3 = model(features)

        # Calculate losses for each hierarchy level
        loss_l1 = criterion(out_l1, l1)
        loss_l2 = criterion(out_l2, l2)
        loss_l3 = criterion(out_l3, l3)

        # Weighted sum of losses
        loss = (
            (config.WEIGHT_L3 * loss_l3)
            + (config.WEIGHT_L2 * loss_l2)
            + (config.WEIGHT_L1 * loss_l1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the accuracy of the primary task (Level 3 categorization).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, _, _, l3 in dataloader:
            features = features.to(device)
            l3 = l3.to(device)

            # We only care about the Level 3 output for accuracy
            _, _, out_l3 = model(features)
            _, predicted = torch.max(out_l3, 1)

            total += l3.size(0)
            correct += (predicted == l3).sum().item()

    return correct / total if total > 0 else 0.0


def fit():
    """
    Main training loop. Initializes model, loaders, and optimizer.
    Runs training with Early Stopping and saves the best model.
    """
    set_seed(config.SEED)
    device = config.DEVICE

    # 1. Prepare Encoder to get class counts for model initialization
    encoder = utils.HierarchyEncoder()
    encoder.prepare()

    # 2. Initialize Model
    # We use the class from library.model
    model = lib_model.HierarchicalMLP(encoder.num_l1, encoder.num_l2, encoder.num_l3)
    model = model.to(device)

    # 3. Load Datasets
    # Ensure features exist
    if not os.path.exists(config.TRAIN_FEATURES_PATH):
        raise FileNotFoundError(
            f"Train features not found at {config.TRAIN_FEATURES_PATH}. Please run feature extraction first."
        )

    train_ds = dataset.EmbeddingDataset(
        config.TRAIN_FEATURES_PATH,
        config.TRAIN_LABELS_L1_PATH,
        config.TRAIN_LABELS_L2_PATH,
        config.TRAIN_LABELS_L3_PATH,
        mode="train",
    )
    val_ds = dataset.EmbeddingDataset(
        config.VAL_FEATURES_PATH,
        config.VAL_LABELS_L1_PATH,
        config.VAL_LABELS_L2_PATH,
        config.VAL_LABELS_L3_PATH,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = validate(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {avg_loss} - Val Acc L3: {val_acc}"
        )

        # Early Stopping and Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {val_acc}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    return best_val_acc


def generate_submission():
    """
    Generates the submission file using the best trained model on the test set.
    """
    set_seed(config.SEED)
    device = config.DEVICE
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No model found. Please train first.")
        return

    # Prepare Encoder
    encoder = utils.HierarchyEncoder()
    encoder.prepare()

    # Init Model & Load Weights
    model = lib_model.HierarchicalMLP(encoder.num_l1, encoder.num_l2, encoder.num_l3)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load Test Data
    if not os.path.exists(config.TEST_FEATURES_PATH):
        raise FileNotFoundError(
            "Test features not found. Please run feature extraction first."
        )

    test_ds = dataset.EmbeddingDataset(
        config.TEST_FEATURES_PATH, ids_path=config.TEST_IDS_PATH, mode="test"
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    all_ids = []
    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)

            # Predict
            _, _, out_l3 = model(features)
            _, predicted = torch.max(out_l3, 1)

            # Store results
            all_ids.extend(ids.numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Decode predictions using the encoder
    print("Decoding predictions...")
    category_ids = encoder.inverse_transform(all_preds)

    # Save to CSV
    df = pd.DataFrame({"_id": all_ids, "category_id": category_ids})
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
