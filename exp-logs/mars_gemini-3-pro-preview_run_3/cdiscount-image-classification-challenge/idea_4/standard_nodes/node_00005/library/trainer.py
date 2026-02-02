import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.model import DualStatClassifier
from library.data_loader import get_embedding_loader, get_label_mapping
from library.utils import seed_everything


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # batch is a dict: {'feature': ..., 'label': ...}
        features = batch["feature"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(train_features, train_labels, val_features, val_labels):
    """
    Main training routine.

    Args:
        train_features (np.ndarray): Training embeddings.
        train_labels (np.ndarray): Raw training category IDs.
        val_features (np.ndarray): Validation embeddings.
        val_labels (np.ndarray): Raw validation category IDs.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Training on device: {device}")

    # 1. Prepare Data
    # Map raw category IDs to indices 0..C-1
    raw_to_idx, idx_to_raw = get_label_mapping()

    # Vectorized mapping
    # We use a pandas series for fast mapping of the array
    print("Mapping labels to indices...")
    train_labels_mapped = np.array([raw_to_idx[y] for y in train_labels])
    val_labels_mapped = np.array([raw_to_idx[y] for y in val_labels])

    print(f"Train set size: {len(train_features)}")
    print(f"Val set size: {len(val_features)}")

    # Create Loaders
    train_loader = get_embedding_loader(
        train_features,
        train_labels_mapped,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = get_embedding_loader(
        val_features,
        val_labels_mapped,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = DualStatClassifier(
        input_dim=Config.INPUT_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 3. Training Loop
    best_val_acc = -1.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss}, Train Acc: {train_acc} - "
            f"Val Loss: {val_loss}, Val Acc: {val_acc}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with Val Acc: {best_val_acc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc}")


def generate_submission(test_features, test_ids):
    """
    Loads the best model, predicts on test features, and generates submission.csv.

    Args:
        test_features (np.ndarray): Test embeddings.
        test_ids (np.ndarray): Raw product IDs corresponding to the features.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Generating submission...")

    # 1. Load Model
    model = DualStatClassifier(
        input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES
    ).to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 2. Create Loader
    test_loader = get_embedding_loader(
        test_features,
        ids=test_ids,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Inference
    all_preds = []
    all_ids = []

    _, idx_to_raw = get_label_mapping()

    with torch.no_grad():
        for batch in test_loader:
            features = batch["feature"].to(device)
            ids = batch["_id"].numpy()

            outputs = model(features)
            _, predicted_indices = torch.max(outputs, 1)

            predicted_indices = predicted_indices.cpu().numpy()

            all_preds.extend(predicted_indices)
            all_ids.extend(ids)

    # 4. Map back to raw category IDs
    # Using a list comprehension for mapping
    final_categories = [idx_to_raw[idx] for idx in all_preds]

    # 5. Create DataFrame and Save
    df_submission = pd.DataFrame({"_id": all_ids, "category_id": final_categories})

    # Ensure correct types
    df_submission["_id"] = df_submission["_id"].astype(int)
    df_submission["category_id"] = df_submission["category_id"].astype(int)

    output_path = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
    df_submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path} with {len(df_submission)} records.")
