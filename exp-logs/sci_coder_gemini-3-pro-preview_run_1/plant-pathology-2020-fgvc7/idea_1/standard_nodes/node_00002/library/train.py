import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device
from library.dataset import AppleDataset, get_transforms, TARGET_COLS
from library.model import ResNet34Baseline


def calculate_class_weights(metadata_path, device):
    """
    Calculates class weights inversely proportional to class frequencies.
    """
    df = pd.read_csv(metadata_path)
    counts = []
    for col in TARGET_COLS:
        if col in df.columns:
            counts.append(df[col].sum())
        else:
            counts.append(0)

    counts = np.array(counts)
    n_classes = len(TARGET_COLS)
    total = counts.sum()

    # Formula: w_j = n_samples / (n_classes * n_samples_j)
    # Add epsilon to prevent division by zero
    weights = total / (n_classes * counts + 1e-6)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size if dataset_size > 0 else 0.0


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Calculate ROC AUC
        # multi_class='ovr' handles the multiclass case
        try:
            # We need to ensure we have samples for all classes in validation for robust AUC
            # However, sklearn handles missing classes in y_true gracefully usually,
            # but here we pass labels (indices) and probs.
            # We use one-hot encoding for labels implicitly required by some versions or
            # pass 1D labels with multi_class='ovr'.
            # Ideally, we should one-hot encode y_true for roc_auc_score with multi_class='ovr'
            # if y_score is shape (n_samples, n_classes).

            # One-hot encode labels for sklearn
            n_classes = len(TARGET_COLS)
            y_true_one_hot = np.eye(n_classes)[all_labels]

            auc_score = roc_auc_score(
                y_true_one_hot, all_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc_score = 0.0
    else:
        auc_score = 0.0

    return avg_loss, auc_score


def train_model(
    train_metadata_path="./metadata/train_metadata.csv",
    val_metadata_path="./metadata/val_metadata.csv",
    input_dir="./input",
    output_dir="./working/idea_1",
    epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    seed=42,
    patience=5,
    max_samples=None,
):
    """
    Main function to train the model with Early Stopping.
    """
    seed_everything(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pth")

    # --- Data Loading ---
    train_dataset = AppleDataset(
        metadata_path=train_metadata_path,
        transform=get_transforms("train", image_size=256),
        input_dir=input_dir,
        mode="train",
    )

    val_dataset = AppleDataset(
        metadata_path=val_metadata_path,
        transform=get_transforms("val", image_size=256),
        input_dir=input_dir,
        mode="val",
    )

    if max_samples:
        train_dataset.df = train_dataset.df.iloc[:max_samples]
        val_dataset.df = val_dataset.df.iloc[:max_samples]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- Model & Training Setup ---
    model = ResNet34Baseline(num_classes=len(TARGET_COLS), pretrained=True)
    model.to(device)

    class_weights = calculate_class_weights(train_metadata_path, device)
    # Cite solution_lesson_node_00001: Inverse Class Frequency Weighting for Macro-Averaged AUC
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs, eta_min=1e-6
    )

    # --- Training Loop with Early Stopping ---
    best_auc = -1.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_model_path


def predict_and_submit(
    model_path,
    test_metadata_path="./metadata/test_metadata.csv",
    input_dir="./input",
    output_path="./submission/submission.csv",
    batch_size=32,
    device=None,
):
    """
    Generates predictions for the test set and saves the submission file.
    """
    if device is None:
        device = get_device()

    # Load Model
    model = ResNet34Baseline(num_classes=len(TARGET_COLS), pretrained=False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Test Data
    test_dataset = AppleDataset(
        metadata_path=test_metadata_path,
        transform=get_transforms("test", image_size=256),
        input_dir=input_dir,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            predictions.append(probs.cpu().numpy())
            image_ids.extend(ids)

    predictions = np.concatenate(predictions)

    # Create Submission DataFrame
    df_sub = pd.DataFrame(predictions, columns=TARGET_COLS)
    df_sub.insert(0, "image_id", image_ids)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
