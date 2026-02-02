import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device
from library.dataset import AppleDataset, get_transforms, TARGET_COLS


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        """
        ResNet-18 Baseline Model for Apple Disease Detection.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use ImageNet pre-trained weights.
        """
        super(ResNet18Baseline, self).__init__()
        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        # ResNet18's fc layer input features is 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def calculate_class_weights(metadata_path, device):
    """
    Calculates class weights inversely proportional to class frequencies.
    """
    df = pd.read_csv(metadata_path)

    # We rely on 'stratify_label' or reconstruct from one-hot cols
    # The dataset.py defines TARGET_COLS order: ["healthy", "multiple_diseases", "rust", "scab"]
    # We need counts for these in this specific order.

    counts = []
    for col in TARGET_COLS:
        # Assuming metadata has these columns as binary/probabilities
        # We sum them up. If they are one-hot, sum is count.
        if col in df.columns:
            counts.append(df[col].sum())
        else:
            # Fallback if columns missing, though metadata generation script ensures they exist
            counts.append(0)

    counts = np.array(counts)
    total = counts.sum()
    n_classes = len(TARGET_COLS)

    # Formula: w_j = n_samples / (n_classes * n_samples_j)
    weights = total / (n_classes * counts + 1e-6)  # add epsilon to avoid div by zero

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_model(
    train_metadata_path="./metadata/train_metadata.csv",
    val_metadata_path="./metadata/val_metadata.csv",
    input_dir="./input",
    output_dir="./working/idea_1",
    epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    seed=42,
    max_samples=None,
):
    """
    Trains the ResNet18 model.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
    """
    seed_everything(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

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

    # Debugging: subset if requested
    if max_samples is not None:
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

    # --- Model Setup ---
    model = ResNet18Baseline(num_classes=len(TARGET_COLS), pretrained=True)
    model.to(device)

    # --- Loss & Optimizer ---
    # Calculate weights to handle imbalance (e.g. multiple_diseases is rare)
    class_weights = calculate_class_weights(train_metadata_path, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    best_auc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Apply softmax for AUC calculation
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Calculate ROC AUC (Macro Average / One-vs-Rest)
        # We need to one-hot encode labels for sklearn roc_auc_score if using multi_class='ovr'
        # Or simply pass labels as (n_samples,) and probs as (n_samples, n_classes) with multi_class='ovr'
        try:
            current_auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            # Handle cases where a class might not be present in the batch (unlikely with stratification but possible in debug)
            current_auc = 0.0

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {current_auc}"
        )

        # Checkpoint
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), best_model_path)

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
    model = ResNet18Baseline(num_classes=len(TARGET_COLS), pretrained=False)
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
    # Columns must be: image_id, healthy, multiple_diseases, rust, scab
    # TARGET_COLS order in dataset.py is ["healthy", "multiple_diseases", "rust", "scab"]
    # This matches the model output order.

    df_sub = pd.DataFrame(predictions, columns=TARGET_COLS)
    df_sub.insert(0, "image_id", image_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
