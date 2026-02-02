import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
from library.config import Config
from library.utils import set_seed, calculate_macro_f1


class AnimalModel(nn.Module):
    """
    ResNet50-based model for Partial Fine-Tuning.
    Uses concatenated Global Average Pooling (GAP) and Global Max Pooling (GMP).
    Cite solution_lesson_node_00010: Concatenate Max and Average Pooling.
    """

    def __init__(self, num_classes):
        super(AnimalModel, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base_model = models.resnet50(weights=weights)

        # Freeze all parameters first
        for param in base_model.parameters():
            param.requires_grad = False

        # Unfreeze Layer 4 for Partial Fine-Tuning
        # Cite solution_lesson_node_00015: Partial Fine-Tuning
        for param in base_model.layer4.parameters():
            param.requires_grad = True

        # Extract feature layers (up to before avgpool)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Classification head (GAP + GMP -> 4096 dim)
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)

        # Global Average Pooling
        avg_pool = torch.mean(x, dim=(2, 3))
        # Global Max Pooling
        max_pool = torch.amax(x, dim=(2, 3))

        # Concatenate
        x = torch.cat([avg_pool, max_pool], dim=1)

        return self.fc(x)

    def extract_features(self, x):
        """Helper for failure analysis to get embeddings."""
        x = self.features(x)
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        return torch.cat([avg_pool, max_pool], dim=1)


def train_model(
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    patience=3,
):
    """
    Trains the AnimalModel using AdamW and mini-batch training.
    Cite solution_lesson_node_00011: Use AdamW for mini-batch regimes.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing training on {device}...")

    # 1. Handle Class Imbalance
    # Compute weights from the dataset dataframe directly
    train_targets = train_loader.dataset.df["Category"].values
    present_classes = np.unique(train_targets)
    class_weights_subset = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=train_targets
    )
    class_weights_full = np.ones(Config.NUM_CLASSES, dtype=np.float32)
    class_weights_full[present_classes] = class_weights_subset
    class_weights = torch.tensor(class_weights_full, dtype=torch.float32).to(device)

    # 2. Initialize Model
    model = AnimalModel(Config.NUM_CLASSES).to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 4. Training Loop
    best_val_f1 = -1.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets_all = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets_all.extend(targets.numpy())

        val_f1 = calculate_macro_f1(val_targets_all, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f} | Val Macro F1: {val_f1:.6f}"
        )

        # Early Stopping Check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Cite solution_lesson_node_00003: Deep copy state dict
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. Best Val F1: {best_val_f1}")
                break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def generate_submission(model, test_loader, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Generating predictions for test set...")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(predictions)
            all_ids.extend(ids)

    # Construct DataFrame
    df = pd.DataFrame({"Id": all_ids, "Category": all_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
