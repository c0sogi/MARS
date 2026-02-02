import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.config import Config

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    where (.) denotes the dot product resulting in a scalar.
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim
        # Weight vector w: (input_dim,)
        self.weight = nn.Parameter(torch.empty(input_dim))
        # Bias vector b: (input_dim,)
        self.bias = nn.Parameter(torch.empty(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights to be small to start near identity behavior
        nn.init.normal_(self.weight, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # x0: (batch, input_dim) - Initial input
        # xl: (batch, input_dim) - Current layer input

        # Compute dot product (xl . w) -> scalar per sample
        # (batch, input_dim) * (input_dim) -> sum(dim=1) -> (batch, 1)
        dot_prod = (xl * self.weight).sum(dim=1, keepdim=True)

        # Apply formula: x0 * scalar + b + xl
        # The broadcasting handles the scalar multiplication across the vector x0
        out = x0 * dot_prod + self.bias + xl
        return out


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear
    Includes a residual connection.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(PreActResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # First sub-block
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        # Second sub-block
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        # Residual connection
        return x + out


class ParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet (Full Pre-Activation).
    Combines a Vector DCN branch and a Deep ResNet branch.
    """

    def __init__(
        self, input_dim, num_classes, hidden_dim=512, num_blocks=4, dropout=0.2
    ):
        super(ParallelDCNResNet, self).__init__()

        # --- Branch 1: Vector-Based DCN ---
        # We use a stack of VectorCrossLayers.
        # Depth matches num_blocks to maintain balanced capacity.
        self.num_cross_layers = num_blocks
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(self.num_cross_layers)]
        )

        # --- Branch 2: Deep Full Pre-Activation ResNet ---
        # Stem to project input to hidden_dim
        self.resnet_stem = nn.Linear(input_dim, hidden_dim)

        # ResNet Blocks
        self.resnet_blocks = nn.Sequential(
            *[PreActResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # --- Combination Head ---
        # Concatenate DCN output (input_dim) + ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.classifier = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)

        # Branch 1: DCN
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        dcn_out = xl

        # Branch 2: ResNet
        resnet_out = self.resnet_stem(x)
        resnet_out = self.resnet_blocks(resnet_out)

        # Concatenate
        combined = torch.cat([dcn_out, resnet_out], dim=1)

        # Classification
        logits = self.classifier(combined)
        return logits


def train_model(
    train_loader, val_loader, input_dim, num_classes=7, epochs=Config.EPOCHS
):
    """
    Executes the training pipeline with AdamW, ReduceLROnPlateau, and Early Stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        input_dim: Integer, number of input features.
        num_classes: Integer, number of target classes.
        epochs: Integer, maximum number of training epochs.

    Returns:
        model: The trained PyTorch model with best weights loaded.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    criterion = nn.CrossEntropyLoss()

    # Early Stopping State
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_loss} Acc: {epoch_acc} | "
            f"Val Loss: {val_loss} Acc: {val_acc}"
        )

        # Update Scheduler
        scheduler.step(val_acc)

        # --- Early Stopping Check ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint
            torch.save(best_model_wts, Config.MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model


def predict_and_submit(model, test_loader, test_ids):
    """
    Generates predictions on the test set and saves the submission CSV.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        test_ids: Array of test IDs corresponding to the loader data.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    predictions = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Map 0-6 back to 1-7 (Original Class Labels)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create submission DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
