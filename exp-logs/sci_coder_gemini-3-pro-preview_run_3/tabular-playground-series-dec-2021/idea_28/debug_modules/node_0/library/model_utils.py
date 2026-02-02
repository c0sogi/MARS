import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config

# --------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l

    This replaces element-wise multiplication with a dot-product scalar mixing
    to ensure proper global dimensional mixing.
    """

    def __init__(self, in_features):
        super(VectorCrossLayer, self).__init__()
        self.in_features = in_features
        # Weight vector w corresponding to input dimension
        self.weight = nn.Parameter(torch.Tensor(in_features))
        # Bias vector b
        self.bias = nn.Parameter(torch.Tensor(in_features))

        # Initialize parameters
        # Xavier uniform for weights to ensure variance preservation initially
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # x0: (Batch, Features) - The original input features
        # xl: (Batch, Features) - The output from the previous layer

        # Calculate dot product score: (Batch, Features) * (Features) -> Sum -> (Batch, 1)
        # This represents (x_l^T w) in the formula, resulting in a scalar per sample
        score = torch.sum(xl * self.weight, dim=1, keepdim=True)

        # Apply mixing: x0 * score + b + xl
        # The scalar score gates the original input x0
        out = (x0 * score) + self.bias + xl
        return out


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block with two linear layers.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)

    This topology maximizes gradient flow for deep tabular networks.
    """

    def __init__(self, features, dropout_rate):
        super(PreActResNetBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(features)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(features, features)

        self.bn2 = nn.BatchNorm1d(features)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(features, features)

    def forward(self, x):
        residual = x

        # First sub-block
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        # Second sub-block
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        # Residual connection
        return out + residual


class ParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet (Full Pre-Activation).

    Branch 1: Stack of VectorCrossLayers (High-order interactions).
    Branch 2: Linear Projection -> Stack of PreActResNetBlocks (Deep representation).
    Head: Concatenation -> Linear Classification Layer.
    """

    def __init__(self, input_dim, hidden_dim, num_blocks, dropout, num_classes):
        super(ParallelDCNResNet, self).__init__()

        # Branch 1: DCN
        # We use a ModuleList for the cross layers
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_blocks)]
        )

        # Branch 2: ResNet Backbone
        # Stem to project input to hidden dimension
        self.resnet_stem = nn.Linear(input_dim, hidden_dim)

        # ResNet Blocks
        self.resnet_blocks = nn.Sequential(
            *[PreActResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Combination Head
        # Concatenates DCN output (input_dim) and ResNet output (hidden_dim)
        combined_dim = input_dim + hidden_dim
        self.head = nn.Linear(combined_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        x_dcn = x
        for layer in self.cross_layers:
            # Pass x0 (original input) and xl (current state)
            x_dcn = layer(x, x_dcn)

        # Branch 2: ResNet
        x_res = self.resnet_stem(x)
        x_res = self.resnet_blocks(x_res)

        # Combine
        x_combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(x_combined)

        return logits


# --------------------------------------------------------------------------
# Training & Evaluation
# --------------------------------------------------------------------------


def train_model(train_loader, val_loader, input_dim, device):
    """
    Trains the ParallelDCNResNet model with AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    set_seed(Config.SEED)

    print(
        f"Initializing model with Input Dim: {input_dim}, Hidden Dim: {Config.HIDDEN_DIM}"
    )

    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Optimizer: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Training Phase
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

        # Validation Phase
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

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {epoch_loss:.6f}, Train Acc: {epoch_acc:.6f}, "
            f"Val Loss: {val_epoch_loss:.6f}, Val Acc: {val_epoch_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_epoch_acc)

        # Early Stopping & Checkpointing
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save immediately to disk
            torch.save(best_model_wts, Config.MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val Acc: {best_acc:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def predict_and_submit(model, test_loader, test_ids, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating predictions...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Map back to 1-7 range (model predicts 0-6)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create DataFrame
    df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved.")
